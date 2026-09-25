"""Main window: the layer panel, the canvas, the extract page, and the shortcuts."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import override

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QWidget,
)

from pixelsb.domain.commands import CommandError, parse, text_of
from pixelsb.domain.geometry import SLIDER_STEPS, initial_zoom, slider_position, slider_zoom
from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    BitOrder,
    DisplayFormat,
    ExtractEncoding,
    LoadedImage,
    Mask,
    PixelCoord,
    Raster,
    RegionMask,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.samples import render_export
from pixelsb.domain.stack import resolve
from pixelsb.domain.transitions import (
    add_layer,
    add_mask_text,
    bits_position,
    clear_layers,
    cycle_format,
    filter_position,
    move_cursor,
    move_layer,
    open_image,
    remove_layer,
    select_all_bits,
    select_lsb,
    select_lsb_at,
    select_lsbs,
    select_only,
    set_bit_order,
    set_channel,
    set_channel_order,
    set_column,
    set_cursor,
    set_extract_encoding,
    set_format,
    set_frame,
    set_layer_enabled,
    set_mask_text,
    set_scan_order,
    set_zoom,
    step_channel,
    step_focus_bit,
    step_plane,
    toggle_bit,
)
from pixelsb.io.loading import ImageLoadError, load_frame, load_image
from pixelsb.io.writing import save_image
from pixelsb.ui import text, theme
from pixelsb.ui.canvas import CanvasMode, ImageCanvas
from pixelsb.ui.controls import RETURN_KEYS
from pixelsb.ui.extract_panel import ExtractPanel
from pixelsb.ui.info import InfoPanel
from pixelsb.ui.layers import LayerPanel
from pixelsb.ui.painting import lighten_clear_button
from pixelsb.ui.side_panels import Panel, SidePanels
from pixelsb.ui.store import Store, Transition
from pixelsb.ui.text import readout_text, status_info, status_view

type ErrorReporter = Callable[[str], None]

_FILTER_ERROR_STYLE = f"QLineEdit {{ border: 1px solid {theme.DANGER}; }}"
_TEXT_INPUTS = (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit, QComboBox)
_CANVAS_MIN_WIDTH = 260
_LEFT_MIN_WIDTH = 240
_LEFT_WIDTH = 288
_NUDGE_STEP = 8  # a shift-arrow moves this many pixels instead of one
_FORMAT_ITEMS = tuple(
    (fmt.value, label)
    for fmt, label in (
        (DisplayFormat.DECIMAL, text.DECIMAL),
        (DisplayFormat.HEX, text.HEX),
        (DisplayFormat.BINARY, text.BINARY),
    )
)
# Keys the canvas never sees: what each one does is the whole table's story.
_ARROW_KEYS = {
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
}
_FOCUS_BIT_KEYS = {Qt.Key.Key_BracketLeft: -1, Qt.Key.Key_BracketRight: 1}
_ZOOM_KEYS = {Qt.Key.Key_Plus: 1, Qt.Key.Key_Equal: 1, Qt.Key.Key_Minus: -1}
_COMMAND_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.MetaModifier
    | Qt.KeyboardModifier.AltModifier
)
_CHANNEL_LETTERS = ("R", "G", "B", "A", "L")
_ORDERED_LETTERS = tuple(str(index) for index in range(1, 10))


class MainWindow(QMainWindow):
    def __init__(self, reporter: ErrorReporter | None = None) -> None:
        super().__init__()
        self.store = Store()
        self._reporter = reporter or self._report_with_dialog
        self._raster: Raster | None = None
        self._raster_key: object = None
        self._command_error = ""
        self._box_edit_key: object = None  # the state the box's own line already covers
        self._box_authoring = False
        self._panel_sized = False
        self.setWindowTitle(text.APP_NAME)
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        self._build_actions()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        application = _application()
        if application is not None:
            application.installEventFilter(self)
        self.store.subscribe(self._apply)
        self._apply(self.store.state)

    def apply(self, transition: Transition) -> None:
        self.store.update(transition)

    def open_path(self, path: Path) -> None:
        try:
            image = load_image(path)
        except ImageLoadError as exc:
            self._reporter(text.open_failed(str(exc)))
            return
        self.open_loaded(image)

    def open_loaded(self, image: LoadedImage) -> None:
        """Show an already-decoded image, fitted and from the top left."""
        zoom = self._fit_zoom((image.width, image.height))
        self.apply(partial(open_image, image=image, zoom=zoom))
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
        QTimer.singleShot(0, self._fit)

    @override
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and watched is self._filter_edit
            and event.key() in RETURN_KEYS
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            # Taken here rather than in returnPressed: the box reports a plain
            # Enter, and the shift is what says to stack the line instead of
            # writing over the operation in hand.
            self._commit_filter(new=True)
            return True
        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and watched is self._filter_edit
            and event.key() == Qt.Key.Key_Escape
        ):
            self._clear_filter()
            return True
        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and not isinstance(watched, _TEXT_INPUTS)
            and self._should_handle_keys()
            and self._handle_key(event)
        ):
            return True
        if watched is self._scroll.viewport():
            if isinstance(event, QDropEvent) and event.type() == QEvent.Type.Drop:
                self._open_dropped(event)
                return True
            if isinstance(event, QDragMoveEvent) and event.type() in {
                QEvent.Type.DragEnter,
                QEvent.Type.DragMove,
            }:
                self._accept_file_drag(event)
                return True
        return super().eventFilter(watched, event)

    @override
    def showEvent(self, event: QShowEvent) -> None:
        if not self._panel_sized:
            self._panel_sized = True
            self._size_panel()
        super().showEvent(event)

    def _size_panel(self) -> None:
        """Give the right panel the width its pages' content needs, once."""
        width = self.width()
        handles = 2 * self._splitter.handleWidth()
        # Both pages hold a two-pane dump; the wider one decides the panel.
        wanted = max(self.extract_panel.preferred_width(), self.info_panel.preferred_width())
        limit = max(width - handles - _LEFT_WIDTH - _CANVAS_MIN_WIDTH, 1)
        panel = min(max(wanted, self.extract_panel.minimumWidth()), limit)
        canvas = max(width - panel - _LEFT_WIDTH - handles, 1)
        self._splitter.setSizes([_LEFT_WIDTH, canvas, panel])

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        application = _application()
        if application is not None:
            application.removeEventFilter(self)
        super().closeEvent(event)

    @override
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self._accept_file_drag(event)

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        self._open_dropped(event)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu(text.FILE_MENU)
        open_action = file_menu.addAction(text.OPEN)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(_drop_checked(self._open_dialog))
        file_menu.addSeparator()
        info_action = file_menu.addAction(text.FILE_INFO)
        info_action.triggered.connect(_drop_checked(self._show_info))
        file_menu.addSeparator()
        quit_action = file_menu.addAction(text.QUIT)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(_drop_checked(self.close))

        view_menu = self.menuBar().addMenu(text.VIEW_MENU)
        self._original_action = view_menu.addAction(text.ORIGINAL)
        self._original_action.triggered.connect(_drop_checked(partial(self.apply, clear_layers)))
        self._lsb_action = view_menu.addAction(text.ALL_LSB)
        self._lsb_action.triggered.connect(_drop_checked(partial(self.apply, select_lsbs)))

        help_menu = self.menuBar().addMenu(text.HELP_MENU)
        help_action = help_menu.addAction(text.SHORTCUTS)
        help_action.triggered.connect(_drop_checked(self._show_shortcuts))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("view")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(self._toolbar_row())

    def _toolbar_row(self) -> QFrame:
        """One row: the display filter on the left, the view controls on the right."""
        row = _row_frame("toolbarRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)
        self._export_button = self._ghost_button(text.VIEW_EXPORT, 0, self._export_view)
        self._export_button.setToolTip(text.VIEW_EXPORT_TIP)
        self._export_button.setFixedHeight(theme.CONTROL_HEIGHT)
        widgets = (
            *self._filter_widgets(),
            *self._mode_widgets(),
            *self._zoom_widgets(),
            self._export_button,
        )
        for widget in widgets:
            layout.addWidget(widget)
        # The filter box takes whatever width the controls to its right leave.
        layout.setStretch(0, 1)
        return row

    def _filter_widgets(self) -> tuple[QWidget, ...]:
        """The command box and the pass count beside it.

        A line is applied when it is asked for — Enter, or Esc to drop the mask
        — never as it is typed: a command can cost a full redraw of the picture,
        and a sentence half written is not a line to run.
        """
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(text.FILTER_PLACEHOLDER)
        self._filter_edit.setClearButtonEnabled(True)
        lighten_clear_button(self._filter_edit)
        self._filter_edit.setMinimumWidth(220)
        self._filter_edit.setFixedHeight(theme.CONTROL_HEIGHT)
        self._filter_edit.setToolTip(text.FILTER_TIP)
        self._filter_edit.textEdited.connect(lambda _text: self._clear_command_error())
        self._filter_edit.returnPressed.connect(self._commit_filter)
        find_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        find_shortcut.activated.connect(self._focus_filter)
        self._filter_count = _muted_label()
        return self._filter_edit, self._filter_count

    def _mode_widgets(self) -> tuple[QWidget, ...]:
        """The exclusive 移动 / 选区 pair."""
        self._mode_move = QPushButton(text.MODE_MOVE)
        self._mode_select = QPushButton(text.MODE_SELECT)
        for button, tip in (
            (self._mode_move, text.MODE_MOVE_TIP),
            (self._mode_select, text.MODE_SELECT_TIP),
        ):
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setFixedHeight(theme.CONTROL_HEIGHT)
        self._mode_move.setObjectName("segmentLeft")
        self._mode_select.setObjectName("segmentRight")
        self._mode_move.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_move)
        self._mode_group.addButton(self._mode_select)
        self._mode_group.setExclusive(True)
        self._mode_group.buttonClicked.connect(self._on_mode)
        return self._mode_move, self._mode_select

    def _zoom_widgets(self) -> tuple[QWidget, ...]:
        """The zoom steppers, the slider, the readout, and the format switch."""
        self._zoom_in = self._ghost_button(text.ZOOM_IN, 28, partial(self._zoom_by, 1))
        self._zoom_out = self._ghost_button(text.ZOOM_OUT, 28, partial(self._zoom_by, -1))
        self._zoom_fit = self._ghost_button(text.ZOOM_FIT, 0, self._fit)
        self._zoom_reset = self._ghost_button(
            text.ZOOM_RESET, 0, partial(self._zoom_to, float(MIN_ZOOM))
        )
        self._zoom_reset.setToolTip(text.ZOOM_RESET_TIP)
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(0, SLIDER_STEPS)
        self._zoom_slider.setFixedWidth(160)
        self._zoom_slider.setToolTip(text.ZOOM_TIP)
        self._zoom_slider.valueChanged.connect(self._on_slider)
        self._zoom_label = _muted_label()
        self._zoom_label.setObjectName("zoomLabel")
        self._zoom_label.setFixedWidth(
            52
        )  # wide enough for the longest label, e.g. 123.46 (times sign)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._format_combo = _combo(text.FORMAT_TIP)
        for value, label in _FORMAT_ITEMS:
            self._format_combo.addItem(label, value)
        self._format_combo.currentIndexChanged.connect(self._on_format)
        widgets = (
            self._zoom_out,
            self._zoom_slider,
            self._zoom_in,
            self._zoom_label,
            self._zoom_fit,
            self._zoom_reset,
            self._format_combo,
        )
        for widget in widgets:
            widget.setFixedHeight(theme.CONTROL_HEIGHT)
        return widgets

    def _ghost_button(self, label: str, width: int, slot: Callable[[], None]) -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("ghost")
        if width:
            button.setFixedWidth(width)
        button.clicked.connect(_drop_checked(slot))
        return button

    def _build_body(self) -> None:
        self._scroll = QScrollArea()
        self._scroll.setObjectName("canvasArea")
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas = ImageCanvas(self._scroll)
        self._scroll.setWidget(self.canvas)
        self._scroll.viewport().setAcceptDrops(True)
        self._scroll.viewport().installEventFilter(self)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.zoom_requested.connect(self._zoom_by)
        self.canvas.zoom_scale_requested.connect(self._zoom_scale)
        self.canvas.file_dropped.connect(lambda path: self.open_path(Path(path)))
        self.canvas.region_selecting.connect(self._on_region_selecting)
        self.canvas.region_selected.connect(self._on_region_selected)
        self.canvas.region_committed.connect(self._on_region_committed)
        self.canvas.region_canceled.connect(self._on_region_canceled)

        self.extract_panel = ExtractPanel()
        self.extract_panel.bit_clicked.connect(self._on_bit)
        self.extract_panel.channel_toggle.connect(self._on_channel_toggle)
        self.extract_panel.column_toggle.connect(self._on_column_toggle)
        self.extract_panel.original_requested.connect(self._on_all_bits)
        self.extract_panel.lsbs_requested.connect(self._on_lsbs)
        self.extract_panel.plane_step.connect(self._on_plane_step)
        self.extract_panel.channel_step.connect(self._on_channel_step)
        self.extract_panel.encoding_requested.connect(self._on_encoding)
        self.extract_panel.channel_order_requested.connect(self._on_channel_order)
        self.extract_panel.bit_order_requested.connect(self._on_bit_order)
        self.extract_panel.scan_requested.connect(self._on_scan_order)
        self.extract_panel.save_requested.connect(self._save_extract)

        self.layers_panel = LayerPanel()
        self.layers_panel.add_requested.connect(self._on_layer_added)
        self.layers_panel.remove_requested.connect(self._on_layer_removed)
        self.layers_panel.move_requested.connect(self._on_layer_moved)
        self.layers_panel.enabled_requested.connect(self._on_layer_enabled)
        self.layers_panel.selection_changed.connect(self._on_layer_selected)

        self.info_panel = InfoPanel()
        self.info_panel.render_requested.connect(self.open_loaded)
        self._panels = SidePanels()
        self._panels.add(Panel.INFO, text.PANEL_INFO, text.PANEL_INFO_TIP, self.info_panel)
        self._panels.add(
            Panel.EXTRACT, text.PANEL_EXTRACT, text.PANEL_EXTRACT_TIP, self.extract_panel
        )

        left = QScrollArea()
        left.setObjectName("layerArea")
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setWidget(self.layers_panel)
        left.setMinimumWidth(_LEFT_MIN_WIDTH)

        splitter = QSplitter()
        self._splitter = splitter
        splitter.addWidget(left)
        splitter.addWidget(self._scroll)
        splitter.addWidget(self._panels)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        self._status_info = QLabel()
        self._status_view = QLabel()
        self._frame_prev = self._ghost_button(text.FRAME_PREV, 24, partial(self._step_frame, -1))
        self._frame_prev.setToolTip(text.FRAME_PREV_TIP)
        self._frame_next = self._ghost_button(text.FRAME_NEXT, 24, partial(self._step_frame, 1))
        self._frame_next.setToolTip(text.FRAME_NEXT_TIP)
        self._frame_status = _muted_label()
        status = QStatusBar()
        status.addWidget(self._status_info, 1)
        status.addPermanentWidget(self._frame_prev)
        status.addPermanentWidget(self._frame_status)
        status.addPermanentWidget(self._frame_next)
        status.addPermanentWidget(self._status_view)
        self.setStatusBar(status)

    def _should_handle_keys(self) -> bool:
        application = _application()
        if application is None:
            return False
        if (
            application.activePopupWidget() is not None
            or application.activeModalWidget() is not None
        ):
            return False
        if isinstance(self.focusWidget(), _TEXT_INPUTS):
            return False
        active = application.activeWindow()
        return active is None or active is self

    def _handle_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        modifiers = event.modifiers()
        if modifiers & _COMMAND_MODIFIERS:
            if key == Qt.Key.Key_C:
                self._copy()
                return True
            return False
        if (arrows := _ARROW_KEYS.get(key)) is not None:
            step = _NUDGE_STEP if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            self._nudge(arrows[0] * step, arrows[1] * step)
            return True
        if (bit_step := _FOCUS_BIT_KEYS.get(key)) is not None:
            self.apply(partial(step_focus_bit, delta=bit_step, layer=self._bits_layer()))
            return True
        if (direction := _ZOOM_KEYS.get(key)) is not None:
            self._zoom_by(direction)
            return True
        if key == Qt.Key.Key_0:
            self._fit()
            return True
        return self._handle_text_key(event)

    def _handle_text_key(self, event: QKeyEvent) -> bool:
        if event.isAutoRepeat():
            return False
        label = event.text().upper()
        if label == "F":
            self.apply(cycle_format)
            return True
        layer = self._bits_layer()
        if label in _CHANNEL_LETTERS:
            self.apply(partial(select_lsb, name=label, layer=layer))
            return True
        if label in _ORDERED_LETTERS:
            self.apply(partial(select_lsb_at, position=int(label) - 1, layer=layer))
            return True
        return False

    def _apply(self, state: ViewerState) -> None:
        raster = self._resolve(state)
        # The layer panel settles its selection first, so the filter box and the
        # bit grid that follow it read a layer the new stack really has.
        self.layers_panel.set_state(state, raster)
        self._sync_controls(state, raster)
        self.canvas.set_state(state, raster)
        self.extract_panel.set_state(state, raster, bits_layer=self._bits_layer())
        self.info_panel.set_image(state.image)
        self._status_info.setText(self._info_text(state, raster))
        self._status_view.setText(status_view(state, raster))
        self._filter_count.setText(self._count_text(state, raster))
        image = state.image
        title = text.APP_NAME if image is None else f"{image.path.name} — {text.APP_NAME}"
        if self.windowTitle() != title:
            self.setWindowTitle(title)

    def _bits_layer(self) -> int | None:
        """The bits mask the grid and the shortcuts work on: the layer in hand.

        The grid lives in the extract panel, so both panels need the same answer;
        ``None`` means no bits mask is in play and an edit brings one into being.
        """
        return bits_position(self.store.state.layers, self.layers_panel.selected)

    def _resolve(self, state: ViewerState) -> Raster | None:
        """The raster a state works out to, remembered for as long as the stack stands.

        Every consumer of one state reads this one raster, and the stack — image
        and layers together — is what decides it, so moving the cursor costs no
        pixel work at all.
        """
        image = state.image
        key = None if image is None else (image, state.layers)
        if key != self._raster_key:
            self._raster_key = key
            self._raster = None if image is None else resolve(image, state.layers)
        return self._raster

    def _info_text(self, state: ViewerState, raster: Raster | None) -> str:
        """Status bar, left: what is open, and any mask line the box or stack refused."""
        info = status_info(state)
        errors = (
            [f"{text.FILTER_ERROR}{failure.message}" for failure in raster.failures]
            if (raster is not None)
            else []
        )
        if self._command_error:
            errors.append(f"{text.FILTER_ERROR}{self._command_error}")
        return "  ".join((info, *errors))

    def _count_text(self, state: ViewerState, raster: Raster | None) -> str:
        """The pass count at the right end of the filter row, empty when idle."""
        image = state.image
        if image is None or raster is None or raster.live is None:
            return ""
        return text.filter_count(raster.live_count, image.width * image.height)

    def _commit_filter(self, *, new: bool = False) -> None:
        """Enter runs the line the box holds, and writes it back as the operation it ran.

        Shift+Enter stacks it instead: the new operation goes on top, and the one
        in hand keeps its command and its place. Either way the box and the
        recipe row then read alike — ``b>r`` lands as ``b > r``, ``xor 0xff`` as
        ``xor 0xFF``. A line still being typed, or one the box refused, stays
        exactly as the typist left it, to be finished.
        """
        typed = self._filter_edit.text()
        self._apply_filter_text(new=new)
        if self._command_error:
            return
        command = self._command_text(typed)
        if command is not None:
            self._write_box(command)

    def _clear_filter(self) -> None:
        """Esc empties the box; on an already empty box it drops the operation it edits.

        Two steps, so escaping out of a line never costs the operation underneath
        it: the first press takes the words back, the second one — with nothing
        left to take — deletes that operation and hands back the canvas.
        """
        if self._filter_edit.text():
            self._filter_edit.clear()
            self._clear_command_error()
            return
        self._apply_filter_text()
        self.canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _clear_command_error(self) -> None:
        """Typing another word clears the complaint: the line is being fixed."""
        if not self._command_error:
            return
        self._command_error = ""
        self._filter_edit.setStyleSheet("")
        self._status_info.setText(self._info_text(self.store.state, self._raster))

    def _apply_filter_text(self, *, new: bool = False) -> None:
        """The box's text becomes a mask: a command, or a region expression.

        A command the box cannot read raises before anything changes, so the
        stack keeps the layer it had and the box keeps the line for fixing.
        The state the box itself applied is remembered as its own, so the sync
        that follows does not echo it back over the typist's words. ``new``
        stacks the line on top whatever the box was editing — Shift+Enter — so
        it applies even when the line is the one already in the box.
        """
        expression = self._filter_edit.text()
        self._command_error = ""
        if self.store.state.image is not None and (new or expression != self._box_text()):
            try:
                self._box_authoring = True
                transition = (
                    partial(add_mask_text, text=expression)
                    if new
                    else partial(set_mask_text, text=expression, layer=self.layers_panel.selected)
                )
                self.apply(transition)
                state = self.store.state
                self._box_edit_key = None if state.image is None else (state.image, state.layers)
            except CommandError as exc:
                self._command_error = str(exc)
            finally:
                self._box_authoring = False
        if self._command_error:
            self._status_info.setText(self._info_text(self.store.state, self._raster))
        self._sync_filter_box(self._raster)

    def _focus_filter(self) -> None:
        self._filter_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._filter_edit.selectAll()

    def _box_position(self) -> int | None:
        """Which layer the filter box is editing, if any."""
        image = self.store.state.image
        planes = () if image is None else image.planes
        return filter_position(self.store.state.layers, planes, self.layers_panel.selected)

    def _box_text(self) -> str:
        """What the filter box holds: the command line of the layer it edits."""
        position = self._box_position()
        image = self.store.state.image
        if position is None or image is None:
            return ""
        return text_of(self.store.state.layers[position].mask, image.planes) or ""

    def _command_text(self, line: str) -> str | None:
        """How the operation a line names writes itself, or ``None`` when it names none.

        This is the wording the recipe row holds, so a line the box has just run
        can be written back in it. ``None`` covers a sentence still being typed
        and a mask no command spells (a bit selection with no bits), so neither
        is rewritten.
        """
        image = self.store.state.image
        if image is None:
            return None
        mask = parse(line, image.planes)
        return None if mask is None else text_of(mask, image.planes)

    def _write_box(self, line: str) -> None:
        """Put a line in the box as the box's own doing, never as the typist's."""
        if self._filter_edit.text() == line:
            return
        self._filter_edit.blockSignals(True)
        self._filter_edit.setText(line)
        self._filter_edit.blockSignals(False)

    def _sync_controls(self, state: ViewerState, raster: Raster | None) -> None:
        image = state.image
        combo = self._format_combo
        combo.blockSignals(True)
        index = combo.findData(state.value_format.value)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._zoom_label.setText(text.zoom_label(state.zoom))
        self._sync_filter_box(raster)
        enabled = image is not None
        for widget in (
            self._filter_edit,
            self._mode_move,
            self._mode_select,
            self._zoom_in,
            self._zoom_out,
            self._zoom_fit,
            self._zoom_reset,
            self._zoom_slider,
            self._export_button,
        ):
            widget.setEnabled(enabled)
        multi = image is not None and image.frame_count > 1
        self._frame_prev.setEnabled(multi)
        self._frame_next.setEnabled(multi)
        self._frame_status.setText(
            text.frame_status(
                image.frame_index,
                image.frame_count,
                image.frame_delays[image.frame_index] if image.frame_delays else 0,
            )
            if multi
            else ""
        )
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(slider_position(state.zoom))
        self._zoom_slider.blockSignals(False)

    # --- the layer panel ---------------------------------------------------

    def _sync_filter_box(self, raster: Raster | None, *, force: bool = False) -> None:
        """Point the filter box at the layer it edits, and mark the text when at fault.

        The box holds the typist's own words, so a resync waits for the state to
        move underneath it: a line the box itself applied is never echoed back
        mid-sentence — trailing spaces and all — and only a change from elsewhere
        (the bit grid, the presets, a picked layer) rewrites the text. Enter is
        the other rewrite: it writes the line back as the command it ran. A
        refused line stays for fixing; picking a layer forces the rewrite
        regardless.
        """
        position = self._box_position()
        image = self.store.state.image
        key = None if image is None else (image, self.store.state.layers)
        if (
            (key != self._box_edit_key or force)
            and not self._box_authoring
            and not self._command_error
        ):
            self._write_box("" if position is None else self._box_text())
            self._command_error = ""
            self._box_edit_key = key
        failed = bool(self._command_error) or (
            raster is not None
            and position is not None
            and any(failure.index == position for failure in raster.failures)
        )
        self._filter_edit.setStyleSheet(_FILTER_ERROR_STYLE if failed else "")

    def _on_layer_selected(self) -> None:
        """The panel picked another layer: the filter box and the bit grid follow it.

        The rewrite is forced: the click names a new edit target, so the box
        shows that layer's command even while the box still holds focus.
        """
        self._sync_filter_box(self._raster, force=True)
        self.extract_panel.set_state(self.store.state, self._raster, bits_layer=self._bits_layer())

    def _on_layer_added(self, mask: Mask) -> None:
        self.apply(partial(add_layer, mask=mask))

    def _on_layer_removed(self, layer: int) -> None:
        self.apply(partial(remove_layer, layer=layer))

    def _on_layer_moved(self, layer: int, step: int) -> None:
        self.apply(partial(move_layer, layer=layer, step=step))

    def _on_layer_enabled(self, layer: int, on: bool) -> None:
        self.apply(partial(set_layer_enabled, layer=layer, on=on))

    def _on_bit(self, layer: int, plane: str, bit: int, exclusive: bool) -> None:
        transition = select_only if exclusive else toggle_bit
        self.apply(partial(transition, plane=plane, bit=bit, layer=layer))

    def _on_channel_toggle(self, layer: int, name: str, checked: bool) -> None:
        self.apply(partial(set_channel, name=name, on=checked, layer=layer))

    def _on_column_toggle(self, layer: int, bit: int, checked: bool) -> None:
        self.apply(partial(set_column, bit=bit, on=checked, layer=layer))

    def _on_plane_step(self, layer: int, delta: int) -> None:
        self.apply(partial(step_plane, delta=delta, layer=layer))

    def _on_channel_step(self, layer: int, delta: int) -> None:
        self.apply(partial(step_channel, delta=delta, layer=layer))

    def _on_all_bits(self, layer: int) -> None:
        self.apply(partial(select_all_bits, layer=layer))

    def _on_lsbs(self, layer: int) -> None:
        self.apply(partial(select_lsbs, layer=layer))

    def _on_format(self, index: int) -> None:
        fmt = _enum_at(self._format_combo, index, DisplayFormat)
        if fmt is None or fmt is self.store.state.value_format:
            return
        self.apply(partial(set_format, fmt=fmt))

    def _on_encoding(self, value: str) -> None:
        encoding = ExtractEncoding(value)
        if encoding is self.store.state.extract_encoding:
            return
        self.apply(partial(set_extract_encoding, encoding=encoding))

    def _on_channel_order(self, names: tuple[str, ...]) -> None:
        self.apply(partial(set_channel_order, names=names))

    def _on_bit_order(self, value: str) -> None:
        self.apply(partial(set_bit_order, bit_order=BitOrder(value)))

    def _on_scan_order(self, value: str) -> None:
        self.apply(partial(set_scan_order, scan=ScanOrder(value)))

    def _step_frame(self, delta: int) -> None:
        """Load and show the neighboring frame, keeping the whole view state."""
        image = self.store.state.image
        if image is None:
            return
        index = image.frame_index + delta
        if not 0 <= index < image.frame_count:
            return
        try:
            frame = load_frame(image.path, index)
        except ImageLoadError as exc:
            self._reporter(text.open_failed(str(exc)))
            return
        self.apply(partial(set_frame, image=frame))

    def _export_view(self) -> None:
        """Save the composed view, post-processing included, as an image file."""
        state = self.store.state
        image = state.image
        if image is None:
            return
        selected, _chosen = QFileDialog.getSaveFileName(
            self, text.VIEW_EXPORT, f"{image.path.stem}.png", text.ANY_FILE
        )
        if selected and self._raster is not None:
            self._write_view(Path(selected), self._raster)

    def _write_view(self, path: Path, raster: Raster) -> None:
        """Save the composed view: what the canvas paints, alpha kept as alpha."""
        try:
            save_image(render_export(raster), path)
        except (OSError, ValueError) as exc:
            self._reporter(text.save_failed(str(exc)))
            return
        self._status_view.setText(text.saved_to(str(path)))

    def _save_extract(self) -> None:
        """Write the full extracted stream to a file, display limits aside."""
        state = self.store.state
        if state.image is None:
            return
        selected, _chosen = QFileDialog.getSaveFileName(
            self, text.EXTRACT_SAVE, "extract.bin", text.ANY_FILE
        )
        if not selected:
            return
        try:
            Path(selected).write_bytes(self.extract_panel.extract_data())
        except OSError as exc:
            self._reporter(text.save_failed(str(exc)))
            return
        self._status_view.setText(text.saved_to(selected))

    def _show_info(self) -> None:
        """The menu item raises the info page of the sidebar."""
        self._panels.set_current(Panel.INFO)

    def _on_hover(self, x: int, y: int) -> None:
        self.apply(partial(set_cursor, coord=PixelCoord(x, y)))

    def _on_region_selecting(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._status_view.setText(text.selection_status(x0, y0, x1, y1))

    def _on_region_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """A settled region is still a preview; Enter appends it to the filter."""
        image = self.store.state.image
        passed = self._region_hits(x0, y0, x1, y1)
        count = (
            text.filter_count(passed, image.width * image.height)
            if passed is not None and image is not None
            else ""
        )
        self._status_view.setText(text.selection_ready(x0, y0, x1, y1, count))

    def _on_region_canceled(self) -> None:
        self._status_view.setText(status_view(self.store.state, self._raster))

    def _on_region_committed(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """The dragged region becomes a region mask of its own, on top of the stack."""
        mask = RegionMask(f"rect({x0}, {y0}, {x1 + 1}, {y1 + 1})")
        self.apply(partial(add_layer, mask=mask))

    def _region_hits(self, x0: int, y0: int, x1: int, y1: int) -> int | None:
        """Pixels the dragged rect would leave standing, against the stack as it is."""
        if self._raster is None:
            return None
        return self._raster.live_in(x0, y0, x1, y1)

    def _on_mode(self, button: QPushButton) -> None:
        self.canvas.set_mode(CanvasMode.SELECT if button is self._mode_select else CanvasMode.PAN)
        # The click left focus on the button; space must reach the canvas to pan.
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def _nudge(self, dx: int, dy: int) -> None:
        self.apply(partial(move_cursor, dx=dx, dy=dy))
        self._reveal_cursor()

    def _zoom_by(
        self, step: int, viewport_x: int | None = None, viewport_y: int | None = None
    ) -> None:
        state = self.store.state
        if state.image is None or step == 0:
            return
        new_zoom = min(max(state.zoom + step, MIN_ZOOM), MAX_ZOOM)
        self._zoom_around(new_zoom, viewport_x, viewport_y)

    def _zoom_scale(
        self, factor: float, viewport_x: int | None = None, viewport_y: int | None = None
    ) -> None:
        """Trackpad pinch and ⌘+scroll: scale the zoom by a factor in place."""
        state = self.store.state
        if state.image is None or factor <= 0:
            return
        new_zoom = min(max(state.zoom * factor, MIN_ZOOM), MAX_ZOOM)
        self._zoom_around(new_zoom, viewport_x, viewport_y)

    def _zoom_around(self, new_zoom: float, viewport_x: int | None, viewport_y: int | None) -> None:
        state = self.store.state
        if state.image is None or new_zoom == state.zoom:
            return
        viewport = self._scroll.viewport()
        anchor_x, anchor_y = self._anchor(state, viewport, viewport_x, viewport_y)
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        image_x = (horizontal.value() + anchor_x) / state.zoom
        image_y = (vertical.value() + anchor_y) / state.zoom
        self.apply(partial(set_zoom, zoom=new_zoom))
        horizontal.setValue(round(image_x * new_zoom - anchor_x))
        vertical.setValue(round(image_y * new_zoom - anchor_y))

    def _anchor(
        self,
        state: ViewerState,
        viewport: QWidget,
        viewport_x: int | None,
        viewport_y: int | None,
    ) -> tuple[float, float]:
        """The viewport point that zooming keeps still, or the cursor if there is one."""
        if viewport_x is not None and viewport_y is not None:
            return viewport_x, viewport_y
        if state.cursor is None:
            return viewport.width() // 2, viewport.height() // 2
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        x = state.cursor.x * state.zoom - horizontal.value() + state.zoom // 2
        y = state.cursor.y * state.zoom - vertical.value() + state.zoom // 2
        return (
            min(max(x, 0), max(viewport.width() - 1, 0)),
            min(max(y, 0), max(viewport.height() - 1, 0)),
        )

    def _on_slider(self, position: int) -> None:
        self._zoom_to(float(slider_zoom(position)))

    def _zoom_to(self, new_zoom: float) -> None:
        self._zoom_around(new_zoom, None, None)

    def _fit(self) -> None:
        size = self.canvas.displayed_size()
        if size is None:
            return
        zoom = self._fit_zoom(size)
        self.apply(partial(set_zoom, zoom=zoom))
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)

    def _fit_zoom(self, size: tuple[int, int]) -> float:
        viewport = self._scroll.viewport().size()
        return initial_zoom(
            size,
            (max(viewport.width() - 2, 1), max(viewport.height() - 2, 1)),
            cap=MAX_ZOOM,
        )

    def _reveal_cursor(self) -> None:
        cursor = self.store.state.cursor
        zoom = self.store.state.zoom
        if cursor is None:
            return
        cell = self.canvas.displayed_at(cursor)
        if cell is None:
            return
        dx, dy = cell
        self._scroll.ensureVisible(
            round(dx * zoom + zoom / 2),
            round(dy * zoom + zoom / 2),
            round(zoom),
            round(zoom),
        )

    def _copy(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(readout_text(self.store.state, self._raster))

    def _open_dialog(self) -> None:
        selected, _chosen = QFileDialog.getOpenFileName(self, text.OPEN, "", text.IMAGE_FILTER)
        if selected:
            self.open_path(Path(selected))

    def _show_shortcuts(self) -> None:
        QMessageBox.information(self, text.SHORTCUTS, text.SHORTCUT_HELP)

    def _report_with_dialog(self, message: str) -> None:
        QMessageBox.warning(self, text.OPEN_FAILED, message)

    def _accept_file_drag(self, event: QDragEnterEvent | QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def _open_dropped(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if (local := url.toLocalFile()) and Path(local).is_file():
                self.open_path(Path(local))
                event.acceptProposedAction()
                return
        event.ignore()


def _combo(tip: str) -> QComboBox:
    combo = QComboBox()
    combo.setToolTip(tip)
    return combo


def _row_frame(name: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    return frame


def _muted_label(label: str = "") -> QLabel:
    widget = QLabel(label)
    widget.setObjectName("muted")
    return widget


def _application() -> QApplication | None:
    application = QApplication.instance()
    if isinstance(application, QApplication):
        return application
    return None


def _drop_checked(slot: Callable[..., object]) -> Callable[..., None]:
    def wrapped(*_args: object) -> None:
        slot()

    return wrapped


def _enum_at[T: DisplayFormat](
    combo: QComboBox,
    index: int,
    kind: type[T],
) -> T | None:
    if index < 0:
        return None
    value = combo.itemData(index)
    if not isinstance(value, str):
        return None
    return kind(value)

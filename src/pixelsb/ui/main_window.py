"""Main window: toolbar, canvas, inspector, and keyboard shortcuts."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import override

import numpy as np
from numpy.typing import NDArray
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
    QCheckBox,
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

from pixelsb.domain.geometry import SLIDER_STEPS, initial_zoom, slider_position, slider_zoom
from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    BitOrder,
    DisplayFormat,
    ExtractEncoding,
    PixelCoord,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.predicate import PredicateError, compile_filter
from pixelsb.domain.selection import effective_selection
from pixelsb.domain.transitions import (
    cycle_format,
    move_cursor,
    open_image,
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
    set_filter_expr,
    set_format,
    set_only_matched,
    set_scan_order,
    set_zoom,
    step_channel,
    step_focus_bit,
    step_plane,
    toggle_bit,
)
from pixelsb.io.loading import ImageLoadError, load_image
from pixelsb.ui import text, theme
from pixelsb.ui.canvas import CanvasMode, ImageCanvas
from pixelsb.ui.inspector import Inspector
from pixelsb.ui.painting import lighten_clear_button
from pixelsb.ui.store import Store, Transition
from pixelsb.ui.text import readout_text, status_info, status_view

type ErrorReporter = Callable[[str], None]

_FILTER_ERROR_STYLE = f"QLineEdit {{ border: 1px solid {theme.DANGER}; }}"
_TEXT_INPUTS = (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit, QComboBox)
_CANVAS_MIN_WIDTH = 260
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
_CHANNEL_LETTERS = ("R", "G", "A", "L")
_ORDERED_LETTERS = tuple(str(index) for index in range(1, 10))


@dataclass(frozen=True, slots=True)
class _Match:
    """The applied display filter's verdict for one state.

    An empty verdict means nothing is being filtered; an ``error`` means the
    expression does not compile, and ``passed`` is how many pixels ``mask``
    holds when there is no error.
    """

    mask: NDArray[np.bool_] | None = None
    error: str | None = None
    passed: int = 0


class MainWindow(QMainWindow):
    def __init__(self, reporter: ErrorReporter | None = None) -> None:
        super().__init__()
        self.store = Store()
        self._reporter = reporter or self._report_with_dialog
        self._match = _Match()
        self._match_key: object = None
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
        """Give the inspector the width its dump needs, once, before any dragging."""
        width = self.width()
        wanted = self.inspector.preferred_width()
        limit = max(width - self._splitter.handleWidth() - _CANVAS_MIN_WIDTH, 1)
        panel = min(max(wanted, self.inspector.minimumWidth()), limit)
        self._splitter.setSizes([max(width - panel, 1), panel])

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
        quit_action = file_menu.addAction(text.QUIT)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(_drop_checked(self.close))

        view_menu = self.menuBar().addMenu(text.VIEW_MENU)
        view_menu.addAction(text.ORIGINAL).triggered.connect(
            _drop_checked(partial(self.apply, select_all_bits))
        )
        view_menu.addAction(text.ALL_LSB).triggered.connect(
            _drop_checked(partial(self.apply, select_lsbs))
        )

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
        for widget in (*self._filter_widgets(), *self._mode_widgets(), *self._zoom_widgets()):
            layout.addWidget(widget)
        # The filter box takes whatever width the controls to its right leave.
        layout.setStretch(0, 1)
        return row

    def _filter_widgets(self) -> tuple[QWidget, ...]:
        """The expression box, its debounce, and the pass count beside it."""
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(text.FILTER_PLACEHOLDER)
        self._filter_edit.setClearButtonEnabled(True)
        lighten_clear_button(self._filter_edit)
        self._filter_edit.setMinimumWidth(220)
        self._filter_edit.setFixedHeight(theme.CONTROL_HEIGHT)
        self._filter_edit.setToolTip(text.FILTER_TIP)
        self._filter_edit.textEdited.connect(lambda _text: self._filter_timer.start())
        self._filter_edit.returnPressed.connect(self._commit_filter)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(200)
        self._filter_timer.timeout.connect(self._apply_filter_text)
        find_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        find_shortcut.activated.connect(self._focus_filter)
        self._filter_count = _muted_label()
        return self._filter_edit, self._filter_count

    def _mode_widgets(self) -> tuple[QWidget, ...]:
        """The compacted-view switch and the exclusive 移动 / 选区 pair."""
        self._only_matched = QCheckBox(text.ONLY_MATCHED)
        self._only_matched.setToolTip(text.ONLY_MATCHED_TIP)
        self._only_matched.setFixedHeight(theme.CONTROL_HEIGHT)
        self._only_matched.toggled.connect(self._on_only_matched)

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
        return self._only_matched, self._mode_move, self._mode_select

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

        self.inspector = Inspector()
        self.inspector.bit_clicked.connect(self._on_bit)
        self.inspector.channel_toggle.connect(self._on_channel_toggle)
        self.inspector.column_toggle.connect(self._on_column_toggle)
        self.inspector.original_requested.connect(partial(self.apply, select_all_bits))
        self.inspector.lsbs_requested.connect(partial(self.apply, select_lsbs))
        self.inspector.plane_step.connect(self._on_plane_step)
        self.inspector.channel_step.connect(self._on_channel_step)
        self.inspector.encoding_requested.connect(self._on_encoding)
        self.inspector.channel_order_requested.connect(self._on_channel_order)
        self.inspector.bit_order_requested.connect(self._on_bit_order)
        self.inspector.scan_requested.connect(self._on_scan_order)
        inspector_scroll = QScrollArea()
        inspector_scroll.setObjectName("inspectorArea")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setWidget(self.inspector)
        inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)

        splitter = QSplitter()
        self._splitter = splitter
        splitter.addWidget(self._scroll)
        splitter.addWidget(inspector_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        self._status_info = QLabel()
        self._status_view = QLabel()
        status = QStatusBar()
        status.addWidget(self._status_info, 1)
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
            self.apply(partial(step_focus_bit, delta=bit_step))
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
        if label in _CHANNEL_LETTERS:
            self.apply(partial(select_lsb, name=label))
            return True
        if label in _ORDERED_LETTERS:
            self.apply(partial(select_lsb_at, index=int(label) - 1))
            return True
        return False

    def _apply(self, state: ViewerState) -> None:
        match = self._filter_match(state)
        self._sync_controls(state)
        self.canvas.set_state(state, match.mask)
        self.inspector.set_state(state, match.mask)
        info = status_info(state)
        filtering = bool(state.filter_expr.strip())
        if filtering and match.error is not None:
            info = f"{info}  {text.FILTER_ERROR}{match.error}"
        self._status_info.setText(info)
        self._status_view.setText(status_view(state))
        self._filter_count.setText(self._count_text(state, match))
        self._filter_edit.setStyleSheet(_FILTER_ERROR_STYLE if match.error else "")
        image = state.image
        title = text.APP_NAME if image is None else f"{image.path.name} — {text.APP_NAME}"
        if self.windowTitle() != title:
            self.setWindowTitle(title)

    def _count_text(self, state: ViewerState, match: _Match) -> str:
        """The pass count at the right end of the filter row, empty when idle."""
        image = state.image
        if image is None or not state.filter_expr.strip() or match.error is not None:
            return ""
        return text.filter_count(match.passed, image.width * image.height)

    def _filter_match(self, state: ViewerState) -> _Match:
        """The compiled filter's verdict, cached on the expression it was built for."""
        image = state.image
        if image is None or not state.filter_expr.strip():
            return _Match()
        key = (state.filter_expr, image, state.selection)
        if key == self._match_key:
            return self._match
        self._match_key = key
        try:
            compiled = compile_filter(state.filter_expr, image.planes)
            mask = compiled.evaluate(image, effective_selection(image, state.selection))
        except PredicateError as exc:
            self._match = _Match(error=str(exc))
        else:
            self._match = _Match(mask=mask, passed=int(mask.sum()))
        return self._match

    def _commit_filter(self) -> None:
        """Enter applies the filter right away and keeps the caret in the box."""
        self._filter_timer.stop()
        self._apply_filter_text()

    def _clear_filter(self) -> None:
        """Esc clears the filter and hands the keyboard back to the canvas."""
        self._filter_timer.stop()
        self._filter_edit.clear()
        self._apply_filter_text()
        self.canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _apply_filter_text(self) -> None:
        expression = self._filter_edit.text()
        state = self.store.state
        cleared = not expression.strip()
        if expression == state.filter_expr and (not cleared or not state.only_matched):
            return

        def update(current: ViewerState) -> ViewerState:
            updated = set_filter_expr(current, expression)
            # A cleared filter has nothing to show exclusively, so uncheck.
            return set_only_matched(updated, on=False) if cleared else updated

        self.apply(update)

    def _on_only_matched(self, checked: bool) -> None:
        self.apply(partial(set_only_matched, on=checked))
        self._fit()

    def _focus_filter(self) -> None:
        self._filter_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._filter_edit.selectAll()

    def _sync_controls(self, state: ViewerState) -> None:
        image = state.image
        combo = self._format_combo
        combo.blockSignals(True)
        index = combo.findData(state.value_format.value)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._zoom_label.setText(text.zoom_label(state.zoom))
        self._filter_edit.blockSignals(True)
        if self._filter_edit.text() != state.filter_expr:
            self._filter_edit.setText(state.filter_expr)
        self._filter_edit.blockSignals(False)
        self._filter_timer.stop()
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
        ):
            widget.setEnabled(enabled)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(slider_position(state.zoom))
        self._zoom_slider.blockSignals(False)
        only = self._only_matched
        only.setEnabled(enabled and bool(state.filter_expr.strip()) and self._match.error is None)
        only.blockSignals(True)
        if only.isChecked() != state.only_matched:
            only.setChecked(state.only_matched)
        only.blockSignals(False)

    def _on_bit(self, plane: str, bit: int, exclusive: bool) -> None:
        if exclusive:
            self.apply(partial(select_only, plane=plane, bit=bit))
        else:
            self.apply(partial(toggle_bit, plane=plane, bit=bit))

    def _on_channel_toggle(self, name: str, checked: bool) -> None:
        self.apply(partial(set_channel, name=name, on=checked))

    def _on_column_toggle(self, bit: int, checked: bool) -> None:
        self.apply(partial(set_column, bit=bit, on=checked))

    def _on_plane_step(self, delta: int) -> None:
        self.apply(partial(step_plane, delta=delta))

    def _on_channel_step(self, delta: int) -> None:
        self.apply(partial(step_channel, delta=delta))

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
        self._status_view.setText(status_view(self.store.state))

    def _on_region_committed(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Append the previewed region; the parenthesized prefix keeps `or` intact."""
        rect = f"rect({x0}, {y0}, {x1 + 1}, {y1 + 1})"
        current = self._filter_edit.text().strip()
        self._filter_edit.setText(f"({current}) and {rect}" if current else rect)
        self._commit_filter()

    def _region_hits(self, x0: int, y0: int, x1: int, y1: int) -> int | None:
        """Pixels the appended rect would leave selected, against the applied filter."""
        image = self.store.state.image
        if image is None or self._match.error is not None:
            return None
        mask = self._match.mask
        if mask is None:
            return (y1 - y0 + 1) * (x1 - x0 + 1)
        return int(mask[y0 : y1 + 1, x0 : x1 + 1].sum())

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
            clipboard.setText(readout_text(self.store.state))

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

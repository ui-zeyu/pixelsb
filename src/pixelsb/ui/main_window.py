"""Main window: toolbar, canvas, inspector, and keyboard shortcuts."""

from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QWidget,
)

from pixelsb.domain.geometry import initial_zoom
from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    DisplayFormat,
    LoadedImage,
    PixelCoord,
    ValueMode,
    ViewerState,
)
from pixelsb.domain.transitions import (
    clear_anchor,
    cycle_format,
    move_cursor,
    open_image,
    reset_readout,
    select_all_bits,
    select_lsb,
    select_lsb_at,
    select_lsbs,
    select_only,
    select_only_readout,
    set_anchor,
    set_channel,
    set_column,
    set_cursor,
    set_detached,
    set_format,
    set_readout_channel,
    set_readout_column,
    set_value_mode,
    set_zoom,
    step_focus_bit,
    toggle_bit,
    toggle_readout_bit,
    toggle_value_mode,
)
from pixelsb.io.loading import ImageLoadError, load_image
from pixelsb.ui import text
from pixelsb.ui.canvas import ImageCanvas
from pixelsb.ui.inspector import Inspector
from pixelsb.ui.store import Store, Transition
from pixelsb.ui.text import readout_text, status_text

type ErrorReporter = Callable[[str], None]


class MainWindow(QMainWindow):
    def __init__(self, reporter: ErrorReporter | None = None) -> None:
        super().__init__()
        self.store = Store()
        self._reporter = reporter or self._report_with_dialog
        self._combo_items: dict[int, tuple[tuple[str, str], ...]] = {}
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
        zoom = self._fit_zoom(image)
        self.apply(lambda state: open_image(state, image, zoom=zoom))
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
        QTimer.singleShot(0, self._fit)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
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

    def closeEvent(self, event: QCloseEvent) -> None:
        application = _application()
        if application is not None:
            application.removeEventFilter(self)
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self._accept_file_drag(event)

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
            _drop_checked(lambda: self.apply(select_all_bits))
        )
        view_menu.addAction(text.ALL_LSB).triggered.connect(
            _drop_checked(lambda: self.apply(select_lsbs))
        )

        help_menu = self.menuBar().addMenu(text.HELP_MENU)
        help_action = help_menu.addAction(text.SHORTCUTS)
        help_action.triggered.connect(_drop_checked(self._show_shortcuts))

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("view")
        toolbar.setMovable(False)
        self._detach = QCheckBox(text.DETACH)
        self._detach.setToolTip(text.DETACH_TIP)
        self._detach.toggled.connect(self._on_detached)
        self._zoom_in = QPushButton(text.ZOOM_IN)
        self._zoom_in.setFixedWidth(32)
        self._zoom_in.clicked.connect(_drop_checked(lambda: self._zoom_by(1)))
        self._zoom_out = QPushButton(text.ZOOM_OUT)
        self._zoom_out.setFixedWidth(32)
        self._zoom_out.clicked.connect(_drop_checked(lambda: self._zoom_by(-1)))
        self._zoom_fit = QPushButton(text.ZOOM_FIT)
        self._zoom_fit.clicked.connect(_drop_checked(self._fit))
        self._zoom_reset = QPushButton(text.ZOOM_RESET)
        self._zoom_reset.setToolTip(text.ZOOM_RESET_TIP)
        self._zoom_reset.clicked.connect(_drop_checked(lambda: self._zoom_to(float(MIN_ZOOM))))
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(int(MIN_ZOOM), int(MAX_ZOOM))
        self._zoom_slider.setFixedWidth(160)
        self._zoom_slider.setToolTip(text.ZOOM_TIP)
        self._zoom_slider.valueChanged.connect(self._zoom_to)
        self._format_combo = _combo(text.FORMAT_TIP)
        self._format_combo.currentIndexChanged.connect(self._on_format)
        self._value_combo = _combo(text.VALUE_MODE_TIP)
        self._value_combo.currentIndexChanged.connect(self._on_value_mode)
        self._anchor_label = QLabel(f"{text.ANCHOR} —")
        self._clear_anchor = QPushButton(text.CLEAR_ANCHOR)
        self._clear_anchor.clicked.connect(_drop_checked(lambda: self.apply(clear_anchor)))

        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        for label, widget in (
            ("", self._detach),
            (text.VALUE, self._format_combo),
            ("", self._value_combo),
            ("", self._zoom_out),
            ("", self._zoom_slider),
            ("", self._zoom_in),
            ("", self._zoom_fit),
            ("", self._zoom_reset),
            ("", self._anchor_label),
            ("", self._clear_anchor),
        ):
            if label:
                layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        layout.addStretch(1)
        toolbar.addWidget(host)

    def _build_body(self) -> None:
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas = ImageCanvas(self._scroll)
        self._scroll.setWidget(self.canvas)
        self._scroll.viewport().setAcceptDrops(True)
        self._scroll.viewport().installEventFilter(self)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.anchored.connect(self._on_anchor)
        self.canvas.zoom_requested.connect(self._zoom_by)
        self.canvas.file_dropped.connect(lambda path: self.open_path(Path(path)))

        self.inspector = Inspector()
        self.inspector.bit_clicked.connect(self._on_bit)
        self.inspector.readout_bit_clicked.connect(self._on_readout_bit)
        self.inspector.channel_toggle.connect(self._on_channel_toggle)
        self.inspector.readout_channel_toggle.connect(self._on_readout_channel)
        self.inspector.column_toggle.connect(self._on_column_toggle)
        self.inspector.readout_column_toggle.connect(self._on_readout_column_toggle)
        self.inspector.original_requested.connect(lambda: self.apply(select_all_bits))
        self.inspector.lsbs_requested.connect(lambda: self.apply(select_lsbs))
        self.inspector.reset_readout_requested.connect(lambda: self.apply(reset_readout))
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setWidget(self.inspector)
        inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        inspector_scroll.setMinimumWidth(300)

        splitter = QSplitter()
        splitter.addWidget(self._scroll)
        splitter.addWidget(inspector_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([840, 360])
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        self._status = QLabel()
        status = QStatusBar()
        status.addWidget(self._status, 1)
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
        active = application.activeWindow()
        return active is None or active is self

    def _handle_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        modifiers = event.modifiers()
        command = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
                | Qt.KeyboardModifier.AltModifier
            )
        )
        if command and key == Qt.Key.Key_C:
            self._copy()
            return True
        if command:
            return False
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        match key:
            case Qt.Key.Key_Left:
                self._nudge(-8 if shift else -1, 0)
                return True
            case Qt.Key.Key_Right:
                self._nudge(8 if shift else 1, 0)
                return True
            case Qt.Key.Key_Up:
                self._nudge(0, -8 if shift else -1)
                return True
            case Qt.Key.Key_Down:
                self._nudge(0, 8 if shift else 1)
                return True
            case Qt.Key.Key_Escape:
                self.apply(clear_anchor)
                return True
            case Qt.Key.Key_BracketLeft:
                self.apply(lambda state: step_focus_bit(state, -1))
                return True
            case Qt.Key.Key_BracketRight:
                self.apply(lambda state: step_focus_bit(state, 1))
                return True
            case Qt.Key.Key_Plus | Qt.Key.Key_Equal:
                self._zoom_by(1)
                return True
            case Qt.Key.Key_Minus:
                self._zoom_by(-1)
                return True
            case Qt.Key.Key_0:
                self._fit()
                return True
            case _:
                return self._handle_text_key(event)

    def _handle_text_key(self, event: QKeyEvent) -> bool:
        if event.isAutoRepeat():
            return False
        label = event.text().upper()
        match label:
            case "F":
                self.apply(cycle_format)
            case "O":
                self.apply(toggle_value_mode)
            case "R" | "G" | "A" | "L":
                self.apply(lambda state, name=label: select_lsb(state, name))
            case "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                self.apply(lambda state, index=int(label) - 1: select_lsb_at(state, index))
            case _:
                return False
        return True

    def _apply(self, state: ViewerState) -> None:
        self._sync_controls(state)
        self.canvas.set_state(state)
        self.inspector.set_state(state)
        self._status.setText(status_text(state))
        image = state.image
        title = text.APP_NAME if image is None else f"{image.path.name} — {text.APP_NAME}"
        if self.windowTitle() != title:
            self.setWindowTitle(title)

    def _sync_controls(self, state: ViewerState) -> None:
        image = state.image
        self._fill(
            self._format_combo,
            (
                (DisplayFormat.DECIMAL.value, text.DECIMAL),
                (DisplayFormat.HEX.value, text.HEX),
                (DisplayFormat.BINARY.value, text.BINARY),
            ),
            state.value_format.value,
        )
        self._fill(
            self._value_combo,
            (
                (ValueMode.ABSOLUTE.value, text.ABSOLUTE),
                (ValueMode.OFFSET.value, text.OFFSET),
            ),
            state.value_mode.value,
        )
        anchor = state.anchor
        self._anchor_label.setText(
            f"{text.ANCHOR} —" if anchor is None else f"{text.ANCHOR} ({anchor.x}, {anchor.y})"
        )
        enabled = image is not None
        for widget in (
            self._detach,
            self._zoom_in,
            self._zoom_out,
            self._zoom_fit,
            self._zoom_reset,
            self._zoom_slider,
        ):
            widget.setEnabled(enabled)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(round(state.zoom))
        self._zoom_slider.blockSignals(False)
        self._detach.blockSignals(True)
        self._detach.setChecked(state.detached)
        self._detach.blockSignals(False)
        self._clear_anchor.setEnabled(anchor is not None)

    def _fill(
        self, combo: QComboBox, items: Sequence[tuple[str, str]], current: str | None
    ) -> None:
        signature = tuple(items)
        combo.blockSignals(True)
        try:
            if self._combo_items.get(id(combo)) != signature:
                combo.clear()
                for value, label in items:
                    combo.addItem(label, value)
                self._combo_items[id(combo)] = signature
            index = -1 if current is None else combo.findData(current)
            if index >= 0 and combo.currentIndex() != index:
                combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)

    def _on_bit(self, plane: str, bit: int, exclusive: bool) -> None:
        if exclusive:
            self.apply(lambda state: select_only(state, plane, bit))
        else:
            self.apply(lambda state: toggle_bit(state, plane, bit))

    def _on_readout_bit(self, plane: str, bit: int, exclusive: bool) -> None:
        if exclusive:
            self.apply(lambda state: select_only_readout(state, plane, bit))
        else:
            self.apply(lambda state: toggle_readout_bit(state, plane, bit))

    def _on_readout_channel(self, name: str, checked: bool) -> None:
        self.apply(lambda state: set_readout_channel(state, name, on=checked))

    def _on_channel_toggle(self, name: str, checked: bool) -> None:
        self.apply(lambda state: set_channel(state, name, on=checked))

    def _on_column_toggle(self, bit: int, checked: bool) -> None:
        self.apply(lambda state: set_column(state, bit, on=checked))

    def _on_readout_column_toggle(self, bit: int, checked: bool) -> None:
        self.apply(lambda state: set_readout_column(state, bit, on=checked))

    def _on_detached(self, checked: bool) -> None:
        if checked is self.store.state.detached:
            return
        self.apply(lambda state: set_detached(state, checked))

    def _on_format(self, index: int) -> None:
        fmt = _enum_at(self._format_combo, index, DisplayFormat)
        if fmt is None or fmt is self.store.state.value_format:
            return
        self.apply(lambda state: set_format(state, fmt))

    def _on_value_mode(self, index: int) -> None:
        mode = _enum_at(self._value_combo, index, ValueMode)
        if mode is None or mode is self.store.state.value_mode:
            return
        self.apply(lambda state: set_value_mode(state, mode))

    def _on_hover(self, x: int, y: int) -> None:
        self.apply(lambda state: set_cursor(state, PixelCoord(x, y)))

    def _on_anchor(self, x: int, y: int) -> None:
        self.apply(lambda state: set_anchor(state, PixelCoord(x, y)))

    def _nudge(self, dx: int, dy: int) -> None:
        self.apply(lambda state: move_cursor(state, dx, dy))
        self._reveal_cursor()

    def _zoom_by(
        self, step: int, viewport_x: int | None = None, viewport_y: int | None = None
    ) -> None:
        state = self.store.state
        if state.image is None or step == 0:
            return
        new_zoom = min(max(state.zoom + step, MIN_ZOOM), MAX_ZOOM)
        if new_zoom == state.zoom:
            return
        viewport = self._scroll.viewport()
        if viewport_x is None or viewport_y is None:
            viewport_x = viewport.width() // 2
            viewport_y = viewport.height() // 2
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        image_x = (horizontal.value() + viewport_x) / state.zoom
        image_y = (vertical.value() + viewport_y) / state.zoom
        self.apply(lambda current: set_zoom(current, new_zoom))
        horizontal.setValue(round(image_x * new_zoom - viewport_x))
        vertical.setValue(round(image_y * new_zoom - viewport_y))

    def _zoom_to(self, new_zoom: float) -> None:
        state = self.store.state
        if state.image is None or new_zoom == state.zoom:
            return
        viewport = self._scroll.viewport()
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        if state.cursor is not None:
            viewport_x = state.cursor.x * state.zoom - horizontal.value() + state.zoom // 2
            viewport_y = state.cursor.y * state.zoom - vertical.value() + state.zoom // 2
            viewport_x = min(max(viewport_x, 0), max(viewport.width() - 1, 0))
            viewport_y = min(max(viewport_y, 0), max(viewport.height() - 1, 0))
        else:
            viewport_x = viewport.width() // 2
            viewport_y = viewport.height() // 2
        image_x = (horizontal.value() + viewport_x) / state.zoom
        image_y = (vertical.value() + viewport_y) / state.zoom
        self.apply(lambda current: set_zoom(current, new_zoom))
        horizontal.setValue(round(image_x * new_zoom - viewport_x))
        vertical.setValue(round(image_y * new_zoom - viewport_y))

    def _fit(self) -> None:
        image = self.store.state.image
        if image is None:
            return
        zoom = self._fit_zoom(image)
        self.apply(lambda state: set_zoom(state, zoom))
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)

    def _fit_zoom(self, image: LoadedImage) -> float:
        viewport = self._scroll.viewport().size()
        return initial_zoom(
            (image.width, image.height),
            (max(viewport.width() - 2, 1), max(viewport.height() - 2, 1)),
            cap=MAX_ZOOM,
        )

    def _reveal_cursor(self) -> None:
        cursor = self.store.state.cursor
        zoom = self.store.state.zoom
        if cursor is None:
            return
        self._scroll.ensureVisible(
            round(cursor.x * zoom + zoom / 2),
            round(cursor.y * zoom + zoom / 2),
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
            local = url.toLocalFile()
            if local and Path(local).is_file():
                self.open_path(Path(local))
                event.acceptProposedAction()
                return
        event.ignore()


def _combo(tip: str) -> QComboBox:
    combo = QComboBox()
    combo.setToolTip(tip)
    return combo


def _application() -> QApplication | None:
    application = QApplication.instance()
    if isinstance(application, QApplication):
        return application
    return None


def _drop_checked(slot: Callable[..., object]) -> Callable[..., None]:
    def wrapped(*_args: object) -> None:
        slot()

    return wrapped


def _enum_at[T: DisplayFormat | ValueMode](
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

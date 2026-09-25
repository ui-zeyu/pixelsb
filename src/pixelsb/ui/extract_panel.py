"""The extract panel: which bits are read, how the stream is laid out, the bytes."""

from PySide6.QtCore import Qt, Signal, SignalInstance
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.classify import Classification, StreamClassifier
from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.extract import (
    ExtractRow,
    applied_order,
    extract_bytes,
    filter_extract,
    format_extract,
    order_choices,
)
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    Raster,
    SamplePlane,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.transitions import bits_shadowed, current_bits
from pixelsb.io.classify import stream_classifier
from pixelsb.ui import text, theme
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.controls import Toggle, TogglePair, hairline, section_title
from pixelsb.ui.extract_view import ExtractView

_WIDTH_SLACK = 16  # the panel's own scrollbar can appear once an image is open
_STEPPER_WIDTH = 30  # the arrow buttons stay compact around the grid
_STEPPER_GAP = 6  # the channel column's gap, reused to indent the plane row
_PANEL_MIN_WIDTH = 300  # the floor below which the orders and the dump stop reading

_BIT_ORDER_TOGGLES = (
    Toggle(BitOrder.MSB, "MSB", text.ORDER_BIT_MSB_TIP),
    Toggle(BitOrder.LSB, "LSB", text.ORDER_BIT_LSB_TIP),
)
_SCAN_TOGGLES = (
    Toggle(ScanOrder.XY, "XY", text.ORDER_SCAN_XY_TIP),
    Toggle(ScanOrder.YZ, "YZ", text.ORDER_SCAN_YZ_TIP),
)


class ExtractPanel(QWidget):
    """Panel on the right: the bit grid, the stream's order, and the bytes themselves."""

    encoding_requested = Signal(str)
    channel_order_requested = Signal(object)  # a tuple of channel names
    bit_order_requested = Signal(str)
    scan_requested = Signal(str)
    save_requested = Signal()
    bit_clicked = Signal(int, str, int, bool)  # layer, plane, bit, this bit alone
    channel_toggle = Signal(int, str, bool)
    column_toggle = Signal(int, int, bool)
    original_requested = Signal(int)
    lsbs_requested = Signal(int)
    plane_step = Signal(int, int)  # layer, delta
    channel_step = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("extractPanel")
        self._state = ViewerState()
        self._bits_layer: int | None = None
        self._extract_key: tuple[object, ...] | None = None
        self._view_key: tuple[object, ...] | None = None
        self._order_key: tuple[object, ...] | None = None
        self._extract_rows: tuple[ExtractRow, ...] = ()
        self._extract_data = b""
        self._classification: Classification | None = None
        self._detections: tuple[Detection, ...] = ()
        self._classify: StreamClassifier | None = None
        self._order_items: tuple[tuple[str, ...], ...] = ()
        self._bits_editor = _BitsEditor(self)
        # The three orders the extract stream reads: which channel's bits come
        # first, which end of a byte the first bit lands in, and which pixel
        # axis runs fastest.
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip(text.ORDER_CHANNEL_TIP)
        self._channel_combo.setFixedHeight(theme.CONTROL_HEIGHT)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_order)
        self._bit_order = TogglePair(self, _BIT_ORDER_TOGGLES, self.bit_order_requested)
        self._scan = TogglePair(self, _SCAN_TOGGLES, self.scan_requested)
        self._extract_search = QLineEdit()
        self._extract_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._extract_search.setFixedHeight(theme.CONTROL_HEIGHT)
        self._extract_search.textChanged.connect(lambda _text: self._refresh_view())
        self._extract_view = ExtractView()
        self._extract_view.setMinimumHeight(180)
        self._extract_view.encoding_changed.connect(self.encoding_requested.emit)
        self.setMinimumWidth(_PANEL_MIN_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(self._bits_editor.header())
        layout.addLayout(self._bits_editor.body())
        layout.addLayout(self._bits_editor.arrows())
        layout.addSpacing(14)
        layout.addWidget(hairline())
        layout.addSpacing(14)
        layout.addWidget(section_title(text.SECTION_EXTRACT))
        layout.addWidget(self._order_row())
        layout.addWidget(self._extract_search)
        layout.addWidget(self._extract_view, 1)
        self._wire_bits()

    def set_state(
        self,
        state: ViewerState,
        raster: Raster | None = None,
        *,
        bits_layer: int | None = None,
    ) -> None:
        """Show one state. ``bits_layer`` is the bits mask the grid shows and edits."""
        self._state = state
        self._bits_layer = bits_layer
        planes = () if state.image is None else state.image.planes
        chosen = current_bits(state, bits_layer)
        # The grid shows the mask in hand; the order row describes the stream, which
        # a shadowing mask above may be the one packing. Before an image is open
        # there is no raster and nothing to pack, so the two agree.
        packed = chosen if raster is None else raster.selection
        self._bits_editor.set_layer(planes, chosen)
        self._bits_editor.set_shadowed(bits_shadowed(state.layers, bits_layer))
        self._sync_order_controls(planes, packed)
        self._sync_extract(state, raster)
        self._extract_view.sync_encoding(state.extract_encoding)

    def extract_data(self) -> bytes:
        """The full extracted stream, untruncated by the dump's display limit."""
        return self._extract_data

    def preferred_width(self) -> int:
        """Panel width that holds the bit grid and both dump panes with room to spare."""
        hint = self.minimumSizeHint().width()
        return max(hint + _WIDTH_SLACK, self.minimumWidth())

    # --- the bit grid ------------------------------------------------------

    def _wire_bits(self) -> None:
        """Route the grid's own signals to the bits mask it edits, adding its index."""
        editor = self._bits_editor
        editor.bit_clicked.connect(self._on_bit)
        editor.channel_toggle.connect(self._on_channel)
        editor.column_toggle.connect(self._on_column)
        editor.original.connect(self._on_original)
        editor.lsbs.connect(self._on_lsbs)
        editor.plane_step.connect(self._on_plane_step)
        editor.channel_step.connect(self._on_channel_step)

    def _on_bit(self, plane: str, bit: int, exclusive: bool) -> None:
        self.bit_clicked.emit(self._bits_layer, plane, bit, exclusive)

    def _on_channel(self, name: str, checked: bool) -> None:
        self.channel_toggle.emit(self._bits_layer, name, checked)

    def _on_column(self, bit: int, checked: bool) -> None:
        self.column_toggle.emit(self._bits_layer, bit, checked)

    def _on_original(self) -> None:
        self.original_requested.emit(self._bits_layer)

    def _on_lsbs(self) -> None:
        self.lsbs_requested.emit(self._bits_layer)

    def _on_plane_step(self, delta: int) -> None:
        self.plane_step.emit(self._bits_layer, delta)

    def _on_channel_step(self, delta: int) -> None:
        self.channel_step.emit(self._bits_layer, delta)

    # --- the stream --------------------------------------------------------

    def _order_row(self) -> QWidget:
        """The three order controls, held to one band above the dump they describe."""
        row = QWidget()
        # The stylesheet's own padding leaves a combo shorter than CONTROL_HEIGHT;
        # one fixed band keeps the three controls level with the search box below.
        row.setFixedHeight(theme.CONTROL_HEIGHT)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._channel_combo)
        layout.addWidget(self._bit_order)
        layout.addWidget(self._scan)
        layout.addStretch(1)
        self._save_button = QPushButton(text.EXTRACT_SAVE)
        self._save_button.setObjectName("ghost")
        self._save_button.setToolTip(text.EXTRACT_SAVE_TIP)
        self._save_button.setFixedHeight(theme.CONTROL_HEIGHT)
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_button)
        return row

    def _on_channel_order(self, index: int) -> None:
        """Ask for the arrangement picked in the list; the state answers through sync."""
        if 0 <= index < len(self._order_items):
            self.channel_order_requested.emit(self._order_items[index])

    def _sync_order_controls(
        self,
        planes: tuple[SamplePlane, ...],
        chosen: frozenset[BitChoice],
    ) -> None:
        """Sync the order controls. This runs on every cursor move, so it caches."""
        choices = order_choices(planes, chosen)
        applied = applied_order(planes, chosen, self._state.extract_order)
        # A preference kept from an earlier, slimmer channel set can sit outside
        # the list; the list then leads with it, so the shown order stays true.
        items = choices if applied in choices else ((applied,) if applied else ()) + choices
        order = self._state.extract_order
        key = (items, applied, order.bit_order, order.scan)
        if key == self._order_key:
            return
        self._order_key = key
        combo = self._channel_combo
        combo.blockSignals(True)
        if items != self._order_items:
            self._order_items = items
            combo.clear()
            for names in items:
                combo.addItem("".join(names), names)
        combo.setEnabled(len(items) > 1)
        combo.setCurrentIndex(items.index(applied) if applied in items else 0)
        combo.blockSignals(False)
        self._bit_order.set_value(order.bit_order.value)
        self._scan.set_value(order.scan.value)

    def _sync_extract(self, state: ViewerState, raster: Raster | None) -> None:
        """Re-extract only when the raster, the order, or the search changed."""
        key = None if raster is None else (raster, state.extract_order)
        if key != self._extract_key:
            self._extract_key = key
            data = b"" if raster is None else extract_bytes(raster, state.extract_order)
            self._extract_data = data
            self._detections = detect_patterns(data)
            self._classification = self._classifier()(data) if data else None
            self._extract_rows = tuple(format_extract(data))
            self._save_button.setEnabled(bool(data))
        self._refresh_view()
        self._extract_view.set_findings(
            text.classification_note(self._classification), self._detections
        )

    def _classifier(self) -> StreamClassifier:
        """The stream reader, resolved on first use so magika loads only then."""
        if self._classify is None:
            self._classify = stream_classifier()
        return self._classify

    def _refresh_view(self) -> None:
        """Re-render the panes when the rows, the search, or the encoding changed."""
        query = self._extract_search.text()
        encoding = self._state.extract_encoding
        key = (self._extract_key, query, encoding)
        if key == self._view_key:
            return
        self._view_key = key
        rows = filter_extract(list(self._extract_rows), query)
        self._extract_view.set_rows(rows, encoding)


class _BitsEditor(QWidget):
    """The bit grid with its presets and the arrows that walk planes and channels.

    The grid edits one bits mask of the stack, named by the index the window hands
    to :meth:`ExtractPanel.set_state`; ``None`` stands for the image itself, where
    every bit is on and an edit brings a mask of its own into being.
    """

    bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)
    original = Signal()
    lsbs = Signal()
    plane_step = Signal(int)
    channel_step = Signal(int)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._matrix = BitMatrix(self)
        self._matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._matrix.channel_toggle.connect(self.channel_toggle.emit)
        self._matrix.column_toggle.connect(self.column_toggle.emit)
        self.plane_prev = _stepper(text.PLANE_PREV, text.PLANE_PREV_TIP, self.plane_step, -1)
        self.plane_next = _stepper(text.PLANE_NEXT, text.PLANE_NEXT_TIP, self.plane_step, 1)
        self.channel_prev = _stepper(
            text.CHANNEL_PREV, text.CHANNEL_PREV_TIP, self.channel_step, -1
        )
        self.channel_next = _stepper(text.CHANNEL_NEXT, text.CHANNEL_NEXT_TIP, self.channel_step, 1)
        self._shadow_note = QLabel(self)
        self._shadow_note.setObjectName("detailWarn")
        self._shadow_note.setWordWrap(True)
        self._shadow_note.setVisible(False)

    def header(self) -> QHBoxLayout:
        """The section title with the two presets on its right."""
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        header.addWidget(section_title(text.SECTION_BITS))
        header.addStretch(1)
        self.original_button = QPushButton(text.ORIGINAL)
        self.lsbs_button = QPushButton(text.ALL_LSB)
        for button, name, signal in (
            (self.original_button, "segmentLeft", self.original),
            (self.lsbs_button, "segmentRight", self.lsbs),
        ):
            button.setObjectName(name)
            button.setToolTip(text.BITS_PRESET_TIP)
            button.setFixedHeight(theme.CONTROL_HEIGHT)
            button.clicked.connect(lambda _checked=False, signal=signal: signal.emit())
            header.addWidget(button)
        return header

    def body(self) -> QHBoxLayout:
        """The grid with the channel arrows centred on its left edge."""
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(_STEPPER_GAP)
        body.addWidget(
            _stacked(self.channel_prev, self.channel_next), 0, Qt.AlignmentFlag.AlignVCenter
        )
        card = QFrame(self)
        card.setObjectName("card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(10, 10, 10, 10)
        inner.setSpacing(0)
        # A 16-bit channel has more columns than the panel has room for, so the
        # grid scrolls sideways rather than being squeezed past what it can show:
        # it keeps its own width and the low bits stay reachable.
        self._grid_area = QScrollArea(card)
        self._grid_area.setObjectName("gridArea")
        self._grid_area.setWidgetResizable(True)
        self._grid_area.setFrameShape(QFrame.Shape.NoFrame)
        self._grid_area.setWidget(self._matrix)
        inner.addWidget(self._grid_area)
        body.addWidget(card, 1)
        return body

    def arrows(self) -> QHBoxLayout:
        """The left/right pair, centred under the grid and clear of the arrows column."""
        row = QHBoxLayout()
        row.setContentsMargins(_STEPPER_WIDTH + _STEPPER_GAP, 0, 0, 0)
        row.setSpacing(10)
        row.addStretch(1)
        for button in (self.plane_prev, self.plane_next):
            row.addWidget(button)
        row.addStretch(1)
        return row

    def set_layer(self, planes: tuple[SamplePlane, ...], selection: frozenset[BitChoice]) -> None:
        self._matrix.set_layer(planes, selection)
        # The scroll area hides the grid's height from the panel's layout, so the
        # card has to be told how tall every channel of this image is.
        self._grid_area.setMinimumHeight(self._matrix.minimumSizeHint().height() + 2)

    def set_shadowed(self, shadowed: bool) -> None:
        """Say so when an enabled bits mask above this one is the one in force."""
        self._shadow_note.setText(text.MASK_BITS_SHADOWED if shadowed else "")
        self._shadow_note.setVisible(shadowed)


def _stacked(first: QWidget, second: QWidget) -> QWidget:
    """Two buttons as a column, centred on the grid's height."""
    host = QWidget(first.parentWidget())
    column = QVBoxLayout(host)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(6)
    column.addWidget(first)
    column.addWidget(second)
    return host


def _stepper(label: str, tip: str, signal: SignalInstance, delta: int) -> QPushButton:
    button = QPushButton(label)
    button.setObjectName("stepper")
    button.setToolTip(tip)
    button.setFixedSize(_STEPPER_WIDTH, _STEPPER_WIDTH)
    button.clicked.connect(lambda _checked=False, delta=delta: signal.emit(delta))
    return button

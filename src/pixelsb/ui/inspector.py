"""Bit selection for the canvas, the presets, and the extract panel."""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal, SignalInstance
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
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
    SamplePlane,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.selection import effective_selection
from pixelsb.io.classify import stream_classifier
from pixelsb.ui import text, theme
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.controls import Toggle, TogglePair, hairline, section_title
from pixelsb.ui.extract_view import ExtractView

_STEPPER_WIDTH = 30  # the arrow buttons stay compact around the grid
_STEPPER_GAP = 6  # the channel column's gap, reused to indent the plane row
_BITS_MIN_WIDTH = 320  # what the bit grid itself needs, margins included
_WIDTH_SLACK = 16  # the panel's own scrollbar can appear once an image is open

_BIT_ORDER_TOGGLES = (
    Toggle(BitOrder.MSB, "MSB", text.ORDER_BIT_MSB_TIP),
    Toggle(BitOrder.LSB, "LSB", text.ORDER_BIT_LSB_TIP),
)
_SCAN_TOGGLES = (
    Toggle(ScanOrder.XY, "XY", text.ORDER_SCAN_XY_TIP),
    Toggle(ScanOrder.YZ, "YZ", text.ORDER_SCAN_YZ_TIP),
)


class Inspector(QWidget):
    """Panel on the right: which bits drive the view, and the extracted bytes."""

    bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)
    original_requested = Signal()
    lsbs_requested = Signal()
    plane_step = Signal(int)
    channel_step = Signal(int)
    encoding_requested = Signal(str)
    channel_order_requested = Signal(object)  # a tuple of channel names
    bit_order_requested = Signal(str)
    scan_requested = Signal(str)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("inspector")
        self._state = ViewerState()
        self._canvas_matrix = BitMatrix()
        self._canvas_matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._canvas_matrix.channel_toggle.connect(self.channel_toggle.emit)
        self._canvas_matrix.column_toggle.connect(self.column_toggle.emit)
        self._canvas_card = self._matrix_card(self._canvas_matrix)
        self._plane_prev = self._stepper_button(
            text.PLANE_PREV, text.PLANE_PREV_TIP, self.plane_step, -1
        )
        self._plane_next = self._stepper_button(
            text.PLANE_NEXT, text.PLANE_NEXT_TIP, self.plane_step, 1
        )
        self._channel_prev = self._stepper_button(
            text.CHANNEL_PREV, text.CHANNEL_PREV_TIP, self.channel_step, -1
        )
        self._channel_next = self._stepper_button(
            text.CHANNEL_NEXT, text.CHANNEL_NEXT_TIP, self.channel_step, 1
        )

        self._extract_key: tuple[object, ...] | None = None
        self._view_key: tuple[object, ...] | None = None
        self._order_key: tuple[object, ...] | None = None
        self._extract_rows: tuple[ExtractRow, ...] = ()
        self._extract_data = b""
        self._classification: Classification | None = None
        self._detections: tuple[Detection, ...] = ()
        self._classify: StreamClassifier | None = None
        self._order_items: tuple[tuple[str, ...], ...] = ()
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
        self._extract_search.textChanged.connect(lambda _text: self._refresh_extract_view())
        self._extract_view = ExtractView()
        self._extract_view.setMinimumHeight(180)
        self._extract_view.encoding_changed.connect(self.encoding_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(self._bits_header())
        layout.addLayout(self._bits_body())
        layout.addLayout(self._plane_row())
        layout.addSpacing(14)
        layout.addWidget(hairline())
        layout.addSpacing(14)
        layout.addWidget(section_title(text.SECTION_EXTRACT))
        layout.addWidget(self._order_row())
        layout.addWidget(self._extract_search)
        layout.addWidget(self._extract_view, 1)
        # Room for the grid plus the arrow column, so it never squashes.
        self.setMinimumWidth(_BITS_MIN_WIDTH + _STEPPER_WIDTH + _STEPPER_GAP)

    def set_state(
        self,
        state: ViewerState,
        match: NDArray[np.bool_] | None = None,
    ) -> None:
        self._state = state
        image = state.image
        planes = () if image is None else image.planes
        canvas_bits = None if image is None else effective_selection(image, state.selection)
        self._canvas_matrix.set_layer(planes, canvas_bits)
        self._sync_order_controls(planes, canvas_bits)
        self._sync_extract(state, canvas_bits, match)
        self._extract_view.sync_encoding(state.extract_encoding)

    def _bits_header(self) -> QHBoxLayout:
        """Section title with the presets on the right."""
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(section_title(text.SECTION_BITS))
        header.addStretch(1)
        presets = QWidget()
        row = QHBoxLayout(presets)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for label, name, signal in (
            (text.ORIGINAL, "segmentLeft", self.original_requested),
            (text.ALL_LSB, "segmentRight", self.lsbs_requested),
        ):
            button = QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(lambda _checked=False, signal=signal: signal.emit())
            row.addWidget(button)
        header.addWidget(presets)
        return header

    def _bits_body(self) -> QHBoxLayout:
        """The grid with the channel arrows centred on its left edge."""
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(_STEPPER_GAP)
        body.addWidget(self._channel_stepper(), 0, Qt.AlignmentFlag.AlignVCenter)
        body.addWidget(self._canvas_card, 1)
        return body

    def _channel_stepper(self) -> QWidget:
        """The up/down pair as a column, centred on the grid's height."""
        stepper = QWidget()
        column = QVBoxLayout(stepper)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self._channel_prev)
        column.addWidget(self._channel_next)
        return stepper

    def _plane_row(self) -> QHBoxLayout:
        """The left/right pair, centred under the grid it clears on the left."""
        row = QHBoxLayout()
        row.setContentsMargins(_STEPPER_WIDTH + _STEPPER_GAP, 0, 0, 0)
        row.setSpacing(10)
        row.addStretch(1)
        row.addWidget(self._plane_prev)
        row.addWidget(self._plane_next)
        row.addStretch(1)
        return row

    def _stepper_button(
        self,
        label: str,
        tip: str,
        signal: SignalInstance,
        delta: int,
    ) -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("stepper")
        button.setToolTip(tip)
        button.setFixedSize(_STEPPER_WIDTH, _STEPPER_WIDTH)
        button.clicked.connect(lambda _checked=False, delta=delta: signal.emit(delta))
        return button

    def _matrix_card(self, matrix: BitMatrix) -> QFrame:
        """A soft gray card around the bit grid, spanning the panel."""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        layout.addWidget(matrix)
        return card

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
        chosen: frozenset[BitChoice] | None,
    ) -> None:
        """Sync the order controls. This runs on every cursor move, so it caches."""
        selection = frozenset() if chosen is None else chosen
        choices = order_choices(planes, selection)
        applied = applied_order(planes, selection, self._state.extract_order)
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

    def _sync_extract(
        self,
        state: ViewerState,
        chosen: frozenset[BitChoice] | None,
        match: NDArray[np.bool_] | None,
    ) -> None:
        """Re-extract only when the image, the bits, the filter, or the order changed."""
        image = state.image
        key = None if image is None else (image, chosen, state.filter_expr, state.extract_order)
        if key != self._extract_key:
            self._extract_key = key
            data = (
                b""
                if image is None
                else extract_bytes(image, chosen, match, order=state.extract_order)
            )
            self._extract_data = data
            self._detections = detect_patterns(data)
            self._classification = self._classifier()(data) if data else None
            self._extract_rows = tuple(format_extract(data))
            self._save_button.setEnabled(bool(data))
        self._refresh_extract_view()
        self._extract_view.set_findings(
            text.classification_note(self._classification), self._detections
        )

    def _classifier(self) -> StreamClassifier:
        """The stream reader, resolved on first use so magika loads only then."""
        if self._classify is None:
            self._classify = stream_classifier()
        return self._classify

    def extract_data(self) -> bytes:
        """The full extracted stream, untruncated by the dump's display limit."""
        return self._extract_data

    def _refresh_extract_view(self) -> None:
        """Re-render the panes when the rows, the search, or the encoding changed."""
        query = self._extract_search.text()
        encoding = self._state.extract_encoding
        key = (self._extract_key, query, encoding)
        if key == self._view_key:
            return
        self._view_key = key
        rows = filter_extract(list(self._extract_rows), query)
        self._extract_view.set_rows(rows, encoding)

    def preferred_width(self) -> int:
        """Panel width that holds both dump panes with a little room to spare."""
        hint = self.minimumSizeHint().width()
        return max(hint + _WIDTH_SLACK, self.minimumWidth())

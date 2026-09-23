"""Bit grids for the canvas / number layers, presets, and the extract panel."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.extract import extract_bytes, filter_extract, format_extract
from pixelsb.domain.models import BitChoice, LoadedImage, ViewerState
from pixelsb.domain.selection import all_bits, effective_selection
from pixelsb.ui import text
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.text import readout_text


class Inspector(QWidget):
    """Canvas grid, number grid (when detached), and the preset actions."""

    bit_clicked = Signal(str, int, bool)
    readout_bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    readout_channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)
    readout_column_toggle = Signal(int, bool)
    original_requested = Signal()
    lsbs_requested = Signal()
    reset_readout_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._state = ViewerState()
        self._canvas_matrix = BitMatrix()
        self._canvas_matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._canvas_matrix.channel_toggle.connect(self.channel_toggle.emit)
        self._canvas_matrix.column_toggle.connect(self.column_toggle.emit)
        self._number_matrix = BitMatrix()
        self._number_matrix.bit_clicked.connect(self.readout_bit_clicked.emit)
        self._number_matrix.channel_toggle.connect(self.readout_channel_toggle.emit)
        self._number_matrix.column_toggle.connect(self.readout_column_toggle.emit)

        self._extract_key: tuple[object, ...] | None = None
        self._extract_lines: list[str] = []
        self._extract_search = QLineEdit()
        self._extract_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._extract_search.textChanged.connect(lambda _text: self._refresh_extract_view())
        self._extract_view = QPlainTextEdit()
        self._extract_view.setReadOnly(True)
        extract_font = QFont()
        extract_font.setStyleHint(QFont.StyleHint.Monospace)
        extract_font.setFamilies(["Menlo", "monospace"])
        self._extract_view.setFont(extract_font)
        self._extract_view.setMinimumHeight(180)

        presets = QHBoxLayout()
        presets.setContentsMargins(0, 0, 0, 0)
        for label, signal in (
            (text.ORIGINAL, self.original_requested),
            (text.ALL_LSB, self.lsbs_requested),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, signal=signal: signal.emit())
            presets.addWidget(button)

        self._canvas_caption = QLabel(text.CANVAS_LAYER)
        self._number_row = QWidget()
        number_layout = QHBoxLayout(self._number_row)
        number_layout.setContentsMargins(0, 0, 0, 0)
        number_layout.addWidget(QLabel(text.NUMBER_LAYER))
        reset = QPushButton(text.READOUT_RESET)
        reset.setFlat(True)
        reset.setToolTip(text.READOUT_RESET_TIP)
        reset.clicked.connect(self.reset_readout_requested.emit)
        number_layout.addWidget(reset)
        number_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(presets)
        layout.addWidget(self._canvas_caption)
        layout.addWidget(self._canvas_matrix)
        layout.addWidget(self._number_row)
        layout.addWidget(self._number_matrix)
        layout.addWidget(QLabel(text.EXTRACT_NOTE))
        layout.addWidget(self._extract_search)
        layout.addWidget(self._extract_view, 1)
        self.setMinimumWidth(280)

    def set_state(self, state: ViewerState) -> None:
        self._state = state
        image = state.image
        planes = () if image is None else image.planes
        canvas_bits = None if image is None else effective_selection(image, state.selection)
        self._canvas_matrix.set_layer(planes, canvas_bits)
        detached = state.detached
        if detached:
            number_bits = None
            if image is not None:
                number_bits = all_bits(image) if state.readout is None else state.readout
            self._number_matrix.set_layer(planes, number_bits)
        self._canvas_caption.setVisible(detached)
        self._number_row.setVisible(detached)
        self._number_matrix.setVisible(detached)
        self._sync_extract(image, canvas_bits)

    def _sync_extract(
        self,
        image: LoadedImage | None,
        chosen: frozenset[BitChoice] | None,
    ) -> None:
        """Re-extract only when the image or the selection changed."""
        key = None if image is None else (id(image), chosen)
        if key == self._extract_key:
            return
        self._extract_key = key
        data = b"" if image is None else extract_bytes(image, chosen)
        self._extract_lines = format_extract(data)
        self._refresh_extract_view()

    def _refresh_extract_view(self) -> None:
        lines = filter_extract(self._extract_lines, self._extract_search.text())
        self._extract_view.setPlainText("\n".join(lines))

    def detail_text(self) -> str:
        return readout_text(self._state)

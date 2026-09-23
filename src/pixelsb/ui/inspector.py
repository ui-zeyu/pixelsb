"""Bit selection for both layers, the layer switch, and the extract panel."""

import re

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
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
from pixelsb.ui import text, theme
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.text import readout_text


class Inspector(QWidget):
    """Panel on the right: which bits drive what, and the extracted bytes."""

    bit_clicked = Signal(str, int, bool)
    readout_bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    readout_channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)
    readout_column_toggle = Signal(int, bool)
    original_requested = Signal()
    lsbs_requested = Signal()
    reset_readout_requested = Signal()
    detached_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("inspector")
        self._state = ViewerState()
        self._canvas_matrix = BitMatrix()
        self._canvas_matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._canvas_matrix.channel_toggle.connect(self.channel_toggle.emit)
        self._canvas_matrix.column_toggle.connect(self.column_toggle.emit)
        self._number_matrix = BitMatrix()
        self._number_matrix.bit_clicked.connect(self.readout_bit_clicked.emit)
        self._number_matrix.channel_toggle.connect(self.readout_channel_toggle.emit)
        self._number_matrix.column_toggle.connect(self.readout_column_toggle.emit)
        self._canvas_caption = _caption(text.CANVAS_LAYER)
        self._number_row = self._build_number_row()
        self._detach = QCheckBox(text.DETACH)
        self._detach.setToolTip(text.DETACH_TIP)
        self._detach.toggled.connect(self.detached_toggled.emit)

        self._extract_key: tuple[object, ...] | None = None
        self._extract_lines: list[str] = []
        self._extract_search = QLineEdit()
        self._extract_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._extract_search.setFixedHeight(theme.CONTROL_HEIGHT)
        self._extract_search.textChanged.connect(lambda _text: self._refresh_extract_view())
        self._extract_note = _caption(text.EXTRACT_NOTE)
        self._extract_note.setWordWrap(True)
        self._extract_view = QPlainTextEdit()
        self._extract_view.setReadOnly(True)
        self._extract_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._extract_view.setMinimumHeight(180)
        extract_font = QFont()
        extract_font.setStyleHint(QFont.StyleHint.Monospace)
        extract_font.setFamilies(["Menlo", "Consolas"])
        extract_font.setPixelSize(11)
        self._extract_view.setFont(extract_font)
        self._highlighter = ExtractHighlighter(self._extract_view.document())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(self._section_header(text.SECTION_BITS, self._detach))
        layout.addLayout(self._preset_row())
        layout.addWidget(self._canvas_caption)
        layout.addWidget(self._canvas_matrix)
        layout.addWidget(self._number_row)
        layout.addWidget(self._number_matrix)
        layout.addSpacing(14)
        layout.addWidget(_hairline())
        layout.addSpacing(14)
        layout.addWidget(_section_title(text.SECTION_EXTRACT))
        layout.addWidget(self._extract_note)
        layout.addWidget(self._extract_search)
        layout.addWidget(self._extract_view, 1)
        self.setMinimumWidth(320)

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
        detached = state.detached
        if detached:
            number_bits = None
            if image is not None:
                number_bits = all_bits(image) if state.readout is None else state.readout
            self._number_matrix.set_layer(planes, number_bits)
        self._canvas_caption.setVisible(detached)
        self._number_row.setVisible(detached)
        self._number_matrix.setVisible(detached)
        self._detach.blockSignals(True)
        self._detach.setChecked(detached)
        self._detach.blockSignals(False)
        self._sync_extract(image, canvas_bits, state.filter_expr, match)

    def _section_header(self, title: str, trailing: QWidget) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section_title(title))
        header.addStretch(1)
        header.addWidget(trailing)
        return header

    def _preset_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for label, name, signal in (
            (text.ORIGINAL, "segmentLeft", self.original_requested),
            (text.ALL_LSB, "segmentRight", self.lsbs_requested),
        ):
            button = QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(lambda _checked=False, signal=signal: signal.emit())
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _build_number_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        reset = QPushButton(text.READOUT_RESET)
        reset.setObjectName("linkButton")
        reset.setToolTip(text.READOUT_RESET_TIP)
        reset.clicked.connect(lambda _checked=False: self.reset_readout_requested.emit())
        layout.addWidget(_caption(text.NUMBER_LAYER))
        layout.addWidget(reset)
        layout.addStretch(1)
        return row

    def _sync_extract(
        self,
        image: LoadedImage | None,
        chosen: frozenset[BitChoice] | None,
        expression: str,
        match: NDArray[np.bool_] | None,
    ) -> None:
        """Re-extract only when the image, the selection, or the filter changed."""
        key = None if image is None else (image, chosen, expression)
        if key == self._extract_key:
            return
        self._extract_key = key
        data = b"" if image is None else extract_bytes(image, chosen, match)
        self._extract_lines = format_extract(data)
        self._refresh_extract_view()

    def _refresh_extract_view(self) -> None:
        lines = filter_extract(self._extract_lines, self._extract_search.text())
        self._extract_view.setPlainText("\n".join(lines))

    def detail_text(self) -> str:
        return readout_text(self._state)


class ExtractHighlighter(QSyntaxHighlighter):
    """Colors the offset, hex, and ASCII columns of the extract view."""

    _DATA_LINE = re.compile(r"[0-9a-f]{8}  ")
    _OFFSET_WIDTH = 8
    _HEX_START = 10
    _HEX_WIDTH = 47
    _ASCII_START = 59

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._offset = _char_format(theme.TEXT_MUTED)
        self._hex = _char_format(theme.TEXT)
        self._ascii = _char_format(theme.TEXT_MUTED)

    def highlightBlock(self, line: str) -> None:
        if not self._DATA_LINE.match(line):
            return
        self.setFormat(0, self._OFFSET_WIDTH, self._offset)
        self.setFormat(self._HEX_START, self._HEX_WIDTH, self._hex)
        if len(line) > self._ASCII_START:
            self.setFormat(self._ASCII_START, len(line) - self._ASCII_START, self._ascii)


def _char_format(color: str) -> QTextCharFormat:
    text_format = QTextCharFormat()
    text_format.setForeground(QColor(color))
    return text_format


def _section_title(label: str) -> QLabel:
    title = QLabel(label)
    title.setObjectName("sectionTitle")
    return title


def _caption(label: str) -> QLabel:
    caption = QLabel(label)
    caption.setObjectName("muted")
    return caption


def _hairline() -> QFrame:
    line = QFrame()
    line.setObjectName("hairline")
    line.setFixedHeight(1)
    return line

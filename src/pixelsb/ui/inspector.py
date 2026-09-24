"""Bit selection for both layers, the layer switch, and the extract panel."""

import re
from math import ceil

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.extract import (
    ASCII_START,
    BYTES_PER_ROW,
    HEX_WIDTH,
    ExtractRow,
    extract_bytes,
    filter_extract,
    format_extract,
)
from pixelsb.domain.models import BitChoice, LoadedImage, ViewerState
from pixelsb.domain.selection import effective_selection
from pixelsb.ui import text, theme
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.text import readout_text

_GUTTER_PAD = 10
_HEADER_GAP = 3
_WIDTH_SLACK = 16  # the panel's own scrollbar can appear once an image is open


class Inspector(QWidget):
    """Panel on the right: which bits drive the view, and the extracted bytes."""

    bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)
    original_requested = Signal()
    lsbs_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("inspector")
        self._state = ViewerState()
        self._canvas_matrix = BitMatrix()
        self._canvas_matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._canvas_matrix.channel_toggle.connect(self.channel_toggle.emit)
        self._canvas_matrix.column_toggle.connect(self.column_toggle.emit)
        self._canvas_card = self._matrix_card(self._canvas_matrix)

        self._extract_key: tuple[object, ...] | None = None
        self._extract_rows: tuple[ExtractRow, ...] = ()
        self._extract_search = QLineEdit()
        self._extract_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._extract_search.setFixedHeight(theme.CONTROL_HEIGHT)
        self._extract_search.textChanged.connect(lambda _text: self._refresh_extract_view())
        self._extract_note = _caption(text.EXTRACT_NOTE)
        self._extract_note.setWordWrap(True)
        self._extract_view = ExtractView()
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
        layout.addLayout(self._bits_header())
        layout.addWidget(self._canvas_card)
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
        self._sync_extract(image, canvas_bits, state.filter_expr, match)

    def _section_header(self, title: str, trailing: QWidget) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section_title(title))
        header.addStretch(1)
        header.addWidget(trailing)
        return header

    def _bits_header(self) -> QHBoxLayout:
        """Section title with the presets on the right."""
        trailing = QWidget()
        row = QHBoxLayout(trailing)
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
        return self._section_header(text.SECTION_BITS, trailing)

    def _matrix_card(self, matrix: BitMatrix) -> QFrame:
        """A soft gray card around the bit grid, spanning the panel."""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        layout.addWidget(matrix)
        return card

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
        self._extract_rows = tuple(format_extract(data))
        self._refresh_extract_view()

    def _refresh_extract_view(self) -> None:
        rows = filter_extract(list(self._extract_rows), self._extract_search.text())
        self._extract_view.set_rows(rows)

    def detail_text(self) -> str:
        return readout_text(self._state)

    def preferred_width(self) -> int:
        """Panel width that shows a whole dump row without a horizontal scrollbar."""
        view = self._extract_view
        missing = view.text_width() - view.viewport().geometry().width()
        return max(ceil(self.width() + missing) + _WIDTH_SLACK, self.minimumWidth())


class ExtractView(QPlainTextEdit):
    """Read-only byte dump whose offsets sit in a painted gutter.

    The document holds the hex and ASCII columns only, so selecting and copying
    never picks up an offset; the gutter and the column header are chrome the
    mouse cannot reach.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._rows: tuple[ExtractRow, ...] = ()
        self._gutter_width = self._measure_gutter()
        self._header_height = self._measure_header()
        self._gutter = _OffsetGutter(self)
        self._header = _ColumnHeader(self)
        self._gutter.resize(0, 0)  # nothing to paint until the first rows arrive
        self._header.resize(0, 0)
        # Reserve the chrome now, so the width the panel needs does not depend on
        # whether rows have arrived yet.
        self.setViewportMargins(self._gutter_width, self._header_height, 0, 0)
        self.updateRequest.connect(self._on_update_request)
        for bar in (self.verticalScrollBar(), self.horizontalScrollBar()):
            # A scrollbar that appears shrinks the viewport, so the chrome moves.
            bar.rangeChanged.connect(lambda _low, _high: self._place_chrome())
        self.verticalScrollBar().valueChanged.connect(lambda _value: self._gutter.update())
        self.horizontalScrollBar().valueChanged.connect(lambda _value: self._header.update())

    def set_rows(self, rows: list[ExtractRow]) -> None:
        self._rows = tuple(rows)
        self.setPlainText("\n".join(row.text for row in self._rows))
        self._gutter_width = self._measure_gutter()
        self._header_height = self._measure_header()
        self.setViewportMargins(self._gutter_width, self._header_height, 0, 0)
        self._place_chrome()
        self._gutter.update()
        self._header.update()

    def offset_text(self, block_number: int) -> str:
        """The gutter label for a document block; empty for note rows."""
        if 0 <= block_number < len(self._rows):
            return self._rows[block_number].offset_text
        return ""

    def header_text(self) -> str:
        """The column ruler, aligned with the hex columns by construction."""
        ruler = " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))
        return f"{ruler}  {text.EXTRACT_ASCII}"

    @property
    def gutter_width(self) -> int:
        """Width of the offset column, which the header shares."""
        return self._gutter_width

    def text_width(self) -> float:
        """Pixels one whole dump row needs: the hex columns and the ASCII column."""
        columns = HEX_WIDTH + 2 + BYTES_PER_ROW
        return QFontMetricsF(self.font()).horizontalAdvance("0" * columns)

    def row_width(self) -> float:
        """Pixels one whole dump row needs, offset gutter included."""
        return self._measure_gutter() + self.text_width()

    def header_origin(self) -> float:
        """Where the header text starts, in header coordinates."""
        margin = self.document().documentMargin() - self.horizontalScrollBar().value()
        return margin + self._gutter_width

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_chrome()

    def _measure_gutter(self) -> int:
        digits = max(
            (len(row.offset_text) for row in self._rows if row.offset is not None),
            default=0,
        )
        advance = QFontMetricsF(self.font()).horizontalAdvance("0")
        return ceil(advance * max(digits, 8)) + _GUTTER_PAD

    def _measure_header(self) -> int:
        return ceil(QFontMetricsF(self.font()).height()) + _HEADER_GAP

    def _place_chrome(self) -> None:
        viewport = self.viewport().geometry()
        self._gutter.setGeometry(
            viewport.left() - self._gutter_width,
            viewport.top(),
            self._gutter_width,
            viewport.height(),
        )
        self._header.setGeometry(
            viewport.left() - self._gutter_width,
            viewport.top() - self._header_height,
            self._gutter_width + viewport.width(),
            self._header_height,
        )

    def _on_update_request(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())


class _OffsetGutter(QWidget):
    """Left margin of the dump: the byte offset of every visible row."""

    def __init__(self, editor: ExtractView) -> None:
        super().__init__(editor)
        self._editor = editor

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(theme.FIELD))
        painter.setPen(QColor(theme.TEXT_MUTED))
        height = QFontMetricsF(self.font()).height()
        width = self.width() - _GUTTER_PAD
        block = self._editor.firstVisibleBlock()
        top = (
            self._editor.blockBoundingGeometry(block).translated(self._editor.contentOffset()).top()
        )
        bottom = top + self._editor.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                label = self._editor.offset_text(block.blockNumber())
                if label:
                    painter.drawText(
                        QRectF(0, top, width, height),
                        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                        label,
                    )
            block = block.next()
            top = bottom
            bottom = top + self._editor.blockBoundingRect(block).height()


class _ColumnHeader(QWidget):
    """Top margin of the dump: byte column numbers and the ASCII label."""

    def __init__(self, editor: ExtractView) -> None:
        super().__init__(editor)
        self._editor = editor

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(theme.FIELD))
        painter.setPen(QColor(theme.HAIRLINE))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.setPen(QColor(theme.TEXT_MUTED))
        flags = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        painter.drawText(
            QRectF(0, 0, self._editor.gutter_width - _GUTTER_PAD, self.height()),
            flags,
            text.EXTRACT_OFFSET,
        )
        origin = self._editor.header_origin()
        painter.drawText(
            QRectF(origin, 0, max(self.width() - origin, 0.0), self.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._editor.header_text(),
        )


class ExtractHighlighter(QSyntaxHighlighter):
    """Colors the hex and ASCII columns of the extract view."""

    _HEX_BYTE = re.compile(r"[0-9a-f]{2}")

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._hex = _char_format(theme.TEXT)
        self._ascii = _char_format(theme.TEXT_MUTED)

    def highlightBlock(self, line: str) -> None:
        if not self._HEX_BYTE.match(line):
            return
        self.setFormat(0, HEX_WIDTH, self._hex)
        if len(line) > ASCII_START:
            self.setFormat(ASCII_START, len(line) - ASCII_START, self._ascii)


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

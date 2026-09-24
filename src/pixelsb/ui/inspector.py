"""Bit selection for both layers, the layer switch, and the extract panel."""

from math import ceil
from typing import override

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, Signal, SignalInstance
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontInfo,
    QFontMetricsF,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPolygonF,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.extract import (
    BYTES_PER_ROW,
    HEX_WIDTH,
    ExtractRow,
    decode_row,
    extract_bytes,
    filter_extract,
    format_extract,
)
from pixelsb.domain.models import BitChoice, ExtractEncoding, LoadedImage, ViewerState
from pixelsb.domain.selection import effective_selection
from pixelsb.ui import text, theme
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.text import readout_text

_GUTTER_PAD = 10
_HEADER_GAP = 3
_PANE_GAP = 10  # between the hex pane and the ASCII pane
_PANE_SLACK = 2  # keeps the fixed ASCII column from clipping its last character
_DUMP_FONT_SIZE = 11
_STEPPER_WIDTH = 30  # the arrow buttons stay compact around the grid
_STEPPER_GAP = 6  # the channel column's gap, reused to indent the plane row
_BITS_MIN_WIDTH = 320  # what the bit grid itself needs, margins included
_WIDTH_SLACK = 16  # the panel's own scrollbar can appear once an image is open


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
        self._extract_rows: tuple[ExtractRow, ...] = ()
        self._extract_search = QLineEdit()
        self._extract_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._extract_search.setFixedHeight(theme.CONTROL_HEIGHT)
        self._extract_search.textChanged.connect(lambda _text: self._refresh_extract_view())
        self._extract_note = _caption(text.EXTRACT_NOTE)
        self._extract_note.setWordWrap(True)
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
        layout.addWidget(_hairline())
        layout.addSpacing(14)
        layout.addWidget(_section_title(text.SECTION_EXTRACT))
        layout.addWidget(self._extract_note)
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
        self._sync_extract(image, canvas_bits, state.filter_expr, match)
        self._extract_view.sync_encoding(state.extract_encoding)

    def _bits_header(self) -> QHBoxLayout:
        """Section title with the presets on the right."""
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section_title(text.SECTION_BITS))
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

    def _sync_extract(
        self,
        image: LoadedImage | None,
        chosen: frozenset[BitChoice] | None,
        expression: str,
        match: NDArray[np.bool_] | None,
    ) -> None:
        """Re-extract only when the image, the selection, or the filter changed."""
        key = None if image is None else (image, chosen, expression)
        if key != self._extract_key:
            self._extract_key = key
            data = b"" if image is None else extract_bytes(image, chosen, match)
            self._extract_rows = tuple(format_extract(data))
        self._refresh_extract_view()

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

    def detail_text(self) -> str:
        return readout_text(self._state)

    def preferred_width(self) -> int:
        """Panel width that holds both dump panes with a little room to spare."""
        hint = self.minimumSizeHint().width()
        return max(hint + _WIDTH_SLACK, self.minimumWidth())


class ExtractView(QWidget):
    """Read-only byte dump: hex on the left, one decoded text column on the right.

    Each pane holds its own text, so a selection copies only that column. The text
    pane's header names the encoding and switches it, and selecting or copying
    never picks up an offset: the gutter and the ruler are chrome.
    """

    encoding_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._encoding = ExtractEncoding.ASCII
        self.hex_pane = _HexPane(self)
        self.text_pane = _TextPane(self)
        for pane in (self.hex_pane, self.text_pane):
            # Set per pane: a font on the container does not survive the
            # stylesheet polish, and a proportional font breaks the columns.
            pane.setFont(dump_font())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_PANE_GAP)
        # Every column holds its own width; panel width beyond them stays empty.
        layout.addWidget(self.hex_pane, 0)
        layout.addWidget(self.text_pane, 0)
        layout.addStretch(1)
        self.text_pane.header_clicked.connect(self.show_encodings)
        # One document in two views: either scrollbar drags the other, and the
        # text pane takes the hex range so a hidden bar cannot skew them.
        hex_bar = self.hex_pane.verticalScrollBar()
        text_bar = self.text_pane.verticalScrollBar()
        hex_bar.rangeChanged.connect(lambda _low, high: text_bar.setMaximum(high))
        hex_bar.valueChanged.connect(text_bar.setValue)
        text_bar.valueChanged.connect(hex_bar.setValue)
        # Equal viewport heights are what keeps the rows aligned, so whatever the
        # hex pane's horizontal bar takes, the text pane reserves in its place.
        self.hex_pane.horizontalScrollBar().rangeChanged.connect(
            lambda _low, high: self.text_pane.set_bottom_inset(
                _bar_extent(self.text_pane) if high > 0 else 0
            )
        )

    def set_rows(self, rows: list[ExtractRow], encoding: ExtractEncoding) -> None:
        self.hex_pane.set_rows(rows)
        self.text_pane.set_rows(rows, encoding)
        self.sync_encoding(encoding)

    def sync_encoding(self, encoding: ExtractEncoding) -> None:
        """Follow the state's encoding without asking for it back."""
        self._encoding = encoding
        self.text_pane.set_encoding_label(encoding)

    def choose_encoding(self, encoding: ExtractEncoding) -> None:
        """Ask for an encoding; the state decides and answers through set_rows."""
        self.encoding_changed.emit(encoding.value)

    def show_encodings(self) -> None:
        """Drop the encoding list from the text pane's header."""
        self._encoding_menu().exec(self.text_pane.header_position())

    def _encoding_menu(self) -> QMenu:
        """The encoding list, with the state's entry ticked."""
        menu = QMenu(self)
        for encoding, label in text.EXTRACT_ENCODINGS:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(encoding is self._encoding)
            action.triggered.connect(
                lambda _checked=False, encoding=encoding: self.choose_encoding(encoding)
            )
        return menu


class _HexPane(QPlainTextEdit):
    """The hex columns, with the offset gutter and the byte ruler as chrome."""

    def __init__(self, panel: ExtractView) -> None:
        super().__init__(panel)
        _configure_pane(self)
        self._rows: tuple[ExtractRow, ...] = ()
        self._gutter_width = self._measure_gutter()
        self._header_height = self._measure_header()
        self._gutter = _OffsetGutter(self)
        self._header = _ColumnHeader(self)
        self._gutter.resize(0, 0)  # nothing to paint until the first rows arrive
        self._header.resize(0, 0)
        # Reserve the chrome now, so the width the panel needs does not depend on
        # whether rows have arrived yet. The bottom strip keeps this viewport the
        # same height as the ASCII pane's, so the two scroll in lockstep.
        self._set_insets()
        self.updateRequest.connect(self._on_update_request)
        for bar in (self.verticalScrollBar(), self.horizontalScrollBar()):
            # A scrollbar that appears shrinks the viewport, so the chrome moves.
            bar.rangeChanged.connect(lambda _low, _high: self._place_chrome())
        self.verticalScrollBar().valueChanged.connect(lambda _value: self._gutter.update())
        self.horizontalScrollBar().valueChanged.connect(lambda _value: self._header.update())

    def set_rows(self, rows: list[ExtractRow]) -> None:
        self._rows = tuple(rows)
        self.setPlainText("\n".join(row.hex_text for row in self._rows))
        self._sync_chrome()

    @override
    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._sync_chrome()

    def offset_text(self, block_number: int) -> str:
        """The gutter label for a document block; empty for note rows."""
        if 0 <= block_number < len(self._rows):
            return self._rows[block_number].offset_text
        return ""

    def header_text(self) -> str:
        """The column ruler, aligned with the hex columns by construction."""
        return " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))

    @property
    def gutter_width(self) -> int:
        """Width of the offset column, which the header shares."""
        return self._gutter_width

    @property
    def header_height(self) -> int:
        """Height of the top margin, which the tab strip beside it must match."""
        return self._header_height

    def text_width(self) -> float:
        """Pixels the hex columns need."""
        return QFontMetricsF(self.font()).horizontalAdvance("0" * HEX_WIDTH)

    def row_width(self) -> float:
        """Pixels one whole dump row needs, offset gutter included."""
        return self._measure_gutter() + self.text_width()

    def header_origin(self) -> float:
        """Where the header text starts, in header coordinates."""
        margin = self.document().documentMargin() - self.horizontalScrollBar().value()
        return margin + self._gutter_width

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_chrome()

    def _sync_chrome(self) -> None:
        """Re-measure the chrome and hold the pane at the width its data needs."""
        for widget in (self._gutter, self._header):
            # The chrome does not inherit the pane's font, and a wider font here
            # would clip the offsets and walk the ruler off the hex columns.
            widget.setFont(self.font())
        self._gutter_width = self._measure_gutter()
        self._header_height = self._measure_header()
        self._set_insets()
        self._place_chrome()
        self._gutter.update()
        self._header.update()
        self._fit_columns()

    def _fit_columns(self) -> None:
        """Hold the pane at the width its columns need, so it never stretches."""
        wanted = ceil(self._columns_width()) + self._chrome_width() + _PANE_SLACK
        if wanted != self.width():
            self.setFixedWidth(wanted)

    def _chrome_width(self) -> int:
        """Pixels the pane spends beside its viewport: gutter, frame and scrollbar."""
        margins = self.contentsMargins()
        return (
            self._gutter_width
            + margins.left()
            + margins.right()
            + self.verticalScrollBar().sizeHint().width()
        )

    def _columns_width(self) -> float:
        """Pixels the viewport must supply: the hex columns plus their margins."""
        return self.text_width() + 2 * self.document().documentMargin()

    def _measure_gutter(self) -> int:
        digits = max(
            (len(row.offset_text) for row in self._rows if row.offset is not None),
            default=0,
        )
        advance = QFontMetricsF(self.font()).horizontalAdvance("0")
        return ceil(advance * max(digits, 8)) + _GUTTER_PAD

    def _measure_header(self) -> int:
        return ceil(QFontMetricsF(self.font()).height()) + _HEADER_GAP

    def _set_insets(self) -> None:
        self.setViewportMargins(self._gutter_width, self._header_height, 0, 0)

    def _place_chrome(self) -> None:
        if not self._rows:  # nothing to label, so no chrome until rows arrive
            self._gutter.resize(0, 0)
            self._header.resize(0, 0)
            return
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


class _TextPane(QPlainTextEdit):
    """The decoded text column, in its own text so a selection copies only it."""

    header_clicked = Signal()

    def __init__(self, panel: ExtractView) -> None:
        super().__init__(panel)
        _configure_pane(self)
        self.setObjectName("textPane")
        # The hex pane carries the scrollbars; this one follows it.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._header_height = self._measure_header()
        self._header = _PaneHeader(self)
        self._header.resize(0, 0)
        self._header.clicked.connect(self.header_clicked.emit)
        self._bottom_inset = 0
        self._set_insets()

    def set_bottom_inset(self, inset: int) -> None:
        """Match the hex pane's horizontal scrollbar, so both viewports stay level."""
        if inset != self._bottom_inset:
            self._bottom_inset = inset
            self._set_insets()

    def set_rows(self, rows: list[ExtractRow], encoding: ExtractEncoding) -> None:
        self.setPlainText("\n".join(decode_row(row, encoding) for row in rows))
        self.set_encoding_label(encoding)
        self._sync_chrome()

    def set_encoding_label(self, encoding: ExtractEncoding) -> None:
        self._header.set_label(text.EXTRACT_ENCODING_LABELS[encoding])

    def header_position(self) -> QPoint:
        """Where the encoding list should drop from."""
        return self._header.mapToGlobal(QPoint(0, self._header.height()))

    @override
    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._sync_chrome()

    def column_width(self) -> float:
        """Pixels the text column needs: its widest glyph times the column width.

        Exact for a monospace font, and a stable reserve for any fallback: it
        depends on the font alone, so the pane never jitters as data changes.
        """
        metrics = QFontMetricsF(self.font())
        widest = max(
            (metrics.horizontalAdvance(chr(code)) for code in range(32, 127)),
            default=0.0,
        )
        return widest * BYTES_PER_ROW

    def _columns_width(self) -> float:
        """Pixels the viewport must supply: the text column plus its margins."""
        return self.column_width() + 2 * self.document().documentMargin()

    def _measure_header(self) -> int:
        return ceil(QFontMetricsF(self.font()).height()) + _HEADER_GAP

    def _set_insets(self) -> None:
        self.setViewportMargins(0, self._header_height, 0, self._bottom_inset)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_header()

    def _place_header(self) -> None:
        viewport = self.viewport().geometry()
        self._header.setGeometry(
            viewport.left(),
            viewport.top() - self._header_height,
            viewport.width(),
            self._header_height,
        )

    def _sync_chrome(self) -> None:
        """Re-measure the header and hold the pane at the width its column needs."""
        self._header.setFont(self.font())
        self._header_height = self._measure_header()
        self._set_insets()
        self._place_header()
        self._header.update()
        overhead = self.contentsMargins()
        wanted = ceil(self._columns_width()) + overhead.left() + overhead.right() + _PANE_SLACK
        if wanted != self.width():
            self.setFixedWidth(wanted)


def dump_font() -> QFont:
    """A fixed-pitch font for the dump, whatever the platform calls it."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    if not QFontInfo(font).fixedPitch():
        fallback = QFont()
        fallback.setStyleHint(QFont.StyleHint.Monospace)
        fallback.setFamilies(["Menlo", "Consolas", "DejaVu Sans Mono", "Courier New"])
        font = fallback
    font.setPixelSize(_DUMP_FONT_SIZE)
    return font


def _configure_pane(pane: QPlainTextEdit) -> None:
    pane.setReadOnly(True)
    pane.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)


def _bar_extent(pane: QPlainTextEdit) -> int:
    """The height a scrollbar takes, reserved so both panes keep one viewport height."""
    return pane.horizontalScrollBar().sizeHint().height()


class _PaneHeader(QWidget):
    """Top margin of the text pane: the encoding name, and the switch for it.

    Painting the name here keeps the header one height with the hex pane's, and
    the whole band is the control: a caret marks it, a click opens the list.
    """

    clicked = Signal()

    def __init__(self, editor: QPlainTextEdit) -> None:
        super().__init__(editor)
        self._label = ""
        self._hovered = False
        self.setToolTip(text.EXTRACT_ENCODING_TIP)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_label(self, label: str) -> None:
        self._label = label
        self.update()

    def label(self) -> str:
        return self._label

    @override
    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self.update()

    @override
    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self.update()

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(theme.HOVER if self._hovered else theme.FIELD))
        painter.setPen(QColor(theme.HAIRLINE))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        color = QColor(theme.ACCENT if self._hovered else theme.TEXT_MUTED)
        painter.setPen(color)
        painter.drawText(
            QRectF(0, 0, self.width(), self.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._label,
        )
        caret = 4.0
        left = QFontMetricsF(self.font()).horizontalAdvance(self._label) + 5
        top = (self.height() - caret) / 2
        if left + caret < self.width():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(left, top),
                        QPointF(left + caret, top),
                        QPointF(left + caret / 2, top + caret),
                    ]
                )
            )


class _OffsetGutter(QWidget):
    """Left margin of the dump: the byte offset of every visible row."""

    def __init__(self, editor: _HexPane) -> None:
        super().__init__(editor)
        self._editor = editor

    @override
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
    """Top margin of the hex pane: the byte column numbers."""

    def __init__(self, editor: _HexPane) -> None:
        super().__init__(editor)
        self._editor = editor

    @override
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

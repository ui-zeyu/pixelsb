"""The extract panel: a hex dump beside one decoded text column.

Each pane holds its own text, so a selection copies only that column. The text
pane's header names the encoding and switches it, and selecting or copying never
picks up an offset: the gutter and the byte ruler are chrome painted outside the
document.
"""

from bisect import bisect_right
from math import ceil
from typing import override

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, Signal
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
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.detect import Detection
from pixelsb.domain.extract import BYTES_PER_ROW, HEX_WIDTH, ExtractRow, decode_rows
from pixelsb.domain.models import ExtractEncoding
from pixelsb.ui import text, theme

_GUTTER_PAD = 10
_HEADER_GAP = 3
_PANE_GAP = 10  # between the hex pane and the text pane
_PANE_SLACK = 2  # keeps a fixed column from clipping its last character
_DUMP_FONT_SIZE = 11
_OFFSET_DIGITS = 8  # an eight-digit offset holds any image we can open
_MAX_CHIPS = 16  # chips past this fold into a "+N" label
_HIGHLIGHT_ALPHA = 42  # the row highlight tint over the dump's own background


class _Pane[HeaderT: QWidget](QPlainTextEdit):
    """One read-only dump column with a painted header band above its viewport.

    A pane holds the width its own columns need, so the two sit side by side at
    a fixed size and the panel's slack stays empty instead of stretching either
    one. Subclasses say what their columns are and where their header goes.
    """

    def __init__(self, panel: QWidget, header_type: type[HeaderT]) -> None:
        super().__init__(panel)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # Built after the widget exists: a child needs its parent constructed.
        self._header = header_type(self)
        self._header_height = self.measure_header()
        self._header.resize(0, 0)

    def measure_header(self) -> int:
        return ceil(QFontMetricsF(self.font()).height()) + _HEADER_GAP

    def sync_chrome(self) -> None:
        """Re-measure the chrome and hold the pane at the width its data needs."""
        for widget in self._chrome_widgets():
            # The chrome does not inherit the pane's font, and a wider font here
            # would clip the offsets and walk the ruler off the hex columns.
            widget.setFont(self.font())
        self._header_height = self.measure_header()
        self._set_insets()
        self.place_header()
        for widget in self._chrome_widgets():
            widget.update()
        self._fit_width()

    def place_header(self) -> None:
        """Put the band across the top of the viewport, its default home."""
        viewport = self.viewport().geometry()
        self._header.setGeometry(
            viewport.left(),
            viewport.top() - self._header_height,
            viewport.width(),
            self._header_height,
        )

    def width_wanted(self) -> float:
        """Pixels one whole dump row needs, offsets and frame included."""
        return self._columns_width() + self._chrome_width() + _PANE_SLACK

    @override
    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.sync_chrome()

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.place_header()

    def _fit_width(self) -> None:
        wanted = ceil(self.width_wanted())
        if wanted != self.width():
            self.setFixedWidth(wanted)

    def _set_insets(self) -> None:
        """Reserve the header band above the viewport."""
        self.setViewportMargins(0, self._header_height, 0, 0)

    def _chrome_widgets(self) -> tuple[QWidget, ...]:
        return (self._header,)

    def _columns_width(self) -> float:
        """Pixels the viewport must supply for this pane's columns."""
        raise NotImplementedError

    def _chrome_width(self) -> int:
        """Pixels the pane spends beside its viewport, the frame alone by default."""
        margins = self.contentsMargins()
        return margins.left() + margins.right()


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


class _HexPane(_Pane[_ColumnHeader]):
    """The hex columns, with the offset gutter and the byte ruler as chrome."""

    def __init__(self, panel: QWidget) -> None:
        super().__init__(panel, _ColumnHeader)
        self._rows: tuple[ExtractRow, ...] = ()
        self._gutter_width = self._measure_gutter()
        self._gutter = _OffsetGutter(self)
        self._gutter.resize(0, 0)  # nothing to paint until the first rows arrive
        # Reserve the chrome now, so the width the panel needs does not depend on
        # whether rows have arrived yet. The bottom strip keeps this viewport the
        # same height as the text pane's, so the two scroll in lockstep.
        self._set_insets()
        self.updateRequest.connect(self._on_update_request)
        for bar in (self.verticalScrollBar(), self.horizontalScrollBar()):
            # A scrollbar that appears shrinks the viewport, so the chrome moves.
            bar.rangeChanged.connect(lambda _low, _high: self.place_header())
        self.verticalScrollBar().valueChanged.connect(lambda _value: self._gutter.update())
        self.horizontalScrollBar().valueChanged.connect(lambda _value: self._header.update())

    def set_rows(self, rows: list[ExtractRow]) -> None:
        self._rows = tuple(rows)
        self.setPlainText("\n".join(row.hex_text for row in self._rows))
        self.sync_chrome()

    def offset_text(self, block_number: int) -> str:
        """The gutter label for a document block; empty for note rows."""
        if 0 <= block_number < len(self._rows):
            return self._rows[block_number].offset_text
        return ""

    def reveal(self, row: int, column: int, length: int) -> None:
        """Scroll to one dump row and select the bytes the detection covers.

        The hex columns lay a byte out as two digits and one separator, so the
        anchor arithmetic mirrors that rhythm; two digits is the floor for a
        one-byte find.
        """
        block = self.document().findBlockByNumber(row)
        if not block.isValid():
            return
        start = block.position() + column * 3
        cursor = self.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(start + max(3 * length - 1, 2), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)
        self.centerCursor()

    def set_highlights(self, spans: list[tuple[int, int, int, bool]]) -> None:
        """Tint the rows' hex columns where detections landed.

        ``(row, column, length, flagged)`` per span, in display coordinates.
        """
        document = self.document()
        selections = []
        for row, column, length, flagged in spans:
            block = document.findBlockByNumber(row)
            if not block.isValid():
                continue
            cursor = QTextCursor(document)
            start = block.position() + column * 3
            cursor.setPosition(start)
            cursor.setPosition(start + max(3 * length - 1, 2), QTextCursor.MoveMode.KeepAnchor)
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = highlight_format(flagged)
            selections.append(selection)
        self.setExtraSelections(selections)

    def header_text(self) -> str:
        """The column ruler, aligned with the hex columns by construction."""
        return " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))

    @property
    def gutter_width(self) -> int:
        """Width of the offset column, which the header shares."""
        return self._gutter_width

    def text_width(self) -> float:
        """Pixels the hex columns need."""
        return QFontMetricsF(self.font()).horizontalAdvance("0" * HEX_WIDTH)

    def header_origin(self) -> float:
        """Where the header text starts, in header coordinates."""
        margin = self.document().documentMargin() - self.horizontalScrollBar().value()
        return margin + self._gutter_width

    @override
    def place_header(self) -> None:
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

    @override
    def _set_insets(self) -> None:
        self.setViewportMargins(self._gutter_width, self._header_height, 0, 0)

    @override
    def _chrome_widgets(self) -> tuple[QWidget, ...]:
        return self._gutter, self._header

    @override
    def _columns_width(self) -> float:
        """The hex columns plus the document margins around them."""
        return self.text_width() + 2 * self.document().documentMargin()

    @override
    def _chrome_width(self) -> int:
        return (
            super()._chrome_width()
            + self._gutter_width
            + self.verticalScrollBar().sizeHint().width()
        )

    def _measure_gutter(self) -> int:
        digits = max(
            (len(row.offset_text) for row in self._rows if row.offset is not None),
            default=_OFFSET_DIGITS,
        )
        advance = QFontMetricsF(self.font()).horizontalAdvance("0")
        return ceil(advance * max(digits, _OFFSET_DIGITS)) + _GUTTER_PAD

    def _on_update_request(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())


class _TextPane(_Pane[_PaneHeader]):
    """The decoded text column, in its own text so a selection copies only it."""

    header_clicked = Signal()
    _bottom_inset = 0

    def __init__(self, panel: QWidget) -> None:
        super().__init__(panel, _PaneHeader)
        self.setObjectName("textPane")
        # The hex pane carries the scrollbars; this one follows it.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._header.clicked.connect(self.header_clicked.emit)

    def set_bottom_inset(self, inset: int) -> None:
        """Match the hex pane's horizontal scrollbar, so both viewports stay level."""
        if inset != self._bottom_inset:
            self._bottom_inset = inset
            self._set_insets()

    def set_rows(self, rows: list[ExtractRow], encoding: ExtractEncoding) -> None:
        self.setPlainText("\n".join(decode_rows(rows, encoding)))
        self.set_encoding_label(encoding)
        self.sync_chrome()

    def set_encoding_label(self, encoding: ExtractEncoding) -> None:
        self._header.set_label(text.EXTRACT_ENCODING_LABELS[encoding])

    def header_position(self) -> QPoint:
        """Where the encoding list should drop from."""
        return self._header.mapToGlobal(QPoint(0, self._header.height()))

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

    @override
    def _set_insets(self) -> None:
        self.setViewportMargins(0, self._header_height, 0, self._bottom_inset)

    @override
    def _columns_width(self) -> float:
        """The text column plus the document margins around it."""
        return self.column_width() + 2 * self.document().documentMargin()


class ExtractView(QWidget):
    """Read-only byte dump: what the stream is, one jump row, hex, then text."""

    encoding_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._encoding = ExtractEncoding.ASCII
        self._rows: tuple[ExtractRow, ...] = ()
        self._detections: tuple[Detection, ...] = ()
        self._marks: tuple[Detection, ...] = ()
        self._findings_key: tuple[str, tuple[Detection, ...]] | None = None
        self._chips: tuple[QWidget, ...] = ()
        self.hex_pane = _HexPane(self)
        self.text_pane = _TextPane(self)
        for pane in (self.hex_pane, self.text_pane):
            # Set per pane: a font on the container does not survive the
            # stylesheet polish, and a proportional font breaks the columns.
            pane.setFont(dump_font())
        self._type_label = QLabel()
        self._type_label.setObjectName("note")
        self._notice = QWidget()
        self._notice_layout = QHBoxLayout(self._notice)
        self._notice_layout.setContentsMargins(0, 0, 0, 4)
        self._notice_layout.setSpacing(2)
        self._notice_layout.addWidget(self._type_label)
        self._notice_layout.addStretch(1)
        self._notice.setVisible(False)
        panes = QHBoxLayout()
        panes.setContentsMargins(0, 0, 0, 0)
        panes.setSpacing(_PANE_GAP)
        # Every column holds its own width; panel width beyond them stays empty.
        panes.addWidget(self.hex_pane, 0)
        panes.addWidget(self.text_pane, 0)
        panes.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._notice)
        layout.addLayout(panes)
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
        self._rows = tuple(rows)
        self.hex_pane.set_rows(rows)
        self.text_pane.set_rows(rows, encoding)
        self.sync_encoding(encoding)
        self._apply_highlights()

    def set_findings(self, type_note: str, detections: tuple[Detection, ...]) -> None:
        """The strip above the dump: the stream's type, then one chip per find.

        A search filters the rows underneath, so the strip only rebuilds when
        the findings themselves moved.
        """
        key = (type_note, detections)
        if key == self._findings_key:
            return
        self._findings_key = key
        self._detections = detections
        self._type_label.setText(type_note)
        self._notice.setVisible(bool(type_note) or bool(detections))
        layout = self._notice_layout
        for chip in self._chips:
            layout.removeWidget(chip)
            chip.deleteLater()
        self._chips = (*map(self._chip, detections[:_MAX_CHIPS]), *self._overflow(len(detections)))
        for chip in self._chips:
            layout.insertWidget(layout.count() - 1, chip)
        self._apply_highlights()

    def _overflow(self, count: int) -> tuple[QWidget, ...]:
        """The muted "+N" when the finds outrun the chip row."""
        if count <= _MAX_CHIPS:
            return ()
        label = QLabel(text.more_detections(count - _MAX_CHIPS))
        label.setObjectName("note")
        return (label,)

    def _chip(self, detection: Detection) -> QPushButton:
        chip = QPushButton(detection.chip)
        chip.setObjectName("linkButton")
        chip.setToolTip(text.DETECTION_TIP)
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.clicked.connect(lambda _checked=False: self._reveal(detection))
        return chip

    def _reveal(self, detection: Detection) -> None:
        spans = _row_spans(self._rows, (detection,))
        if spans:
            self.hex_pane.reveal(*spans[0][:3])

    def set_marks(self, marks: tuple[Detection, ...]) -> None:
        """Static highlight spans of the dump itself, kept beside the finds."""
        self._marks = marks
        self._apply_highlights()

    def _apply_highlights(self) -> None:
        """Re-tint the dump: the document resets whenever the rows change."""
        self.hex_pane.set_highlights(_row_spans(self._rows, (*self._marks, *self._detections)))

    def sync_encoding(self, encoding: ExtractEncoding) -> None:
        """Follow the state's encoding without asking for it back."""
        self._encoding = encoding
        self.text_pane.set_encoding_label(encoding)

    def content_width(self) -> int:
        """Pixels both panes need side by side, their gap included.

        The panes hold fixed widths once rows arrive, so this is the width a
        panel must offer to show the dump without scrolling it sideways —
        askable before any rows exist, since it follows the fonts alone.
        """
        return ceil(self.hex_pane.width_wanted()) + ceil(self.text_pane.width_wanted()) + _PANE_GAP

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


def _row_spans(
    rows: tuple[ExtractRow, ...],
    detections: tuple[Detection, ...],
) -> list[tuple[int, int, int, bool]]:
    """Where each detection sits in the rows on screen, filtered rows included.

    ``(row, byte column, length, flagged)`` against the rows as displayed — a
    find inside a row the search filtered out simply has no span. Rows keep
    their stream order, so the row a find lands in is one binary search away,
    however many rows the dump holds.
    """
    placed = [
        (index, offset) for index, row in enumerate(rows) if (offset := row.offset) is not None
    ]
    offsets = [offset for _index, offset in placed]
    spans = []
    for detection in detections:
        position = bisect_right(offsets, detection.offset) - 1
        if position < 0:
            continue
        index, offset = placed[position]
        column = detection.offset - offset
        if column < BYTES_PER_ROW:
            spans.append(
                (index, column, min(detection.length, BYTES_PER_ROW - column), detection.flagged)
            )
    return spans


def highlight_format(flagged: bool) -> QTextCharFormat:
    color = QColor(theme.WARNING if flagged else theme.ACCENT)
    color.setAlpha(_HIGHLIGHT_ALPHA)
    fmt = QTextCharFormat()
    fmt.setBackground(color)
    return fmt


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


def _bar_extent(pane: QPlainTextEdit) -> int:
    """The height a scrollbar takes, reserved so both panes keep one viewport height."""
    return pane.horizontalScrollBar().sizeHint().height()

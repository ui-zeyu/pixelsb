"""The file-info page of the sidebar: the file, the container census, the EXIF."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import override

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.classify import StreamClassifier
from pixelsb.domain.container import Block, BlockRole, ContainerReport, render_blocks
from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.extract import BYTES_PER_ROW, filter_extract, format_extract
from pixelsb.domain.models import ExtractEncoding, LoadedImage
from pixelsb.io.classify import stream_classifier
from pixelsb.io.inspect import exif_entries, inspect_container
from pixelsb.io.loading import image_from_pixels
from pixelsb.ui import text, theme
from pixelsb.ui.controls import drain, hairline, section_title
from pixelsb.ui.extract_view import ExtractView, dump_font

_MAX_BLOCKS = 64  # the census list stops here; huge files still scan in full
_CHUNK_HEADER = 8  # length plus type: the bytes in front of a chunk's payload
_DUMP_MARGINS = 88  # the rail, the page's margins, and a little slack around the dump
_DUMP_MAX_BYTES = 512  # the dump frame grows for this much data, then scrolls
_DUMP_MIN_HEIGHT = 116
_DUMP_MAX_HEIGHT = 236
_DUMP_CHROME = 56  # the dump frame's caption strip, byte ruler, and margins
_DUMP_ROW_HEIGHT = 17


class InfoPanel(QWidget):
    """One sidebar page reporting on the file behind the open image.

    Three titled cards: 文件信息, 检查 (the census — anomalies above the block
    list: a radio picks the block whose full chunk dumps below in the extract
    view's style, clicking a block name renders its whole stream as pixels),
    and EXIF. The census reads the whole file, so it runs once per file and
    waits for the page; rendering a block swaps the canvas image but not the
    file being described, so the list, the selection, and the marks all stay.
    """

    render_requested = Signal(object)  # a LoadedImage rendered from blocks

    def __init__(self) -> None:
        super().__init__()
        self._image: LoadedImage | None = None
        self._scanned_path: Path | None = None
        self._report: ContainerReport | None = None
        self._file_data = b""
        self._selected: int | None = None
        self._radios: dict[int, QRadioButton] = {}
        self._name_buttons: dict[int, QPushButton] = {}
        self._canvas_row: int | None = None
        self._dump_encoding = ExtractEncoding.ASCII
        self._classify: StreamClassifier | None = None
        self._empty = _note(text.NO_IMAGE)
        self._name = _note(name="infoName")
        self._path = _note()
        self._facts = _note(name="infoLine")
        self._planes = _note()
        self._warn = _note(name="warnNote")
        self._hits = _note()
        self._blocks_layout = QGridLayout()
        self._blocks_layout.setHorizontalSpacing(12)
        self._blocks_layout.setVerticalSpacing(3)
        blocks_host = QWidget()
        blocks_host.setLayout(self._blocks_layout)
        self._canvas_note = _note(name="infoLine")
        self._dump_search = QLineEdit()
        self._dump_search.setPlaceholderText(text.EXTRACT_SEARCH_TIP)
        self._dump_search.setFixedHeight(theme.CONTROL_HEIGHT)
        self._dump_search.textChanged.connect(lambda _text: self._show_dump())
        # The dump is the extract panel's own view, unscaled; the frame around
        # it takes the overflow so the block list above stays the page's width.
        self._dump_view = ExtractView()
        self._dump = QScrollArea()
        self._dump.setObjectName("dumpArea")
        self._dump.setWidgetResizable(True)
        self._dump.setFrameShape(QFrame.Shape.NoFrame)
        self._dump.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._dump.setWidget(self._dump_view)
        self._dump.setMinimumHeight(_DUMP_MIN_HEIGHT)
        self._dump.setFixedHeight(_DUMP_MIN_HEIGHT)
        self._dump_view.encoding_changed.connect(self._on_dump_encoding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addWidget(self._empty)
        self._form = QWidget()
        rows = QVBoxLayout(self._form)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(12)
        rows.addWidget(
            _card(
                section_title(text.INFO_TITLE),
                self._name,
                self._path,
                hairline(),
                self._facts,
                self._planes,
            )
        )
        rows.addWidget(
            _card(
                section_title(text.SECTION_SCAN),
                self._warn,
                self._hits,
                blocks_host,
                self._canvas_note,
            )
        )
        # The dump sits outside the card: it is the extract view itself, and the
        # card frame would spend width the panel's own dump does not.
        rows.addWidget(self._dump_search)
        rows.addWidget(self._dump)
        self._exif_layout = QVBoxLayout()
        rows.addWidget(_card(section_title(text.SECTION_EXIF), _exif_host(self._exif_layout)))
        rows.addStretch(1)
        layout.addWidget(self._form, 1)
        self._form.setVisible(False)

    def set_image(self, image: LoadedImage | None) -> None:
        """Point the page at ``image``; the scan happens when the page shows."""
        if image is self._image:
            return
        self._image = image
        self._render()

    @override
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._render()

    def _render(self) -> None:
        """Fill the page for the current image; a rendered canvas changes no scan."""
        if not self.isVisible():
            return
        image = self._image
        self._empty.setVisible(image is None)
        self._form.setVisible(image is not None)
        if image is None:
            self._scanned_path = None
            return
        if image.path == self._scanned_path:
            return  # a block render replaces the canvas image, not the file described here
        self._scanned_path = image.path
        self._name.setText(text.breakable(image.path.name))
        self._path.setText(text.breakable(str(image.path)))
        self._facts.setText(text.file_facts(image))
        self._planes.setText(text.planes_text(image))
        try:
            self._file_data = image.path.read_bytes()
        except OSError:
            self._file_data = b""
        self._render_census(inspect_container(image.path))
        self._show_exif(exif_entries(image.path))

    def _render_census(self, report: ContainerReport) -> None:
        """A clean file says nothing; the blocks and their dump speak for it."""
        lines = text.finding_lines(report)
        self._warn.setVisible(bool(lines))
        self._warn.setText("\n".join(f"{text.WARNING_MARK} {line}" for line in lines))
        hits = text.payload_lines(report.findings)
        self._hits.setVisible(bool(hits))
        self._hits.setText("\n".join(hits))
        self._report = report
        self._selected = 0 if report.blocks else None
        # The canvas already shows the file's first stream: its head row carries
        # the radio and the accent, without re-rendering anything.
        self._canvas_row = next(
            (index for index, block in enumerate(report.blocks) if block.group == 1), None
        )
        first = report.blocks[self._canvas_row] if self._canvas_row is not None else None
        self._canvas_note.setText(
            text.canvas_note(first, len(self._stream_payloads(first)))
            if first is not None
            else text.original_note()
        )
        self._show_blocks()
        self._show_dump()

    def preferred_width(self) -> int:
        """Panel width that shows the block dump without scrolling it sideways."""
        return self._dump_view.content_width() + _DUMP_MARGINS

    def _show_blocks(self) -> None:
        drain(self._blocks_layout)
        self._name_buttons = {}
        self._radios = {}
        report = self._report
        if report is None:
            return
        shown = report.blocks[:_MAX_BLOCKS]
        radios = _radio_rows(shown)
        for row, block in enumerate(shown):
            if row in radios:
                radio = QRadioButton()
                radio.setToolTip(text.BLOCK_RENDER_TIP)
                # The checked radio is the stream on the canvas. The file's own
                # stream starts there, and saying so must not render anything.
                radio.blockSignals(True)
                radio.setChecked(row == self._canvas_row)
                radio.blockSignals(False)
                radio.toggled.connect(
                    lambda checked, row=row: self._on_block_rendered(row, checked)
                )
                self._blocks_layout.addWidget(radio, row, 0)
                self._radios[row] = radio
            self._blocks_layout.addWidget(_note(f"0x{block.offset:08x}"), row, 1)
            name = QPushButton(block.label)
            name.setObjectName("blockName")
            name.setToolTip(text.DUMP_TIP)
            name.setCursor(Qt.CursorShape.PointingHandCursor)
            name.clicked.connect(lambda _checked=False, row=row: self._on_block_dumped(row))
            self._blocks_layout.addWidget(name, row, 2)
            self._name_buttons[row] = name
            size = _note(f"{block.length} B")
            size.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._blocks_layout.addWidget(size, row, 3)
            self._blocks_layout.addWidget(
                _note(
                    text.block_role_text(block),
                    name="warnNote" if block.role is BlockRole.ANCILLARY else "note",
                ),
                row,
                4,
            )
            self._blocks_layout.addWidget(_preview(block), row, 5)
        self._name_width()
        # The payload taste fills the right end, so a row reads across the panel.
        self._blocks_layout.setColumnStretch(5, 1)
        hidden = len(report.blocks) - _MAX_BLOCKS
        if hidden > 0:
            self._blocks_layout.addWidget(_note(text.more_blocks(hidden)), len(shown), 0, 1, 6)
        self._mark_rendered()

    def _name_width(self) -> None:
        """One width for every block name, so the column of buttons lines up."""
        buttons = list(self._name_buttons.values())
        if not buttons:
            return
        widest = max(button.sizeHint().width() for button in buttons)
        for button in buttons:
            button.setFixedWidth(widest)

    def _on_block_dumped(self, row: int) -> None:
        """The block name: the chunk whose bytes the dump below shows."""
        self._selected = row
        self._show_dump()
        self._mark_rendered()

    def _on_block_rendered(self, row: int, checked: bool) -> None:
        """The radio: render the chosen block's whole stream on the canvas."""
        if not checked:
            return
        image = self._image
        report = self._report
        if image is None or report is None or report.header is None:
            self._revert_canvas_row()  # nothing to render, so the radio must not claim it
            return
        block = report.blocks[row]
        payloads = self._stream_payloads(block)
        try:
            pixels = render_blocks(payloads, report.header, report.palette)
        except (ValueError, OverflowError) as exc:
            _restyle(self._canvas_note, "warnNote")
            self._canvas_note.setText(text.render_failed(str(exc)))
            self._revert_canvas_row()
            return
        _restyle(self._canvas_note, "infoLine")
        self._canvas_note.setText(text.canvas_note(block, len(payloads)))
        self._canvas_row = row
        self._mark_rendered()
        self.render_requested.emit(image_from_pixels(image.path, pixels))

    def _revert_canvas_row(self) -> None:
        """Put the radios back where the canvas really is, after a failed render."""
        for row, radio in self._radios.items():
            radio.blockSignals(True)
            radio.setChecked(row == self._canvas_row)
            radio.blockSignals(False)

    def _mark_rendered(self) -> None:
        """Dress the rows: the stream on the canvas at its head, the dumped block tinted."""
        for row, button in self._name_buttons.items():
            shown = row == self._canvas_row
            dumped = row == self._selected
            if button.property("shown") == shown and button.property("dumped") == dumped:
                continue
            button.setProperty("shown", shown)
            button.setProperty("dumped", dumped)
            theme.repolish(button)

    def _stream_payloads(self, block: Block) -> list[bytes]:
        """The block's bytes, or those of every chunk in its stream."""
        report = self._report
        if report is None or not block.group:
            return [block.payload]
        return [member.payload for member in report.blocks if member.group == block.group]

    def _show_dump(self) -> None:
        report = self._report
        block = (
            report.blocks[self._selected]
            if report is not None and self._selected is not None
            else None
        )
        if block is None or not self._file_data:
            self._dump_view.set_rows([], self._dump_encoding)
            self._dump_view.set_findings("", ())
            self._dump.setFixedHeight(_DUMP_MIN_HEIGHT)
            return
        chunk = self._file_data[block.offset : block.offset + block.length]
        rows = filter_extract(format_extract(chunk, base=block.offset), self._dump_search.text())
        self._dump_view.set_rows(rows, self._dump_encoding)
        self._dump_view.set_marks(self._dump_marks(block))
        self._dump_view.set_findings(
            text.dump_caption(block, self._guessed_type(block.payload)),
            self._dump_findings(block),
        )
        # The frame is as tall as the chunk's rows need, so a short block does
        # not leave the panel half empty.
        grown = min(len(rows), _DUMP_MAX_BYTES // BYTES_PER_ROW + 1)
        wanted = _DUMP_CHROME + grown * _DUMP_ROW_HEIGHT
        self._dump.setFixedHeight(min(max(wanted, _DUMP_MIN_HEIGHT), _DUMP_MAX_HEIGHT))

    def _on_dump_encoding(self, value: str) -> None:
        self._dump_encoding = ExtractEncoding(value)
        self._show_dump()

    def _dump_marks(self, block: Block) -> tuple[Detection, ...]:
        """What the dump tints by itself: the chunk's own header and CRC."""
        marks = [Detection(block.offset, length=min(_CHUNK_HEADER, block.length))]
        if block.length > _CHUNK_HEADER:
            marks.append(Detection(block.offset + block.length - 4, length=4))
        return tuple(marks)

    def _dump_findings(self, block: Block) -> tuple[Detection, ...]:
        """The payload's detections, moved to their file offsets for the chips."""
        return tuple(
            replace(detection, offset=detection.offset + block.payload_at)
            for detection in detect_patterns(block.payload)
        )

    def _guessed_type(self, payload: bytes) -> str:
        if self._classify is None:
            self._classify = stream_classifier()
        classification = self._classify(payload)
        return classification.label if classification else ""

    def _show_exif(self, entries: tuple[tuple[str, str], ...]) -> None:
        drain(self._exif_layout)
        if not entries:
            self._exif_layout.addWidget(_note(text.INFO_NO_EXIF))
            return
        table = QGridLayout()
        table.setContentsMargins(0, 0, 0, 0)
        table.setHorizontalSpacing(12)
        table.setVerticalSpacing(4)
        for row, (tag, value) in enumerate(entries):
            table.addWidget(_note(tag), row, 0)
            table.addWidget(_note(text.breakable(value), name="infoLine"), row, 1)
        table.setColumnStretch(1, 1)
        self._exif_layout.addLayout(table)


def _exif_host(layout: QVBoxLayout) -> QWidget:
    """The EXIF card's inner widget: a host the table is drained into."""
    host = QWidget()
    host.setLayout(layout)
    return host


def _card(*widgets: QWidget) -> QFrame:
    """A soft gray card grouping one section, like the bit grid's card."""
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    return card


def _restyle(label: QLabel, name: str) -> None:
    """Swap a label's style object and make the stylesheet see the new name."""
    label.setObjectName(name)
    theme.repolish(label)


def _radio_rows(blocks: Sequence[Block]) -> set[int]:
    """The rows that carry a radio: one per stream, one per standalone block.

    A stream's chunks render as one picture, so offering a radio on each of
    them would read as three separate choices; the stream gets its radio at its
    head instead, and its other chunks keep the 流 N label that ties them to it.
    """
    heads: set[int] = set()
    streams: set[int] = set()
    for index, block in enumerate(blocks):
        if not block.group:
            heads.add(index)
        elif block.group not in streams:
            streams.add(block.group)
            heads.add(index)
    return heads


def _preview(block: Block) -> QLabel:
    """The block's payload as printable text, at the right end of its row."""
    label = _note(block.preview)
    label.setFont(dump_font())
    # Whatever the panel width leaves is enough: a long taste clips at the right
    # edge rather than wrapping, which would make its row taller than the rest.
    label.setWordWrap(False)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label


def _note(content: str = "", name: str = "note") -> QLabel:
    label = QLabel(content)
    label.setObjectName(name)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label

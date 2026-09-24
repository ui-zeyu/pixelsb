"""The extract panel: a painted gutter and header, selectable data only."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtGui import QImage, QTextCursor
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from pixelsb.domain.extract import ASCII_START, BYTES_PER_ROW
from pixelsb.domain.models import ViewerState
from pixelsb.domain.transitions import open_image
from pixelsb.io.loading import load_image
from pixelsb.ui import text
from pixelsb.ui.inspector import Inspector
from pixelsb.ui.main_window import MainWindow


def _inspector(qtbot: QtBot, path: Path) -> Inspector:
    """A shown panel holding one image, so the extract view has a real viewport."""
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.resize(360, 700)
    inspector.show()
    inspector.set_state(open_image(ViewerState(), load_image(path), zoom=4.0))
    qtbot.wait(20)  # let the layout apply the grids before geometry is read
    return inspector


def _dark_columns(image: QImage, x0: int, x1: int, y0: int, y1: int) -> list[int]:
    return [
        x for x in range(x0, x1) if any(image.pixelColor(x, y).value() < 170 for y in range(y0, y1))
    ]


def test_offsets_stay_out_of_the_extract_text(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    document = view.document()
    assert document.blockCount() == 6
    assert view.toPlainText().startswith("41 42 43")
    assert "00000000" not in view.toPlainText()
    assert view.offset_text(0) == "00000000"
    assert view.offset_text(1) == "00000010"
    assert view.offset_text(document.blockCount()) == ""


def test_selecting_everything_copies_data_without_offsets(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    cursor = view.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    selected = cursor.selectedText()
    assert "00000000" not in selected
    assert "41 42 43" in selected


def test_the_extract_note_does_not_repeat_the_section_title() -> None:
    assert text.SECTION_EXTRACT not in text.EXTRACT_NOTE
    assert text.EXTRACT_NOTE.startswith("按通道顺序")


def test_the_panel_starts_wide_enough_for_a_full_dump_row(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(extract_png)
    view = window.inspector._extract_view
    assert window._splitter.sizes()[1] >= window.inspector.preferred_width()
    assert view.viewport().geometry().width() >= view.text_width()
    assert view.horizontalScrollBar().maximum() == 0


def test_the_panel_leaves_the_canvas_a_minimum_width(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(760, 600)
    window.show()
    canvas, panel = window._splitter.sizes()
    assert panel >= window.inspector.minimumWidth()
    assert canvas >= 240


def test_the_header_rules_the_byte_columns(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    ruler = " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))
    assert view.header_text() == f"{ruler}  {text.EXTRACT_ASCII}"
    assert len(ruler) == ASCII_START - 2


def test_the_header_lines_up_with_the_hex_digits(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    image = view.grab().toImage()
    viewport = view.viewport().geometry()
    header_start = view._header.x() + view.gutter_width + 1
    header = _dark_columns(image, header_start, image.width(), 0, viewport.top())
    data = _dark_columns(image, viewport.left(), image.width(), viewport.top(), image.height())
    assert header
    assert data
    assert abs(header[0] - data[0]) <= 2


def test_the_gutter_paints_the_offsets(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    image = view.grab().toImage()
    viewport = view.viewport().geometry()
    gutter = _dark_columns(image, 0, viewport.left(), viewport.top(), image.height())
    assert gutter
    assert max(gutter) < viewport.left()
    assert view._gutter.geometry().top() == viewport.top()


def test_the_bit_grids_fill_the_panel_width(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    card = inspector._canvas_card
    matrix = inspector._canvas_matrix
    assert card.objectName() == "card"
    assert card.x() == 16
    assert card.width() >= inspector.width() - 40  # the card spans the panel
    assert matrix.width() >= card.width() - 24  # and the grid fills the card
    rightmost = max(box.geometry().right() for box in matrix._boxes.values())
    assert rightmost >= matrix.width() * 0.9  # the bit columns spread to the edges


def test_the_presets_share_the_section_header(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    buttons = [button.text() for button in inspector.findChildren(QPushButton)]
    assert buttons == [text.ORIGINAL, text.ALL_LSB]


def test_note_rows_have_no_offset(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    inspector._extract_search.setText("zzz")  # matches nothing, so only a note is left
    view = inspector._extract_view
    assert view.document().blockCount() == 1
    assert view.offset_text(0) == ""


def test_an_empty_stream_shows_a_note_without_an_offset(qtbot: QtBot, extract_png: Path) -> None:
    state = replace(
        open_image(ViewerState(), load_image(extract_png), zoom=4.0), selection=frozenset()
    )
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.set_state(state)
    view = inspector._extract_view
    assert view.toPlainText() == "（无数据）"
    assert view.offset_text(0) == ""


def test_a_panel_without_an_image_paints_no_chrome(qtbot: QtBot) -> None:
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.set_state(ViewerState())
    view = inspector._extract_view
    assert view.toPlainText() == ""
    assert view._gutter.size().isEmpty()
    assert view._header.size().isEmpty()

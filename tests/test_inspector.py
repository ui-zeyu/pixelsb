"""The extract panel: a painted gutter and header, selectable data only."""

from dataclasses import replace
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QFontInfo, QFontMetricsF, QImage, QTextCursor
from PySide6.QtWidgets import QPushButton, QWidget
from pytestqt.qtbot import QtBot

from pixelsb.domain.extract import ASCII_START, BYTES_PER_ROW, format_extract
from pixelsb.domain.models import BitChoice, BitOrder, ExtractEncoding, ScanOrder, ViewerState
from pixelsb.domain.transitions import (
    open_image,
    set_bit_order,
    set_channel_order,
    set_scan_order,
)
from pixelsb.io.loading import load_image
from pixelsb.ui import text, theme
from pixelsb.ui.inspector import _STEPPER_GAP, _STEPPER_WIDTH, Inspector
from pixelsb.ui.main_window import MainWindow

_MARGIN = 4  # the document margin Qt puts around the text inside a pane


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
    pane = _inspector(qtbot, extract_png)._extract_view.hex_pane
    document = pane.document()
    assert document.blockCount() == 6
    assert pane.toPlainText().startswith("41 42 43")
    assert "00000000" not in pane.toPlainText()
    assert pane.offset_text(0) == "00000000"
    assert pane.offset_text(1) == "00000010"
    assert pane.offset_text(document.blockCount()) == ""


def test_the_panes_hold_one_column_each(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    assert view.hex_pane.toPlainText().startswith("41 42 43")
    assert view.text_pane.toPlainText().startswith("ABC")
    assert view.text_pane.document().blockCount() == view.hex_pane.document().blockCount()
    assert "ABC" not in view.hex_pane.toPlainText()
    assert "41" not in view.text_pane.toPlainText()


def test_a_selection_carries_only_its_own_column(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    text_cursor = view.text_pane.textCursor()
    text_cursor.select(QTextCursor.SelectionType.Document)
    selected = text_cursor.selectedText().replace("\u2029", "\n")  # block separator
    assert selected == view.text_pane.toPlainText()
    assert "41 42 43" not in selected
    hex_cursor = view.hex_pane.textCursor()
    hex_cursor.select(QTextCursor.SelectionType.Document)
    assert "41 42 43" in hex_cursor.selectedText()
    assert "00000000" not in hex_cursor.selectedText()
    assert "ABC" not in hex_cursor.selectedText()


def test_the_dump_is_monospaced_and_snug(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    for pane in (view.hex_pane, view.text_pane):
        assert QFontInfo(pane.font()).fixedPitch()
        metrics = QFontMetricsF(pane.font())
        # One advance per glyph, or the columns cannot line up.
        assert metrics.horizontalAdvance("W") == metrics.horizontalAdvance("0")
    # Each pane hugs its columns: wide enough for them, no wider than the
    # document margins plus the scrollbar room each pane reserves.
    slack = 2 * _MARGIN + 16
    hex_pane = view.hex_pane
    assert hex_pane.viewport().width() >= hex_pane.text_width()
    assert hex_pane.viewport().width() <= hex_pane.text_width() + slack
    assert hex_pane.horizontalScrollBar().maximum() == 0
    text_pane = view.text_pane
    assert text_pane.viewport().width() >= text_pane.column_width()
    assert text_pane.viewport().width() <= text_pane.column_width() + slack


def test_the_panes_scroll_together(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    rows = format_extract(bytes(range(256)) * 2)  # long enough to scroll
    view.set_rows(rows, ExtractEncoding.ASCII)
    hex_bar = view.hex_pane.verticalScrollBar()
    text_bar = view.text_pane.verticalScrollBar()
    # Equal viewports are what keeps the rows of the two panes level.
    assert view.hex_pane.viewport().height() == view.text_pane.viewport().height()
    assert hex_bar.maximum() > 0
    hex_bar.setValue(hex_bar.maximum() // 2)
    assert text_bar.value() == hex_bar.value()
    text_bar.setValue(3)
    assert hex_bar.value() == 3


def test_every_order_control_explains_what_it_does(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    assert inspector._channel_combo.toolTip() == text.ORDER_CHANNEL_TIP
    tips = {button.text(): button.toolTip() for button in inspector.findChildren(QPushButton)}
    assert tips["MSB"] == text.ORDER_BIT_MSB_TIP
    assert tips["LSB"] == text.ORDER_BIT_LSB_TIP
    assert tips["XY"] == text.ORDER_SCAN_XY_TIP
    assert tips["YZ"] == text.ORDER_SCAN_YZ_TIP


def test_the_panel_starts_wide_enough_for_a_full_dump_row(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(extract_png)
    pane = window.inspector._extract_view.hex_pane
    assert window._splitter.sizes()[1] >= window.inspector.preferred_width()
    assert pane.viewport().geometry().width() >= pane.text_width()
    assert pane.horizontalScrollBar().maximum() == 0
    text_pane = window.inspector._extract_view.text_pane
    assert text_pane.viewport().geometry().width() >= text_pane.column_width()


def test_the_panel_leaves_the_canvas_a_minimum_width(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(760, 600)
    window.show()
    canvas, panel = window._splitter.sizes()
    assert panel >= window.inspector.minimumWidth()
    assert canvas >= 240


def test_the_headers_rule_their_own_columns(qtbot: QtBot, extract_png: Path) -> None:
    view = _inspector(qtbot, extract_png)._extract_view
    ruler = " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))
    assert view.hex_pane.header_text() == ruler
    assert len(ruler) == ASCII_START - 2
    # The two headers are one height, so the rows of both panes start level.
    assert view.text_pane._header.height() == view.hex_pane._header.height()
    hex_top = view.hex_pane.mapTo(view, view.hex_pane.viewport().geometry().topLeft()).y()
    text_top = view.text_pane.mapTo(view, view.text_pane.viewport().geometry().topLeft()).y()
    assert hex_top == text_top


def test_the_header_offers_the_encodings(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    view = inspector._extract_view
    header = view.text_pane._header
    assert header.label() == "ASCII"  # ASCII by default
    actions = view._encoding_menu().actions()
    assert [action.text() for action in actions] == [
        label for _encoding, label in text.EXTRACT_ENCODINGS
    ]
    assert [action.isChecked() for action in actions] == [True, False, False, False]
    requested: list[str] = []
    inspector.encoding_requested.connect(requested.append)
    actions[2].trigger()  # UTF-16LE
    assert requested == [ExtractEncoding.UTF16_LE.value]
    view.set_rows([], ExtractEncoding.UTF16_LE)
    assert header.label() == "UTF-16LE"
    assert view._encoding_menu().actions()[2].isChecked()


def test_the_header_lines_up_with_the_hex_digits(qtbot: QtBot, extract_png: Path) -> None:
    pane = _inspector(qtbot, extract_png)._extract_view.hex_pane
    image = pane.grab().toImage()
    viewport = pane.viewport().geometry()
    header_start = pane._header.x() + pane.gutter_width + 1
    header = _dark_columns(image, header_start, image.width(), 0, viewport.top())
    data = _dark_columns(image, viewport.left(), image.width(), viewport.top(), image.height())
    assert header
    assert data
    assert abs(header[0] - data[0]) <= 2


def test_the_gutter_paints_the_offsets(qtbot: QtBot, extract_png: Path) -> None:
    pane = _inspector(qtbot, extract_png)._extract_view.hex_pane
    image = pane.grab().toImage()
    viewport = pane.viewport().geometry()
    gutter = _dark_columns(image, 0, viewport.left(), viewport.top(), image.height())
    assert gutter
    assert max(gutter) < viewport.left()
    assert pane._gutter.geometry().top() == viewport.top()


def test_the_bit_grids_fill_the_panel_width(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    card = inspector._canvas_card
    matrix = inspector._canvas_matrix
    assert card.objectName() == "card"
    layout = inspector.layout()
    assert layout is not None
    margins = layout.contentsMargins()
    column = _STEPPER_WIDTH + _STEPPER_GAP
    assert card.x() == margins.left() + column  # the channel column sits to its left
    inset = margins.left() + margins.right() + column
    assert card.width() >= inspector.width() - inset - 4  # the card keeps the rest
    assert matrix.width() >= card.width() - 24  # and the grid fills the card
    rightmost = max(box.geometry().right() for box in matrix._boxes.values())
    assert rightmost >= matrix.width() * 0.9  # the bit columns spread to the edges


def _top_left(widget: QWidget, parent: QWidget) -> QPoint:
    return widget.mapTo(parent, QPoint(0, 0))


def test_the_steppers_sit_around_the_grid(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    card = inspector._canvas_card
    card_rect = QRect(_top_left(card, inspector), card.size())
    prev_pos = _top_left(inspector._plane_prev, inspector)
    next_pos = _top_left(inspector._plane_next, inspector)
    # Left/right: under the grid, centred on it, with a gap in between.
    assert prev_pos.y() >= card_rect.bottom()
    assert next_pos.x() >= prev_pos.x() + inspector._plane_prev.width() + 6
    pair_center = (prev_pos.x() + next_pos.x() + inspector._plane_next.width()) / 2
    assert abs(pair_center - card_rect.center().x()) <= 3
    # Up/down: a column left of the grid, centred on its height.
    up_pos = _top_left(inspector._channel_prev, inspector)
    down_pos = _top_left(inspector._channel_next, inspector)
    assert up_pos.x() + inspector._channel_prev.width() <= card_rect.x()
    assert down_pos.y() >= up_pos.y() + inspector._channel_prev.height()
    column_center = (up_pos.y() + down_pos.y() + inspector._channel_next.height()) / 2
    assert abs(column_center - card_rect.center().y()) <= 3


def test_the_section_holds_the_steppers_the_presets_and_the_orders(
    qtbot: QtBot, extract_png: Path
) -> None:
    inspector = _inspector(qtbot, extract_png)
    labels = {button.text() for button in inspector.findChildren(QPushButton)}
    assert labels == {
        text.PLANE_PREV,
        text.PLANE_NEXT,
        text.CHANNEL_PREV,
        text.CHANNEL_NEXT,
        text.ORIGINAL,
        text.ALL_LSB,
        "MSB",
        "LSB",
        "XY",
        "YZ",
    }


def test_the_steppers_emit_their_steps(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    planes: list[int] = []
    channels: list[int] = []
    inspector.plane_step.connect(planes.append)
    inspector.channel_step.connect(channels.append)
    inspector._plane_next.click()
    inspector._plane_prev.click()
    inspector._channel_next.click()
    inspector._channel_prev.click()
    assert planes == [1, -1]
    assert channels == [1, -1]


def test_note_rows_have_no_offset(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    inspector._extract_search.setText("zzz")  # matches nothing, so only a note is left
    view = inspector._extract_view
    assert view.hex_pane.document().blockCount() == 1
    assert view.hex_pane.offset_text(0) == ""
    assert view.text_pane.toPlainText() == ""


def test_an_empty_stream_shows_a_note_without_an_offset(qtbot: QtBot, extract_png: Path) -> None:
    state = replace(
        open_image(ViewerState(), load_image(extract_png), zoom=4.0), selection=frozenset()
    )
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.set_state(state)
    view = inspector._extract_view
    assert view.hex_pane.toPlainText() == "（无数据）"
    assert view.text_pane.toPlainText() == ""
    assert view.hex_pane.offset_text(0) == ""


def test_a_panel_without_an_image_paints_no_chrome(qtbot: QtBot) -> None:
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.set_state(ViewerState())
    view = inspector._extract_view
    assert view.hex_pane.toPlainText() == ""
    assert view.text_pane.toPlainText() == ""
    assert view.hex_pane._gutter.size().isEmpty()
    assert view.hex_pane._header.size().isEmpty()


def _button(pair: QWidget, label: str) -> QPushButton:
    return next(button for button in pair.findChildren(QPushButton) if button.text() == label)


def test_the_channel_list_offers_every_arrangement_of_the_channels_in_use(
    qtbot: QtBot, extract_png: Path
) -> None:
    inspector = _inspector(qtbot, extract_png)
    combo = inspector._channel_combo
    assert combo.isEnabled()
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "RGB",
        "RBG",
        "GRB",
        "GBR",
        "BRG",
        "BGR",
    ]
    assert combo.currentText() == "RGB"


def test_the_channel_list_holds_only_the_channels_that_carry_bits(
    qtbot: QtBot, extract_png: Path
) -> None:
    inspector = Inspector()
    qtbot.addWidget(inspector)
    state = open_image(ViewerState(), load_image(extract_png), zoom=4.0)
    chosen = frozenset({BitChoice("R", 3), BitChoice("B", 0)})
    inspector.set_state(replace(state, selection=chosen))
    combo = inspector._channel_combo
    assert [combo.itemText(index) for index in range(combo.count())] == ["RB", "BR"]
    assert combo.currentText() == "RB"
    inspector.set_state(replace(state, selection=frozenset({BitChoice("G", 0)})))
    assert [combo.itemText(index) for index in range(combo.count())] == ["G"]
    assert not combo.isEnabled()


def test_the_order_controls_follow_the_state(qtbot: QtBot, extract_png: Path) -> None:
    inspector = Inspector()
    qtbot.addWidget(inspector)
    state = open_image(ViewerState(), load_image(extract_png), zoom=4.0)
    state = set_channel_order(state, ("B", "G", "R"))
    state = set_bit_order(state, BitOrder.LSB)
    state = set_scan_order(state, ScanOrder.YZ)
    inspector.set_state(state)
    assert inspector._channel_combo.currentText() == "BGR"
    assert _button(inspector._bit_order, "LSB").isChecked()
    assert not _button(inspector._bit_order, "MSB").isChecked()
    assert _button(inspector._scan, "YZ").isChecked()
    assert not _button(inspector._scan, "XY").isChecked()


def test_picking_an_order_reports_it(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    channels: list[object] = []
    bit_orders: list[str] = []
    scans: list[str] = []
    inspector.channel_order_requested.connect(channels.append)
    inspector.bit_order_requested.connect(bit_orders.append)
    inspector.scan_requested.connect(scans.append)
    inspector._channel_combo.setCurrentIndex(5)
    _button(inspector._bit_order, "LSB").click()
    _button(inspector._scan, "YZ").click()
    assert channels == [("B", "G", "R")]
    assert bit_orders == [BitOrder.LSB.value]
    assert scans == [ScanOrder.YZ.value]


def test_the_order_row_shares_one_band_across_the_panel(qtbot: QtBot, extract_png: Path) -> None:
    inspector = _inspector(qtbot, extract_png)
    controls = (inspector._channel_combo, inspector._bit_order, inspector._scan)
    assert {widget.height() for widget in controls} == {theme.CONTROL_HEIGHT}
    assert len({widget.geometry().center().y() for widget in controls}) == 1
    right = inspector.width() - inspector.contentsMargins().right()
    assert max(widget.geometry().right() for widget in controls) <= right
    assert inspector._bit_order.geometry().right() < inspector._scan.geometry().left()


def test_the_dump_follows_the_order(qtbot: QtBot, extract_png: Path) -> None:
    inspector = Inspector()
    qtbot.addWidget(inspector)
    state = open_image(ViewerState(), load_image(extract_png), zoom=4.0)
    inspector.set_state(state)
    pane = inspector._extract_view.hex_pane
    assert pane.toPlainText().startswith("41 42 43")
    inspector.set_state(set_channel_order(state, ("B", "G", "R")))
    assert pane.toPlainText().startswith("43 42 41")
    inspector.set_state(set_bit_order(state, BitOrder.LSB))
    assert pane.toPlainText().startswith("82 42 c2")


def test_the_dump_follows_the_scan_order(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "column.png"
    image = Image.new("RGB", (3, 2))
    for row in range(2):
        image.putpixel((0, row), (1, 0, 0))  # the lowest red bit, first column only
    image.save(path)
    inspector = Inspector()
    qtbot.addWidget(inspector)
    state = open_image(ViewerState(), load_image(path), zoom=4.0)
    state = replace(state, selection=frozenset({BitChoice("R", 0)}))
    inspector.set_state(state)
    pane = inspector._extract_view.hex_pane
    assert pane.toPlainText().startswith("90")
    inspector.set_state(set_scan_order(state, ScanOrder.YZ))
    assert pane.toPlainText().startswith("c0")


def test_a_kept_channel_order_the_list_cannot_offer_leads_the_list(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 2), (1, 2, 3, 4)).save(path)
    inspector = Inspector()
    qtbot.addWidget(inspector)
    state = open_image(ViewerState(), load_image(path), zoom=4.0)
    # Four channels carrying bits leave the list offering only the order and its
    # reverse, so the order kept from a three-channel image shows first of all.
    inspector.set_state(set_channel_order(state, ("B", "G", "R")))
    combo = inspector._channel_combo
    assert [combo.itemText(index) for index in range(combo.count())] == ["BGRA", "RGBA", "ABGR"]
    assert combo.currentText() == "BGRA"
    requested: list[object] = []
    inspector.channel_order_requested.connect(requested.append)
    combo.setCurrentIndex(1)
    assert requested == [("R", "G", "B", "A")]

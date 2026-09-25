"""The extract panel: a painted gutter and header, selectable data only."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtGui import QFontInfo, QFontMetricsF, QImage, QTextCursor
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from pixelsb.domain.extract import ASCII_START, BYTES_PER_ROW, format_extract
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    BitsMask,
    ExtractEncoding,
    Layer,
    Mask,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.transitions import set_bit_order, set_channel_order, set_scan_order
from pixelsb.io.loading import load_image
from pixelsb.ui import text, theme
from pixelsb.ui.extract_panel import ExtractPanel
from pixelsb.ui.main_window import MainWindow
from tests.support import board_of, button, make_image
from tests.support import layers as layers_of


def _state(path: Path, *masks: Mask) -> ViewerState:
    return ViewerState(image=load_image(path), layers=layers_of(*masks), zoom=4.0)


def _with_bits(state: ViewerState, chosen: set[BitChoice]) -> ViewerState:
    """The same state with that bit selection added as a mask on top."""
    return replace(state, layers=(*state.layers, Layer(BitsMask(frozenset(chosen)))))


def _panel(qtbot: QtBot, path: Path) -> ExtractPanel:
    """A shown panel holding one image, so the extract view has a real viewport."""
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    panel.resize(360, 700)
    panel.show()
    state = _state(path)
    panel.set_state(state, board_of(state))
    qtbot.wait(20)  # let the layout apply the grids before geometry is read
    return panel


def _show(panel: ExtractPanel, state: ViewerState, bits_layer: int | None = None) -> None:
    panel.set_state(state, board_of(state), bits_layer=bits_layer)


def _dark_columns(image: QImage, x0: int, x1: int, y0: int, y1: int) -> list[int]:
    return [
        x for x in range(x0, x1) if any(image.pixelColor(x, y).value() < 170 for y in range(y0, y1))
    ]


def test_offsets_stay_out_of_the_extract_text(qtbot: QtBot, extract_png: Path) -> None:
    pane = _panel(qtbot, extract_png)._extract_view.hex_pane
    document = pane.document()
    assert document.blockCount() == 6
    assert pane.toPlainText().startswith("41 42 43")
    assert "00000000" not in pane.toPlainText()
    assert pane.offset_text(0) == "00000000"
    assert pane.offset_text(1) == "00000010"
    assert pane.offset_text(document.blockCount()) == ""


def test_the_panes_hold_one_column_each(qtbot: QtBot, extract_png: Path) -> None:
    view = _panel(qtbot, extract_png)._extract_view
    assert view.hex_pane.toPlainText().startswith("41 42 43")
    assert view.text_pane.toPlainText().startswith("ABC")
    assert view.text_pane.document().blockCount() == view.hex_pane.document().blockCount()
    assert "ABC" not in view.hex_pane.toPlainText()
    assert "41" not in view.text_pane.toPlainText()


def test_a_selection_carries_only_its_own_column(qtbot: QtBot, extract_png: Path) -> None:
    view = _panel(qtbot, extract_png)._extract_view
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
    view = _panel(qtbot, extract_png)._extract_view
    for pane in (view.hex_pane, view.text_pane):
        assert QFontInfo(pane.font()).fixedPitch()
        metrics = QFontMetricsF(pane.font())
        # One advance per glyph, or the columns cannot line up.
        assert metrics.horizontalAdvance("W") == metrics.horizontalAdvance("0")


def test_the_panes_scroll_together(qtbot: QtBot, extract_png: Path) -> None:
    view = _panel(qtbot, extract_png)._extract_view
    rows = format_extract(bytes(range(256)) * 8)  # long enough to scroll
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
    panel = _panel(qtbot, extract_png)
    assert panel._channel_combo.toolTip() == text.ORDER_CHANNEL_TIP
    tips = {widget.text(): widget.toolTip() for widget in panel.findChildren(QPushButton)}
    assert tips["MSB"] == text.ORDER_BIT_MSB_TIP
    assert tips["LSB"] == text.ORDER_BIT_LSB_TIP
    assert tips["XY"] == text.ORDER_SCAN_XY_TIP
    assert tips["YZ"] == text.ORDER_SCAN_YZ_TIP
    assert tips[text.EXTRACT_SAVE] == text.EXTRACT_SAVE_TIP


def test_the_panel_starts_wide_enough_for_a_full_dump_row(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(extract_png)
    pane = window.extract_panel._extract_view.hex_pane
    assert window._splitter.sizes()[2] >= window.extract_panel.preferred_width()
    assert pane.viewport().geometry().width() >= pane.text_width()
    assert pane.horizontalScrollBar().maximum() == 0
    text_pane = window.extract_panel._extract_view.text_pane
    assert text_pane.viewport().geometry().width() >= text_pane.column_width()


def test_the_panels_leave_the_canvas_a_minimum_width(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    layer_column, canvas, panel = window._splitter.sizes()
    assert layer_column >= window.layers_panel.minimumWidth()
    assert panel >= window.extract_panel.minimumWidth()
    assert canvas >= 240


def test_the_headers_rule_their_own_columns(qtbot: QtBot, extract_png: Path) -> None:
    view = _panel(qtbot, extract_png)._extract_view
    ruler = " ".join(f"{index:02x}" for index in range(BYTES_PER_ROW))
    assert view.hex_pane.header_text() == ruler
    assert len(ruler) == ASCII_START - 2
    # The two headers are one height, so the rows of both panes start level.
    assert view.text_pane._header.height() == view.hex_pane._header.height()
    hex_top = view.hex_pane.mapTo(view, view.hex_pane.viewport().geometry().topLeft()).y()
    text_top = view.text_pane.mapTo(view, view.text_pane.viewport().geometry().topLeft()).y()
    assert hex_top == text_top


def test_the_header_offers_the_encodings(qtbot: QtBot, extract_png: Path) -> None:
    panel = _panel(qtbot, extract_png)
    view = panel._extract_view
    header = view.text_pane._header
    assert header.label() == "ASCII"  # ASCII by default
    actions = view._encoding_menu().actions()
    assert [action.text() for action in actions] == [
        label for _encoding, label in text.EXTRACT_ENCODINGS
    ]
    assert [action.isChecked() for action in actions] == [True, False, False, False]
    requested: list[str] = []
    panel.encoding_requested.connect(requested.append)
    actions[2].trigger()  # UTF-16LE
    assert requested == [ExtractEncoding.UTF16_LE.value]
    view.set_rows([], ExtractEncoding.UTF16_LE)
    assert header.label() == "UTF-16LE"
    assert view._encoding_menu().actions()[2].isChecked()


def test_the_header_lines_up_with_the_hex_digits(qtbot: QtBot, extract_png: Path) -> None:
    pane = _panel(qtbot, extract_png)._extract_view.hex_pane
    image = pane.grab().toImage()
    viewport = pane.viewport().geometry()
    header_start = pane._header.x() + pane.gutter_width + 1
    header = _dark_columns(image, header_start, image.width(), 0, viewport.top())
    data = _dark_columns(image, viewport.left(), image.width(), viewport.top(), image.height())
    assert header
    assert data
    assert abs(header[0] - data[0]) <= 2


def test_the_gutter_paints_the_offsets(qtbot: QtBot, extract_png: Path) -> None:
    pane = _panel(qtbot, extract_png)._extract_view.hex_pane
    image = pane.grab().toImage()
    viewport = pane.viewport().geometry()
    gutter = _dark_columns(image, 0, viewport.left(), viewport.top(), image.height())
    assert gutter
    assert max(gutter) < viewport.left()
    assert pane._gutter.geometry().top() == viewport.top()


def test_note_rows_have_no_offset(qtbot: QtBot, extract_png: Path) -> None:
    panel = _panel(qtbot, extract_png)
    panel._extract_search.setText("zzz")  # matches nothing, so only a note is left
    view = panel._extract_view
    assert view.hex_pane.document().blockCount() == 1
    assert view.hex_pane.offset_text(0) == ""
    assert view.text_pane.toPlainText() == ""


def test_an_empty_selection_shows_a_note_without_an_offset(qtbot: QtBot, extract_png: Path) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    _show(panel, _state(extract_png, BitsMask(frozenset())))
    view = panel._extract_view
    assert view.hex_pane.toPlainText() == "（无数据）"
    assert view.text_pane.toPlainText() == ""
    assert view.hex_pane.offset_text(0) == ""


def test_a_panel_without_an_image_paints_no_chrome(qtbot: QtBot) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    panel.set_state(ViewerState(), None)
    view = panel._extract_view
    assert view.hex_pane.toPlainText() == ""
    assert view.text_pane.toPlainText() == ""
    assert view.hex_pane._gutter.size().isEmpty()
    assert view.hex_pane._header.size().isEmpty()


def test_the_channel_list_holds_only_the_channels_that_carry_bits(
    qtbot: QtBot, extract_png: Path
) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    state = _state(extract_png)
    _show(panel, state)
    combo = panel._channel_combo
    assert combo.count() == 6
    assert combo.currentText() == "RGB"
    pair = _with_bits(state, {BitChoice("R", 3), BitChoice("B", 0)})
    _show(panel, pair, bits_layer=len(pair.layers) - 1)
    assert [combo.itemText(index) for index in range(combo.count())] == ["RB", "BR"]
    assert combo.currentText() == "RB"
    single = _with_bits(state, {BitChoice("G", 0)})
    _show(panel, single, bits_layer=len(single.layers) - 1)
    assert [combo.itemText(index) for index in range(combo.count())] == ["G"]
    assert not combo.isEnabled()


def test_the_channel_list_follows_the_stream_a_shadowing_mask_packs(
    qtbot: QtBot, extract_png: Path
) -> None:
    """The order row describes the bytes; the mask in hand may not be the one packing.

    Bits selections replace one another, so a mask above the one the grid shows is
    what the stream reads — the list has to follow that, not the checked boxes.
    """
    state = _state(extract_png)
    layered = replace(
        state,
        layers=(
            Layer(BitsMask(frozenset({BitChoice("R", 0)}))),
            Layer(BitsMask(frozenset({BitChoice("B", 2)}))),
        ),
    )
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    _show(panel, layered, bits_layer=0)  # the one in hand, shadowed by the one above
    combo = panel._channel_combo
    assert panel._bits_editor._matrix._boxes[BitChoice("R", 0)].isChecked()  # the grid: in hand
    assert panel._bits_editor._shadow_note.text() == text.MASK_BITS_SHADOWED
    assert [combo.itemText(index) for index in range(combo.count())] == ["B"]  # the list: packed


def test_the_bit_grid_reads_the_mask_in_hand(qtbot: QtBot, extract_png: Path) -> None:
    chosen = frozenset({BitChoice("R", 3), BitChoice("B", 0)})
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    _show(panel, _state(extract_png, BitsMask(chosen)), bits_layer=0)
    matrix = panel._bits_editor._matrix
    assert matrix._boxes[BitChoice("R", 3)].isChecked()
    assert matrix._boxes[BitChoice("B", 0)].isChecked()
    assert not matrix._boxes[BitChoice("R", 0)].isChecked()


def test_the_grid_shows_every_bit_without_a_bits_mask(qtbot: QtBot, extract_png: Path) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    _show(panel, _state(extract_png))
    assert all(box.isChecked() for box in panel._bits_editor._matrix._boxes.values())


def test_the_grid_and_the_presets_speak_for_the_mask_in_hand(
    qtbot: QtBot, extract_png: Path
) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    _show(panel, _state(extract_png, BitsMask(frozenset({BitChoice("R", 0)}))), bits_layer=0)
    asked: list[tuple[object, ...]] = []
    panel.bit_clicked.connect(lambda *args: asked.append(args))
    panel.original_requested.connect(lambda layer: asked.append(("all", layer)))
    panel.lsbs_requested.connect(lambda layer: asked.append(("lsbs", layer)))
    panel.plane_step.connect(lambda layer, delta: asked.append(("plane", layer, delta)))
    panel._bits_editor._matrix.bit_clicked.emit("G", 2, True)
    panel._bits_editor.original_button.click()
    panel._bits_editor.lsbs_button.click()
    panel._bits_editor.plane_next.click()
    assert asked == [(0, "G", 2, True), ("all", 0), ("lsbs", 0), ("plane", 0, 1)]


def test_the_grid_says_when_the_mask_in_hand_is_shadowed(qtbot: QtBot, extract_png: Path) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    state = _state(
        extract_png,
        BitsMask(frozenset({BitChoice("R", 0)})),
        BitsMask(frozenset({BitChoice("G", 0)})),
    )
    _show(panel, state, bits_layer=0)
    assert not panel._bits_editor._shadow_note.isHidden()
    assert panel._bits_editor._shadow_note.text() == text.MASK_BITS_SHADOWED
    _show(panel, state, bits_layer=1)
    assert panel._bits_editor._shadow_note.isHidden()


def test_a_deep_channel_keeps_its_grid_wide_enough_for_every_bit(qtbot: QtBot) -> None:
    """Sixteen columns do not fit the panel: the grid scrolls rather than squeezing."""
    planes = (SamplePlane("L", 0, 16, SampleOrigin.RAW),)
    image = make_image(np.zeros((2, 2, 1), dtype=np.uint16), planes)
    state = ViewerState(image=image, layers=layers_of(BitsMask(frozenset({BitChoice("L", 0)}))))
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    panel.resize(640, 700)
    panel.show()
    _show(panel, state, bits_layer=0)
    matrix = panel._bits_editor._matrix
    assert sorted(choice.bit for choice in matrix._boxes) == list(range(16))
    assert matrix.width() >= matrix.minimumSizeHint().width()


def test_the_order_controls_follow_the_state(qtbot: QtBot, extract_png: Path) -> None:
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    state = set_channel_order(_state(extract_png), ("B", "G", "R"))
    state = set_bit_order(state, BitOrder.LSB)
    state = set_scan_order(state, ScanOrder.YZ)
    _show(panel, state)
    assert panel._channel_combo.currentText() == "BGR"
    assert button(panel._bit_order, "LSB").isChecked()
    assert not button(panel._bit_order, "MSB").isChecked()
    assert button(panel._scan, "YZ").isChecked()
    assert not button(panel._scan, "XY").isChecked()


def test_picking_an_order_reports_it(qtbot: QtBot, extract_png: Path) -> None:
    panel = _panel(qtbot, extract_png)
    channels: list[object] = []
    bit_orders: list[str] = []
    scans: list[str] = []
    panel.channel_order_requested.connect(channels.append)
    panel.bit_order_requested.connect(bit_orders.append)
    panel.scan_requested.connect(scans.append)
    panel._channel_combo.setCurrentIndex(5)
    button(panel._bit_order, "LSB").click()
    button(panel._scan, "YZ").click()
    assert channels == [("B", "G", "R")]
    assert bit_orders == [BitOrder.LSB.value]
    assert scans == [ScanOrder.YZ.value]


def test_the_order_row_shares_one_band_across_the_panel(qtbot: QtBot, extract_png: Path) -> None:
    panel = _panel(qtbot, extract_png)
    controls = (panel._channel_combo, panel._bit_order, panel._scan)
    assert {widget.height() for widget in controls} == {theme.CONTROL_HEIGHT}
    assert len({widget.geometry().center().y() for widget in controls}) == 1
    right = panel.width() - panel.contentsMargins().right()
    assert max(widget.geometry().right() for widget in controls) <= right
    assert panel._bit_order.geometry().right() < panel._scan.geometry().left()


def test_the_dump_follows_the_scan_order(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "column.png"
    image = Image.new("RGB", (3, 2))
    for row in range(2):
        image.putpixel((0, row), (1, 0, 0))  # the lowest red bit, first column only
    image.save(path)
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    state = _state(path, BitsMask(frozenset({BitChoice("R", 0)})))
    _show(panel, state)
    pane = panel._extract_view.hex_pane
    assert pane.toPlainText().startswith("90")
    _show(panel, set_scan_order(state, ScanOrder.YZ))
    assert pane.toPlainText().startswith("c0")


def test_the_stream_follows_the_pixels_a_region_mask_kept(qtbot: QtBot, extract_png: Path) -> None:
    """A mask changes the bytes, not just the picture: the dump loses those pixels."""
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    every = _state(extract_png, BitsMask(frozenset({BitChoice("R", 0)})))
    _show(panel, every)
    whole = panel._extract_view.hex_pane.toPlainText()
    _show(panel, replace(every, layers=(*every.layers, Layer(RegionMask("left < 4")))))
    halved = panel._extract_view.hex_pane.toPlainText()
    # Half the pixels, so half the bytes the dump spells out.
    assert halved.split().count("00") < whole.split().count("00")


def test_a_kept_channel_order_the_list_cannot_offer_leads_the_list(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 2), (1, 2, 3, 4)).save(path)
    panel = ExtractPanel()
    qtbot.addWidget(panel)
    # Four channels carrying bits leave the list offering only the order and its
    # reverse, so the order kept from a three-channel image shows first of all.
    state = set_channel_order(_state(path), ("B", "G", "R"))
    _show(panel, state)
    combo = panel._channel_combo
    assert [combo.itemText(index) for index in range(combo.count())] == ["BGRA", "RGBA", "ABGR"]
    assert combo.currentText() == "BGRA"
    requested: list[object] = []
    panel.channel_order_requested.connect(requested.append)
    combo.setCurrentIndex(1)
    assert requested == [("R", "G", "B", "A")]

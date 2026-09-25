from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QPointingDevice
from PySide6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from pixelsb.domain import geometry
from pixelsb.domain.extract import format_extract
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    BitsMask,
    CropMask,
    DisplayFormat,
    ExtractEncoding,
    InvertMask,
    PixelCoord,
    RegionMask,
    ScanOrder,
    ThresholdMask,
    XorMask,
)
from pixelsb.domain.transitions import (
    add_layer,
    select_only,
    set_cursor,
    set_format,
    set_mask_text,
    set_zoom,
)
from pixelsb.io.loading import image_from_pixels
from pixelsb.ui import painting, text, theme
from pixelsb.ui.main_window import MainWindow
from pixelsb.ui.side_panels import Panel
from pixelsb.ui.text import readout_text
from tests.support import bits_of, button, make_image, mask_of, masks, planes_of, planes_rgba


def test_open_bit_plane_and_detail_text(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(set_cursor, coord=PixelCoord(1, 0)))
    window.apply(partial(select_only, plane="R", bit=0))
    window.apply(partial(set_format, fmt=DisplayFormat.BINARY))
    state = window.store.state
    assert state.cursor == PixelCoord(1, 0)
    assert bits_of(state) == frozenset({BitChoice("R", 0)})
    assert window.canvas.displayed_size() == (2, 2)
    detail = readout_text(state, window._raster)
    assert "光标 (1, 0)" in detail
    assert "画面 R0" in detail


def test_left_click_keeps_the_cursor_where_hover_left_it(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    window.open_path(rgb_png)
    zoom = int(window.store.state.zoom)
    # The first synthetic move only enters the canvas; the second one moves.
    qtbot.mouseMove(window.canvas, QPoint(0, 0))
    qtbot.mouseMove(window.canvas, QPoint(zoom + 1, 1))
    assert window.store.state.cursor == PixelCoord(1, 0)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    assert window.store.state.cursor == PixelCoord(1, 0)


def test_failed_open_keeps_the_current_image(qtbot: QtBot, rgb_png: Path, tmp_path: Path) -> None:
    messages: list[str] = []
    window = MainWindow(reporter=messages.append)
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.open_path(tmp_path / "missing.png")
    assert window.store.state.image is not None
    assert window.store.state.image.path == rgb_png
    assert messages


def test_a_region_mask_reads_the_values_the_stack_leaves_below_it(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    # A bare channel is the value as stored, whichever bits are selected above it.
    window.apply(partial(add_layer, mask=RegionMask("R == 255")))
    assert window._raster is not None
    first = window._raster.live
    assert first is not None
    assert first.tolist() == [[True, False], [False, False]]
    window.apply(partial(add_layer, mask=BitsMask(frozenset({BitChoice("R", 7)}))))
    assert window._raster is not None
    second = window._raster.live
    assert second is not None
    assert np.array_equal(first, second)
    # .bits reads what the selection builds, so a mask on top of it moves with it:
    # R.7 is the bit itself, and the pixel whose R is 255 has it set.
    window.apply(partial(add_layer, mask=RegionMask("R.bits == 1")))
    assert window._raster is not None
    third = window._raster.live
    assert third is not None
    assert third.tolist() == first.tolist()
    window.apply(partial(add_layer, mask=RegionMask("R.bits == 0")))
    assert window._raster is not None
    fourth = window._raster.live
    assert fourth is not None
    assert not fourth.any()


def test_a_bad_region_expression_is_reported_and_keeps_the_image(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(set_mask_text, text="A > 0"))
    assert window._raster is not None
    assert window._raster.live is None  # a mask that cannot run filters nothing
    (failure,) = window._raster.failures
    assert "未知字段" in failure.message
    assert "命令错误" in window._status_info.text()
    assert masks(window.store.state) == (RegionMask("A > 0"),)
    assert window._filter_edit.styleSheet()  # the box says which layer is at fault


def test_filter_box_typing_and_focus_flow(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "B >= R")
    assert edit.text() == "B >= R"
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert masks(window.store.state) == (RegionMask("B >= R"),)
    assert window.focusWidget() is edit
    qtbot.keyClicks(edit, " and 0 <= top")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert masks(window.store.state) == (RegionMask("B >= R and 0 <= top"),)
    qtbot.keyClick(edit, Qt.Key.Key_Escape)
    assert masks(window.store.state) == (RegionMask("B >= R and 0 <= top"),)  # the words go first
    assert edit.text() == ""
    assert window.focusWidget() is edit
    qtbot.keyClick(edit, Qt.Key.Key_Escape)
    assert masks(window.store.state) == ()  # an empty box: Esc drops the operation
    assert window.focusWidget() is window.canvas


def test_the_filter_box_edits_the_region_layer_the_panel_selected(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(add_layer, mask=RegionMask("R > 0")))
    window.apply(partial(add_layer, mask=RegionMask("G > 0")))
    assert window._filter_edit.text() == "G > 0"  # the newest layer is the one being edited
    window.layers_panel._select(0)
    assert window._filter_edit.text() == "R > 0"  # the box follows the panel's selection
    window._filter_edit.setText("B > 0")
    window._apply_filter_text()
    assert masks(window.store.state) == (RegionMask("B > 0"), RegionMask("G > 0"))


def test_typing_a_filter_with_no_region_layer_selected_adds_one(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """The box edits the layer in hand; with none in hand it makes a new one on top."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(InvertMask())
    window.layers_panel._select(None)
    assert window._filter_edit.text() == ""
    window._filter_edit.setText("B >= R")
    window._apply_filter_text()
    assert masks(window.store.state) == (InvertMask(), RegionMask("B >= R"))
    # The new mask is now the one in hand, so the next edit writes it.
    window._filter_edit.setText("G >= R")
    window._apply_filter_text()
    assert masks(window.store.state) == (InvertMask(), RegionMask("G >= R"))


def test_typing_a_command_with_no_layer_in_hand_adds_its_mask(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    assert window._filter_edit.text() == ""
    window._filter_edit.setText("b.0")
    window._apply_filter_text()
    assert masks(window.store.state) == (BitsMask(frozenset({BitChoice("B", 0)})),)
    assert window._filter_edit.text() == "b.0"  # the new layer is in hand, command and all


def test_the_grid_and_the_command_input_agree(qtbot: QtBot, rgb_png: Path) -> None:
    """Typing ``b.0`` checks the grid's B 0 box; a grid click writes the line back."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._filter_edit.setText("b.0")
    window._apply_filter_text()
    matrix = window.extract_panel._bits_editor._matrix
    assert matrix._boxes[BitChoice("B", 0)].isChecked()
    assert not matrix._boxes[BitChoice("R", 0)].isChecked()
    assert window._filter_edit.text() == "b.0"
    matrix.bit_clicked.emit("B", 1, False)  # a click on the grid
    assert bits_of(window.store.state) == frozenset({BitChoice("B", 0), BitChoice("B", 1)})
    assert window._filter_edit.text() == "b.0 or b.1"


def test_a_line_of_two_kinds_is_refused_and_names_no_operation(qtbot: QtBot, rgb_png: Path) -> None:
    """``b > r and b.0`` is two operations in one line: it is refused, not guessed at."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._filter_edit.setText("b > r and b.0")
    window._apply_filter_text()
    assert masks(window.store.state) == ()
    assert "一行只写一个操作" in window._status_info.text()
    assert window._filter_edit.styleSheet()  # the box marks the line it refused
    assert window._filter_edit.text() == "b > r and b.0"  # kept for fixing
    window._filter_edit.setText("b > r")
    window._apply_filter_text()
    assert masks(window.store.state) == (RegionMask("b > r"),)
    window._filter_edit.setText("b.0")
    window._apply_filter_text()
    assert masks(window.store.state) == (BitsMask(frozenset({BitChoice("B", 0)})),)
    # A command written over the region's own command replaces it: kind and all.
    assert window._filter_edit.text() == "b.0"


def test_typing_a_line_in_pieces_never_doubles_an_operation(qtbot: QtBot, rgb_png: Path) -> None:
    """A line is applied when it is asked for, and a half-typed one changes nothing.

    Every prefix of ``b>r and b.`` goes through the box here; the half-typed
    ``b>r and b.`` used to be filed as a region mask over the layer in hand,
    which the finished line then wrote a second region onto.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    line = "b>r and b."
    for end in range(1, len(line) + 1):
        window._filter_edit.setText(line[:end])
        window._apply_filter_text()
        assert len(window.store.state.layers) <= 1, f"prefix {line[:end]!r}"
    assert masks(window.store.state) == (RegionMask("b > r"),)
    assert window._filter_edit.text() == line  # the box keeps the typist's line
    window._filter_edit.setText("b>r and b.0")  # finished, but still two operations
    window._apply_filter_text()
    assert masks(window.store.state) == (RegionMask("b > r"),)  # refused: the stack stands


def test_typing_applies_nothing_until_enter(qtbot: QtBot, rgb_png: Path) -> None:
    """No preview: the stack stands still while the line is written."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "b > r")
    assert masks(window.store.state) == ()
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert masks(window.store.state) == (RegionMask("b > r"),)


def test_typing_is_never_rewritten_mid_sentence(qtbot: QtBot, rgb_png: Path) -> None:
    """The box is the typist's while a line is written; Enter writes it back as its command."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "b")
    window._commit_filter()
    assert edit.text() == "b"  # the command's own wording: nothing to write back
    qtbot.keyClicks(edit, " or g")
    window._apply_filter_text()
    assert edit.text() == "b or g"  # the space survived
    expected = frozenset(
        {BitChoice("B", bit) for bit in range(8)} | {BitChoice("G", bit) for bit in range(8)}
    )
    assert bits_of(window.store.state) == expected
    window.canvas.setFocus()  # focus moves on: the typist's words stay put
    assert edit.text() == "b or g"
    edit.setFocus()
    qtbot.keyClicks(edit, " ")  # a trailing space…
    window._apply_filter_text()
    assert edit.text() == "b or g "  # …survives an apply: the box owns its own words
    window._commit_filter()  # Enter is what writes the line back as the operation it ran
    assert edit.text() == "g or b"  # the image's channel order, the recipe row's wording
    matrix = window.extract_panel._bits_editor._matrix
    matrix.bit_clicked.emit("R", 7, True)  # an outside change is what rewrites the line
    assert edit.text() == "r.7"


def test_enter_writes_the_line_back_as_the_command_it_ran(qtbot: QtBot, rgb_png: Path) -> None:
    """The box and the recipe row read alike: Enter keeps the command's own wording."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    image = window.store.state.image
    assert image is not None
    for typed, command in (
        ("b>r", "b > r"),
        ("thr 0x80", "thr 128"),
        ("xor 0xff", "xor 0xFF"),
        ("R or B", "r or b"),
        (" b.0 ", "b.0"),
    ):
        window._filter_edit.setText(typed)
        window._commit_filter()
        assert window._filter_edit.text() == command, typed
        assert text.mask_detail(window.store.state.layers[-1].mask, image.planes) == command, typed


def test_enter_leaves_a_line_it_could_not_run_as_typed(qtbot: QtBot, rgb_png: Path) -> None:
    """Only a line that ran is rewritten: a half-written or refused one is the typist's."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window._filter_edit.setText("b>r and b.")  # a sentence still being written
    window._commit_filter()
    assert window._filter_edit.text() == "b>r and b."
    assert masks(window.store.state) == ()
    window._filter_edit.setText("thr 1x")  # refused, and kept for fixing
    window._commit_filter()
    assert window._filter_edit.text() == "thr 1x"
    assert masks(window.store.state) == ()
    assert "命令错误" in window._status_info.text()
    assert window._filter_edit.styleSheet()  # the box marks the line it refused


def test_a_second_command_replaces_the_first_instead_of_stacking(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """Rewriting the box rewrites the operation it shows, even when the kind changes.

    Typing ``b>r`` and then ``b.0`` over it used to leave the region in the stack
    and add a second operation beside it; the box names one operation, so it edits
    that one. Stacking a second operation is Shift+Enter's job.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setText("b>r")
    window._commit_filter()
    assert masks(window.store.state) == (RegionMask("b > r"),)
    edit.clear()  # delete the words, write another line
    edit.setText("b.0")
    window._commit_filter()
    assert masks(window.store.state) == (BitsMask(frozenset({BitChoice("B", 0)})),)
    assert edit.text() == "b.0"


def test_escape_takes_the_words_first_and_the_operation_second(qtbot: QtBot, rgb_png: Path) -> None:
    """One Esc empties the box; the next one, on an empty box, drops that operation."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "thr 128")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert masks(window.store.state) == (ThresholdMask(128),)
    qtbot.keyClick(edit, Qt.Key.Key_Escape)
    assert edit.text() == ""  # the words go…
    assert masks(window.store.state) == (ThresholdMask(128),)  # …the operation stays
    assert window.focusWidget() is edit  # and the box keeps the keyboard
    qtbot.keyClick(edit, Qt.Key.Key_Escape)
    assert masks(window.store.state) == ()
    assert window.focusWidget() is window.canvas


def test_shift_enter_stacks_the_line_instead_of_writing_over_the_one_in_hand(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """The shift keeps the operation being edited and adds a second one on top."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "thr 128")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert masks(window.store.state) == (ThresholdMask(128),)
    edit.clear()
    qtbot.keyClicks(edit, "thr 200")
    qtbot.keyClick(edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert masks(window.store.state) == (ThresholdMask(128), ThresholdMask(200))
    assert edit.text() == "thr 200"  # written back as its command, like a plain Enter
    edit.clear()  # nothing to stack: shift adds nothing
    qtbot.keyClick(edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert masks(window.store.state) == (ThresholdMask(128), ThresholdMask(200))
    qtbot.keyClicks(edit, "thr 64")
    qtbot.keyClick(edit, Qt.Key.Key_Return)  # a plain Enter edits the one in hand again
    assert masks(window.store.state) == (ThresholdMask(128), ThresholdMask(64))


def test_typing_again_clears_the_complaint_about_a_refused_line(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """The mark says 改这里: the next keystroke takes it back, a readable line applies."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._filter_edit.setText("thr 1x")
    window._apply_filter_text()
    window._filter_edit.setFocus()
    qtbot.keyClick(window._filter_edit, Qt.Key.Key_Backspace)  # typing again clears the mark
    assert window._filter_edit.styleSheet() == ""
    assert "命令错误" not in window._status_info.text()
    assert masks(window.store.state) == ()  # and still applies nothing on its own
    window._filter_edit.setText("thr 128")  # the next readable line clears it again
    window._apply_filter_text()
    assert masks(window.store.state) == (ThresholdMask(128),)
    assert "命令错误" not in window._status_info.text()
    assert window._filter_edit.styleSheet() == ""


def test_a_filter_line_waiting_for_enter_survives_a_state_change(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """Typing is newer than the state it was typed against, so nothing may undo it."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._filter_edit.setFocus()
    qtbot.keyClicks(window._filter_edit, "B >= R")
    window.apply(partial(set_cursor, coord=PixelCoord(1, 1)))  # a cursor move mid-sentence
    assert window._filter_edit.text() == "B >= R"
    window._commit_filter()
    assert masks(window.store.state) == (RegionMask("B >= R"),)


def test_a_crop_layer_crops_the_canvas_without_changing_the_stream(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(set_mask_text, text="grid(0, 0, 2, 2)"))
    before = window.extract_panel.extract_data()
    window.apply(partial(add_layer, mask=CropMask()))
    assert window.canvas.displayed_size() == (1, 1)
    assert window.extract_panel.extract_data() == before


def test_switching_an_invert_layer_off_restores_the_pixels(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    original = window.canvas._frame.rgb
    assert original is not None
    original = original.copy()
    window.layers_panel.add_requested.emit(InvertMask())
    inverted = window.canvas._frame.rgb
    assert inverted is not None
    assert np.array_equal(inverted, 255 - original)
    window.layers_panel.enabled_requested.emit(0, False)
    restored = window.canvas._frame.rgb
    assert restored is not None
    assert np.array_equal(restored, original)


def test_the_command_box_writes_whatever_layer_is_selected(qtbot: QtBot, rgb_png: Path) -> None:
    """The box shows the selected mask's command and rewrites that layer in place."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    panel = window.layers_panel
    panel.add_requested.emit(ThresholdMask())
    assert window._filter_edit.text() == "thr 128"
    panel.add_requested.emit(XorMask())
    assert window._filter_edit.text() == "xor 0xFF"
    panel._select(0)
    assert window._filter_edit.text() == "thr 128"
    window._filter_edit.setText("thr 200")
    window._apply_filter_text()
    assert mask_of(window.store.state, ThresholdMask).level == 200
    window._filter_edit.setText("inv")  # a different kind rewrites the layer all the same
    window._apply_filter_text()
    assert masks(window.store.state) == (InvertMask(), XorMask(0xFF))
    assert window._filter_edit.text() == "inv"


def test_clicking_a_recipe_row_shows_its_command_and_edits_that_operation(
    qtbot: QtBot, rgb_png: Path
) -> None:
    """Picking an operation with the mouse is what puts it in the command input."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    panel = window.layers_panel
    panel.add_requested.emit(ThresholdMask())
    panel.add_requested.emit(XorMask())
    assert window._filter_edit.text() == "xor 0xFF"
    row = panel._rows[0]
    qtbot.mouseClick(row._detail, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert panel.selected == 0
    assert window._filter_edit.text() == "thr 128"
    window._filter_edit.setText("thr 200")
    window._apply_filter_text()
    assert mask_of(window.store.state, ThresholdMask).level == 200
    assert masks(window.store.state) == (ThresholdMask(200), XorMask(0xFF))
    qtbot.mouseClick(panel._rows[None]._name, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert panel.selected is None
    assert window._filter_edit.text() == ""


def test_the_bits_grid_edits_the_layer_its_editor_belongs_to(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(BitsMask(frozenset({BitChoice("R", 0)})))
    matrix = window.extract_panel._bits_editor._matrix
    assert matrix._boxes[BitChoice("R", 0)].isChecked()
    matrix.bit_clicked.emit("G", 1, False)  # a click on the grid
    assert bits_of(window.store.state) == frozenset({BitChoice("R", 0), BitChoice("G", 1)})
    window.extract_panel._bits_editor.channel_toggle.emit("B", True)
    expected = {BitChoice("R", 0), BitChoice("G", 1)}
    expected.update(BitChoice("B", bit) for bit in range(8))
    assert bits_of(window.store.state) == frozenset(expected)


def test_switching_tools_hands_focus_back_to_the_canvas(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._mode_select.click()
    assert window.focusWidget() is window.canvas  # space must reach the canvas, not the button
    qtbot.keyPress(window.canvas, Qt.Key.Key_Space)
    assert window._mode_select.isChecked()  # space pans, it does not toggle the tool
    assert window.canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor
    qtbot.keyRelease(window.canvas, Qt.Key.Key_Space)
    assert window.canvas.cursor().shape() == Qt.CursorShape.CrossCursor


def test_select_mode_previews_then_commits_on_enter(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    zoom = int(window.store.state.zoom)
    assert window._mode_move.isChecked()
    assert not window._mode_select.isChecked()
    window._mode_select.click()
    assert window._mode_select.isChecked()
    assert not window._mode_move.isChecked()
    qtbot.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(window.canvas, QPoint(zoom + 1, zoom + 1))
    assert "选区" in window._status_view.text()
    qtbot.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(zoom + 1, zoom + 1))
    # Still a preview: nothing lands in the stack until Enter commits the rect.
    assert "回车加为区域操作" in window._status_view.text()
    assert "通过 4/4" in window._status_view.text()
    assert masks(window.store.state) == ()
    assert window._filter_count.text() == ""
    qtbot.keyClick(window.canvas, Qt.Key.Key_Return)
    assert masks(window.store.state) == (RegionMask("rect(0, 0, 2, 2)"),)
    assert window._filter_edit.text() == "rect(0, 0, 2, 2)"
    assert window._status_view.text().startswith("光标")
    qtbot.keyClick(window.canvas, Qt.Key.Key_Escape)
    assert masks(window.store.state) == (
        RegionMask("rect(0, 0, 2, 2)"),
    )  # Esc only drops a preview


def test_committing_a_region_adds_a_mask_of_its_own(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_mask_text, text="B >= 200"))
    assert window._raster is not None
    assert window._raster.live_count == 1  # only the blue pixel at (0, 1)
    extract = window.extract_panel._extract_rows
    window._on_region_selected(1, 0, 1, 1)  # the right column holds no match
    assert masks(window.store.state) == (RegionMask("B >= 200"),)  # preview only
    assert "通过 0/4" in window._status_view.text()
    window._on_region_committed(1, 0, 1, 1)
    assert masks(window.store.state) == (RegionMask("B >= 200"), RegionMask("rect(1, 0, 2, 2)"))
    assert window._raster is not None
    assert window._raster.live_count == 0
    assert window.extract_panel._extract_rows != extract


def test_escape_drops_the_preview_without_touching_the_stack(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_mask_text, text="B >= 200"))
    window._on_region_selected(1, 0, 1, 1)
    window._on_region_canceled()
    assert masks(window.store.state) == (RegionMask("B >= 200"),)
    assert window._raster is not None
    assert window._raster.live_count == 1
    assert window._status_view.text().startswith("光标")


def test_the_extract_tabs_and_the_state_agree(qtbot: QtBot, extract_png: Path) -> None:
    from pixelsb.domain.transitions import set_extract_encoding

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(extract_png)
    view = window.extract_panel._extract_view
    rows = format_extract("中".encode("utf-16-le"))
    view.set_rows(rows, ExtractEncoding.UTF16_LE)
    assert view.text_pane.toPlainText() == "中"
    view.set_rows(rows, ExtractEncoding.ASCII)
    assert view.text_pane.toPlainText() == "-N"  # the same bytes, read as ASCII
    assert view.text_pane._header.label() == "ASCII"
    # The header asks for an encoding; the state answers and the header follows.
    view.choose_encoding(ExtractEncoding.UTF16_BE)
    assert window.store.state.extract_encoding is ExtractEncoding.UTF16_BE
    window.apply(partial(set_extract_encoding, encoding=ExtractEncoding.UTF8))
    assert view.text_pane._header.label() == "UTF-8"


def test_the_bit_shortcuts_bring_their_own_mask(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.canvas.setFocus()
    qtbot.keyClick(window.canvas, Qt.Key.Key_R)
    assert masks(window.store.state) == (BitsMask(frozenset({BitChoice("R", 0)})),)
    qtbot.keyClick(window.canvas, Qt.Key.Key_2)
    assert bits_of(window.store.state) == frozenset({BitChoice("G", 0)})
    qtbot.keyClick(window.canvas, Qt.Key.Key_B)  # every channel letter works, B included
    assert bits_of(window.store.state) == frozenset({BitChoice("B", 0)})
    qtbot.keyClick(window.canvas, Qt.Key.Key_BracketRight)
    assert bits_of(window.store.state) == frozenset({BitChoice("B", 1)})


def test_the_bit_shortcuts_edit_the_mask_the_grid_shows(qtbot: QtBot, rgb_png: Path) -> None:
    """The grid and the keyboard write the same mask: the selected bits one."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(add_layer, mask=BitsMask(frozenset({BitChoice("R", 0)}))))
    window.apply(partial(add_layer, mask=InvertMask()))
    window.layers_panel._select(0)
    window.canvas.setFocus()
    qtbot.keyClick(window.canvas, Qt.Key.Key_G)
    assert masks(window.store.state) == (
        BitsMask(frozenset({BitChoice("G", 0)})),
        InvertMask(),
    )


def test_the_bits_steppers_make_the_mask_they_edit(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    # The walk itself belongs to the transitions; here the buttons just have to
    # be wired to it, one step each way.
    window.apply(partial(add_layer, mask=BitsMask(frozenset({BitChoice("R", 0)}))))
    editor = window.extract_panel._bits_editor
    editor.plane_next.click()
    assert bits_of(window.store.state) == frozenset({BitChoice("G", 7)})  # one on from R0
    editor.plane_prev.click()
    assert bits_of(window.store.state) == frozenset({BitChoice("R", 0)})
    editor.channel_next.click()  # a view that is not one channel starts at its head
    assert bits_of(window.store.state) == frozenset({BitChoice("R", bit) for bit in range(8)})
    editor.channel_next.click()
    assert bits_of(window.store.state) == frozenset({BitChoice("G", bit) for bit in range(8)})


def test_the_bits_presets_rewrite_the_mask_they_edit(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(BitsMask(frozenset({BitChoice("R", 3)})))
    window.extract_panel._bits_editor.lsbs_button.click()
    assert bits_of(window.store.state) == frozenset(
        {BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)}
    )
    window.extract_panel._bits_editor.original_button.click()
    assert len(bits_of(window.store.state)) == 24
    assert masks(window.store.state) == (BitsMask(bits_of(window.store.state)),)


def test_typing_in_the_filter_box_keeps_its_keys(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_zoom, zoom=4))
    window._filter_edit.setFocus()
    qtbot.keyClicks(window._filter_edit, "+B")
    assert window._filter_edit.text() == "+B"
    assert window.store.state.zoom == 4
    window.canvas.setFocus()
    qtbot.keyClick(window.canvas, Qt.Key.Key_Plus)
    assert window.store.state.zoom == 5


def test_arrow_key_moves_the_cursor(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(0, 0)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(1, 0)


def test_the_zoom_slider_is_geometric(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_zoom, zoom=8.0))
    assert window._zoom_slider.value() == geometry.slider_position(8.0)
    window._zoom_slider.setValue(geometry.slider_position(16.0))
    assert window.store.state.zoom == 16.0
    assert window._zoom_label.text() == text.zoom_label(16.0)


def test_the_zoom_label_fits_the_widest_value(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._zoom_label.setText(text.zoom_label(123.4567))
    assert window._zoom_label.sizeHint().width() <= window._zoom_label.width()


def test_a_trackpad_pinch_scales_the_zoom_in_place(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_zoom, zoom=4.0))
    window._zoom_scale(1.25, 100, 50)
    assert window.store.state.zoom == pytest.approx(5.0)
    window._zoom_scale(0.5, 100, 50)
    assert window.store.state.zoom == pytest.approx(2.5)


def test_a_native_pinch_gesture_requests_a_scale(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    window.apply(partial(set_zoom, zoom=4.0))
    before = window.store.state.zoom
    event = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture,
        QPointingDevice.primaryPointingDevice(),
        2,
        QPointF(30, 20),
        QPointF(30, 20),
        QPointF(31, 21),
        0.04,
        QPointF(0, 0),
    )
    # Real delivery invokes the widget's event() with the native gesture.
    assert window.canvas.event(event) is True
    assert window.store.state.zoom == pytest.approx(before * 1.04)
    # The same gesture over the letterbox reaches the canvas through the filter.
    assert window.canvas.eventFilter(window._scroll.viewport(), event) is True
    assert window.store.state.zoom == pytest.approx(before * 1.04 * 1.04)


def test_zooming_keeps_the_pointer_pixel_stationary(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(700, 400)
    window.show()
    window.open_path(extract_png)
    window.apply(partial(set_zoom, zoom=64.0))
    horizontal = window._scroll.horizontalScrollBar()
    vertical = window._scroll.verticalScrollBar()
    horizontal.setValue(120)
    vertical.setValue(60)
    pointer_x, pointer_y = 200, 100
    before = (
        (horizontal.value() + pointer_x) / 64.0,
        (vertical.value() + pointer_y) / 64.0,
    )
    window._zoom_scale(2.0, pointer_x, pointer_y)
    after = (
        (horizontal.value() + pointer_x) / 128.0,
        (vertical.value() + pointer_y) / 128.0,
    )
    assert after[0] == pytest.approx(before[0], abs=0.5)
    assert after[1] == pytest.approx(before[1], abs=0.5)


def test_the_filter_box_gets_a_light_clear_icon(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    button_ = window._filter_edit.findChild(QToolButton)
    assert button_ is not None
    expected = painting.clear_icon().pixmap(16, 16).toImage()
    assert button_.icon().pixmap(16, 16).toImage() == expected
    window._filter_edit.setText("B >= R")
    assert button_.isVisible() is True


def test_the_clear_icon_is_a_light_cross() -> None:
    image = painting.clear_icon().pixmap(32, 32).toImage()
    opaque = [
        (color.red() + color.green() + color.blue())
        for y in range(image.height())
        for x in range(image.width())
        if (color := image.pixelColor(x, y)).alpha() > 200
    ]
    # A muted gray cross; the style's own clear icon is a near-black disc.
    assert opaque
    assert min(opaque) > 250
    assert image.pixelColor(0, 0).alpha() == 0


def test_the_toolbar_is_a_single_row_of_shared_height(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    row = window._filter_edit.parentWidget()
    assert row is not None
    assert row.objectName() == "toolbarRow"
    for widget in (
        window._filter_edit,
        window._filter_count,
        window._mode_move,
        window._mode_select,
        window._zoom_out,
        window._zoom_slider,
        window._zoom_in,
        window._zoom_label,
        window._zoom_fit,
        window._zoom_reset,
        window._format_combo,
        window._export_button,
    ):
        assert widget.parentWidget() is row
    heights = {
        window._filter_edit.height(),
        window._zoom_out.height(),
        window._zoom_fit.height(),
        window._format_combo.height(),
        window._zoom_slider.height(),
        window.extract_panel._extract_search.height(),
    }
    assert heights == {theme.CONTROL_HEIGHT}
    assert row.height() == theme.CONTROL_HEIGHT + 20


def test_panel_and_status_widgets_track_the_state(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    assert window._zoom_label.text() == text.zoom_label(window.store.state.zoom)
    assert "rgb.png" in window._status_info.text()
    assert window._status_view.text().startswith("光标")

    window._format_combo.setCurrentIndex(window._format_combo.findData(DisplayFormat.BINARY.value))
    assert window.store.state.value_format is DisplayFormat.BINARY

    window._filter_edit.setText("B >= R")
    window._commit_filter()
    assert window._filter_count.text() == "通过 3/4"
    assert window._status_view.text().startswith("光标")
    window.grab()


def test_the_extract_order_controls_drive_the_dump(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(extract_png)
    panel = window.extract_panel
    pane = panel._extract_view.hex_pane
    assert pane.toPlainText().startswith("41 42 43")

    panel._channel_combo.setCurrentIndex(panel._channel_combo.count() - 1)
    assert window.store.state.extract_order.planes == ("B", "G", "R")
    assert pane.toPlainText().startswith("43 42 41")

    button(panel._scan, "YZ").click()
    assert window.store.state.extract_order.scan is ScanOrder.YZ
    button(panel._bit_order, "LSB").click()
    assert window.store.state.extract_order.bit_order is BitOrder.LSB
    # Low first hands back each channel's stored byte, in the order just picked.
    assert pane.toPlainText().startswith("c2 42 82")

    # The three settings survive opening another image.
    window.open_path(extract_png)
    assert window.store.state.extract_order.scan is ScanOrder.YZ


def _animated_gif(path: Path) -> Path:
    frames = [Image.new("RGB", (2, 2), color) for color in ((0, 0, 0), (255, 255, 255))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, loop=0)
    return path


def test_frame_steps_load_the_neighbor_frame(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(_animated_gif(tmp_path / "anim.gif"))
    assert window._frame_prev.isEnabled()
    window._step_frame(1)
    assert window.store.state.image is not None
    assert window.store.state.image.frame_index == 1
    assert "帧 2/2" in window._frame_status.text()
    window._step_frame(1)  # already on the last frame; the press is a no-op
    assert window.store.state.image.frame_index == 1
    window._step_frame(-1)
    assert window.store.state.image.frame_index == 0


def test_a_frame_step_keeps_the_stack(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(_animated_gif(tmp_path / "anim.gif"))
    window.layers_panel.add_requested.emit(InvertMask())
    window._step_frame(1)
    assert masks(window.store.state) == (InvertMask(),)


def test_the_export_of_a_transparent_picture_keeps_its_alpha(qtbot: QtBot, tmp_path: Path) -> None:
    """The board the canvas paints is a viewing aid; the file keeps the channel."""
    samples = np.zeros((2, 2, 4), dtype=np.uint16)
    samples[..., :3] = 100
    samples[..., 3] = [[0, 128], [255, 64]]
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_loaded(make_image(samples, planes_rgba()))
    assert window._raster is not None
    destination = tmp_path / "alpha.png"
    window._write_view(destination, window._raster)
    with Image.open(destination) as written:
        assert written.mode == "RGBA"
        exported = np.asarray(written)
    assert exported[..., :3].tolist() == [[[100, 100, 100]] * 2] * 2
    assert exported[..., 3].tolist() == [[0, 128], [255, 64]]
    board = window.canvas._frame.rgb
    assert board is not None
    assert board[0, 0].tolist() == [232, 232, 232]  # the canvas shows the checkerboard


def test_the_export_carries_the_masks_into_the_file(
    qtbot: QtBot, rgb_png: Path, tmp_path: Path
) -> None:
    """What is written is what the canvas shows, mask effects and all."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(InvertMask())
    assert window._raster is not None
    destination = tmp_path / "inverted.png"
    window._write_view(destination, window._raster)
    exported = np.asarray(Image.open(destination).convert("RGB"))
    plain = np.asarray(Image.open(rgb_png).convert("RGB"))
    composed = window.canvas._frame.rgb
    assert composed is not None
    assert np.array_equal(exported, composed)
    assert not np.array_equal(exported, plain)  # the inversion is in the file


def test_the_extract_panel_holds_the_full_stream_for_saving(
    qtbot: QtBot, extract_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(extract_png)
    assert window.extract_panel.extract_data() == b"ABC" * 32
    window.layers_panel.add_requested.emit(RegionMask("left < 4"))
    assert window.extract_panel.extract_data() == b"ABC" * 16  # half the pixels


def test_the_file_info_menu_raises_the_info_page(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    assert window._panels.current is Panel.INFO  # the file-info page leads the rail
    window._panels.set_current(Panel.EXTRACT)
    assert window._panels.current is Panel.EXTRACT
    window._show_info()
    assert window._panels.current is Panel.INFO
    assert window.info_panel.isVisible()
    assert "rgb.png" in window.info_panel._name.text()


def test_open_loaded_shows_a_rendered_image_in_the_canvas(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    rgb = np.zeros((3, 4, 3), np.uint8)
    window.open_loaded(image_from_pixels(rgb_png, rgb))
    assert window.canvas.displayed_size() == (4, 3)
    assert window.store.state.image is not None
    assert planes_of(window.store.state.image) == ("R", "G", "B")


def test_the_view_menu_resets_to_the_image_and_to_the_lowest_bits(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(InvertMask())
    window._original_action.trigger()
    assert masks(window.store.state) == ()
    window._lsb_action.trigger()
    assert masks(window.store.state) == (
        BitsMask(frozenset({BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)})),
    )
    window.layers_panel.add_requested.emit(CropMask())
    window._original_action.trigger()
    assert masks(window.store.state) == ()


def test_the_readout_follows_the_stack(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(set_cursor, coord=PixelCoord(0, 1)))
    assert "B  FF" in readout_text(window.store.state, window._raster)
    window.layers_panel.add_requested.emit(ThresholdMask(level=200))
    assert "B  FF" in readout_text(window.store.state, window._raster)  # 255 survives 200
    window.layers_panel.add_requested.emit(InvertMask())
    assert "B  00" in readout_text(window.store.state, window._raster)


def test_a_cropped_canvas_says_when_the_cursor_pixel_is_not_in_the_view(
    qtbot: QtBot, rgb_png: Path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(partial(set_cursor, coord=PixelCoord(0, 0)))
    window.layers_panel.add_requested.emit(RegionMask("left >= 1"))
    window.layers_panel.add_requested.emit(CropMask())
    assert readout_text(window.store.state, window._raster) == text.NOT_IN_VIEW
    window.apply(partial(set_cursor, coord=PixelCoord(1, 0)))
    assert "R  " in readout_text(window.store.state, window._raster)


def test_open_again_keeps_the_stack_and_prunes_nothing(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.layers_panel.add_requested.emit(BitsMask(frozenset({BitChoice("R", 3)})))
    window.open_path(rgb_png)
    assert masks(window.store.state) == (BitsMask(frozenset({BitChoice("R", 3)})),)

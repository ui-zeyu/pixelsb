from pathlib import Path

import numpy as np
import pytest

from pixelsb.domain.models import (
    MAX_ZOOM,
    BitChoice,
    BitOrder,
    DisplayFormat,
    ExtractEncoding,
    ExtractOrder,
    PixelCoord,
    ScanOrder,
    ViewerState,
)
from pixelsb.domain.transitions import (
    cycle_format,
    move_cursor,
    open_image,
    select_lsb,
    select_only,
    set_bit_order,
    set_channel_order,
    set_cursor,
    set_extract_encoding,
    set_only_matched,
    set_scan_order,
    set_zoom,
    step_channel,
    step_focus_bit,
    step_plane,
    step_zoom,
    toggle_bit,
)
from tests.support import make_image, planes_rgb


def test_open_image_clears_navigation_and_keeps_view_settings() -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("a.png"))
    other = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    state = open_image(
        ViewerState(value_format=DisplayFormat.BINARY, zoom=4),
        image,
    )
    state = set_cursor(state, PixelCoord(0, 0))
    state = select_only(state, "R", 3)
    opened = open_image(state, other)
    assert opened.cursor is None
    assert opened.selection is None
    assert opened.focus == BitChoice("R", 0)
    assert opened.value_format is DisplayFormat.BINARY
    assert opened.zoom == 4
    assert opened.image is other


def test_only_matched_is_idempotent_and_survives_opening_an_image() -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    state = set_only_matched(ViewerState(), on=True)
    assert state.only_matched
    assert set_only_matched(state, on=True) is state
    assert not set_only_matched(state, on=False).only_matched
    assert open_image(state, image).only_matched


def test_stepping_the_focus_stays_inside_the_plane() -> None:
    state = select_only(
        open_image(ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())),
        "R",
        0,
    )
    assert step_focus_bit(state, -1).focus == BitChoice("R", 0)
    top = step_focus_bit(select_only(state, "R", 7), 1)
    assert top.focus == BitChoice("R", 7)
    assert top.selection == frozenset({BitChoice("R", 7)})


def test_move_cursor_starts_at_the_origin_and_clamps() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    )
    state = move_cursor(state, 10, 10)
    assert state.cursor == PixelCoord(0, 0)
    state = move_cursor(state, 10, 10)
    assert state.cursor == PixelCoord(1, 1)
    state = move_cursor(state, -5, -5)
    assert state.cursor == PixelCoord(0, 0)


def test_extract_encoding_switches_and_survives_an_image_switch() -> None:
    image = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    state = open_image(ViewerState(), image)
    assert state.extract_encoding is ExtractEncoding.ASCII
    utf8 = set_extract_encoding(state, ExtractEncoding.UTF8)
    assert utf8.extract_encoding is ExtractEncoding.UTF8
    assert set_extract_encoding(utf8, ExtractEncoding.UTF8) is utf8
    assert open_image(utf8, image).extract_encoding is ExtractEncoding.UTF8


def test_stepping_planes_walks_the_display_order() -> None:
    image = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    state = open_image(ViewerState(), image)
    # A wide view enters the ladder at the head: the first step lands on R7.
    first = step_plane(state, 1)
    assert first.selection == frozenset({BitChoice("R", 7)})
    assert step_plane(first, 1).selection == frozenset({BitChoice("R", 6)})
    backwards = step_plane(first, -1)
    assert backwards.selection == frozenset({BitChoice("B", 0)})  # wrapped past the head
    assert step_plane(backwards, -1).selection == frozenset({BitChoice("B", 1)})
    assert step_plane(state, -1).selection == frozenset({BitChoice("B", 0)})
    # Down the ladder: R0 is followed by G7, and B0 wraps back to R7.
    r0 = select_only(state, "R", 0)
    assert step_plane(r0, 1).selection == frozenset({BitChoice("G", 7)})
    b0 = select_only(state, "B", 0)
    assert step_plane(b0, 1).selection == frozenset({BitChoice("R", 7)})
    empty = ViewerState()
    assert step_plane(empty, 1) is empty  # no image: nothing to step


def test_stepping_channels_shows_every_bit_of_one_channel() -> None:
    image = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    state = open_image(ViewerState(), image)
    red = step_channel(state, 1)
    assert red.selection == frozenset({BitChoice("R", bit) for bit in range(8)})
    assert red.focus == BitChoice("R", 0)
    green = step_channel(red, 1)
    assert green.selection == frozenset({BitChoice("G", bit) for bit in range(8)})
    blue = step_channel(green, 1)
    assert blue.selection == frozenset({BitChoice("B", bit) for bit in range(8)})
    assert step_channel(blue, 1).selection == red.selection  # wraps back to R
    assert step_channel(red, -1).selection == blue.selection
    assert step_channel(state, -1).selection == blue.selection
    empty = ViewerState()
    assert step_channel(empty, 1) is empty


def test_missing_plane_shortcut_leaves_the_state_alone() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    )
    assert select_lsb(state, "L") is state


def test_channel_and_column_group_switches() -> None:
    from pixelsb.domain.selection import all_bits
    from pixelsb.domain.transitions import set_channel, set_column

    image = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    state = open_image(ViewerState(), image)
    off = set_channel(state, "R", on=False)
    assert off.selection == all_bits(image) - {BitChoice("R", bit) for bit in range(8)}
    on = set_channel(off, "R", on=True)
    assert on.selection is None
    column = set_column(state, 0, on=False)
    assert column.selection == all_bits(image) - {
        BitChoice("R", 0),
        BitChoice("G", 0),
        BitChoice("B", 0),
    }


def test_filter_expression_survives_opening_another_image() -> None:
    from pixelsb.domain.transitions import set_filter_expr

    state = open_image(
        ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    )
    state = set_filter_expr(state, "B >= R and 0 <= top <= 4")
    assert state.filter_expr == "B >= R and 0 <= top <= 4"
    other = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb(), path=Path("c.png"))
    assert open_image(state, other).filter_expr == state.filter_expr
    assert set_filter_expr(state, "").filter_expr == ""


def test_zoom_and_format_cycle() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    )
    assert cycle_format(state).value_format is DisplayFormat.BINARY
    assert cycle_format(cycle_format(state)).value_format is DisplayFormat.DECIMAL
    assert step_focus_bit(select_only(state, "R", 0), -1).focus == BitChoice("R", 0)
    added = toggle_bit(select_only(state, "R", 0), "G", 0)
    assert added.selection == frozenset({BitChoice("R", 0), BitChoice("G", 0)})
    assert step_zoom(set_zoom(state, MAX_ZOOM), 1).zoom == MAX_ZOOM
    with pytest.raises(ValueError, match="outside"):
        set_zoom(state, 0)


def test_the_extract_order_survives_opening_another_image() -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("a.png"))
    other = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    state = set_channel_order(ViewerState(), ("B", "G", "R"))
    state = set_bit_order(state, BitOrder.LSB)
    state = set_scan_order(state, ScanOrder.YZ)
    expected = ExtractOrder(planes=("B", "G", "R"), bit_order=BitOrder.LSB, scan=ScanOrder.YZ)
    assert open_image(state, image).extract_order == expected
    assert open_image(state, other).extract_order == expected


def test_each_order_leaves_the_other_two_alone() -> None:
    state = set_channel_order(ViewerState(), ("B", "G", "R"))
    assert state.extract_order == ExtractOrder(planes=("B", "G", "R"))
    state = set_scan_order(state, ScanOrder.YZ)
    assert state.extract_order == ExtractOrder(planes=("B", "G", "R"), scan=ScanOrder.YZ)
    state = set_bit_order(state, BitOrder.LSB)
    assert state.extract_order == ExtractOrder(
        planes=("B", "G", "R"), bit_order=BitOrder.LSB, scan=ScanOrder.YZ
    )


def test_setting_an_order_that_is_already_in_force_changes_nothing() -> None:
    state = set_scan_order(ViewerState(), ScanOrder.YZ)
    assert set_scan_order(state, ScanOrder.YZ) is state
    assert set_bit_order(state, BitOrder.MSB) is state
    assert set_channel_order(state, ()) is state


def test_the_channel_order_refuses_a_repeated_channel() -> None:
    with pytest.raises(ValueError, match="repeats a plane name"):
        set_channel_order(ViewerState(), ("R", "R"))

from pathlib import Path

import numpy as np
import pytest

from pixelsb.domain import commands
from pixelsb.domain.models import (
    MAX_ZOOM,
    BitChoice,
    BitOrder,
    BitsMask,
    CropMask,
    DisplayFormat,
    ExtractEncoding,
    ExtractOrder,
    GrayscaleMask,
    InvertMask,
    LoadedImage,
    PixelCoord,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ScanOrder,
    ThresholdMask,
    ViewerState,
)
from pixelsb.domain.selection import all_bits
from pixelsb.domain.transitions import (
    add_layer,
    add_mask_text,
    bits_position,
    clear_layers,
    cycle_format,
    filter_position,
    move_cursor,
    move_layer,
    open_image,
    remove_layer,
    select_all_bits,
    select_lsb,
    select_lsbs,
    select_only,
    set_bit_order,
    set_channel,
    set_channel_order,
    set_column,
    set_cursor,
    set_extract_encoding,
    set_frame,
    set_layer_enabled,
    set_mask_text,
    set_scan_order,
    set_zoom,
    step_channel,
    step_focus_bit,
    step_plane,
    step_zoom,
    toggle_bit,
)
from tests.support import bits_of, make_image, mask_of, masks, planes_rgb

_PLANES = planes_rgb()


def _image(samples: list | None = None, path: str = "a.png") -> LoadedImage:
    values = np.zeros((2, 2, 3), dtype=np.uint16) if samples is None else np.array(samples)
    return make_image(np.asarray(values, dtype=np.uint16), _PLANES, path=Path(path))


def _open(image: LoadedImage | None = None) -> ViewerState:
    return open_image(ViewerState(), image if image is not None else _image())


def test_open_image_clears_navigation_and_keeps_view_settings() -> None:
    other = _image(path="b.png")
    state = open_image(ViewerState(value_format=DisplayFormat.BINARY, zoom=4), _image())
    state = set_cursor(state, PixelCoord(0, 0))
    state = select_only(state, "R", 3)
    opened = open_image(state, other)
    assert opened.cursor is None
    assert opened.focus == BitChoice("R", 0)
    assert opened.value_format is DisplayFormat.BINARY
    assert opened.zoom == 4
    assert opened.image is other


def test_open_image_carries_the_mask_stack_over() -> None:
    state = add_layer(_open(), RegionMask("R > 0"))
    assert masks(open_image(state, _image(path="b.png"))) == (RegionMask("R > 0"),)


def test_the_stack_runs_bottom_to_top_and_can_be_reordered() -> None:
    state = add_layer(add_layer(_open(), RegionMask("R > 0")), InvertMask())
    assert masks(state) == (RegionMask("R > 0"), InvertMask())
    raised = move_layer(state, 0, 1)
    assert masks(raised) == (InvertMask(), RegionMask("R > 0"))
    assert masks(move_layer(raised, 1, 1)) == masks(raised)  # already at the top
    assert masks(move_layer(raised, 0, -1)) == masks(raised)
    assert masks(move_layer(state, 0, -1)) == (RegionMask("R > 0"), InvertMask())


def test_layers_can_be_switched_off_and_taken_out() -> None:
    state = add_layer(_open(), InvertMask())
    assert masks(state) == (InvertMask(),)
    off = set_layer_enabled(state, 0, on=False)
    assert not off.layers[0].enabled
    assert set_layer_enabled(off, 0, on=False) is off  # already off
    assert set_layer_enabled(off, 0, on=True).layers[0].enabled
    assert masks(remove_layer(state, 0)) == ()
    assert clear_layers(state).layers == ()


def test_clearing_an_empty_stack_changes_nothing() -> None:
    state = _open()
    assert clear_layers(state) is state


def test_a_layer_index_outside_the_stack_is_refused() -> None:
    state = add_layer(_open(), InvertMask())
    for call in (
        lambda: remove_layer(state, 1),
        lambda: move_layer(state, 1, 1),
        lambda: set_layer_enabled(state, -1, on=False),
    ):
        with pytest.raises(ValueError, match="outside"):
            call()


def test_the_filter_box_binds_the_layer_whose_mask_has_a_command_line() -> None:
    state = add_layer(add_layer(_open(), RegionMask("R > 0")), RegionMask("G > 0"))
    assert filter_position(state.layers, _PLANES, 1) == 1
    assert filter_position(state.layers, _PLANES, 0) == 0
    assert filter_position(state.layers, _PLANES) is None  # no layer in hand: nothing to edit
    others = add_layer(add_layer(_open(), InvertMask()), RegionMask("R > 0"))
    assert filter_position(others.layers, _PLANES, 0) == 0  # the invert has its own command
    assert filter_position(others.layers, _PLANES, 9) is None  # nor is a layer that is gone
    emptied = add_layer(_open(), BitsMask(frozenset()))
    assert filter_position(emptied.layers, _PLANES, 0) is None  # no bits, no line


def test_writing_a_filter_text_adds_a_mask_on_top() -> None:
    state = set_mask_text(_open(), "B >= R")
    assert masks(state) == (RegionMask("B >= R"),)
    assert masks(set_mask_text(state, "G >= R")) == (
        RegionMask("B >= R"),
        RegionMask("G >= R"),
    )  # no layer named: a new mask on top
    assert masks(set_mask_text(state, "G >= R", layer=0)) == (
        RegionMask("G >= R"),
    )  # named: rewritten


def test_a_command_line_becomes_its_mask() -> None:
    assert masks(set_mask_text(_open(), "b.0")) == (BitsMask(frozenset({BitChoice("B", 0)})),)
    bound = add_layer(_open(), InvertMask())
    assert masks(set_mask_text(bound, "thr 200", layer=0)) == (ThresholdMask(200),)


def test_a_line_names_one_operation_of_one_kind() -> None:
    """Two kinds in one line would be two operations, and the line names one."""
    state = add_layer(_open(), BitsMask(frozenset({BitChoice("R", 3)})))
    with pytest.raises(commands.CommandError, match="一行只写一个操作"):
        set_mask_text(state, "b > r and b.0", layer=0)
    with pytest.raises(commands.CommandError, match="单独一行"):
        set_mask_text(state, "thr 128 and b.0", layer=0)
    assert masks(state) == (BitsMask(frozenset({BitChoice("R", 3)})),)  # the stack stands


def test_a_line_rewrites_the_operation_in_hand_whatever_kind_it_was() -> None:
    """Picking an operation and writing a command edits that operation, kind and all."""
    region = add_layer(_open(), RegionMask("R > 0"))
    assert masks(set_mask_text(region, "b.0", layer=0)) == (
        BitsMask(frozenset({BitChoice("B", 0)})),
    )  # the line in the box replaces what the box was showing
    bits = add_layer(region, BitsMask(frozenset({BitChoice("R", 3)})))
    assert masks(set_mask_text(bits, "b.0", layer=1)) == (
        RegionMask("R > 0"),
        BitsMask(frozenset({BitChoice("B", 0)})),
    )  # the layer in hand is the one rewritten
    assert masks(set_mask_text(bits, "b > r", layer=0)) == (
        RegionMask("b > r"),
        BitsMask(frozenset({BitChoice("R", 3)})),
    )
    assert masks(set_mask_text(bits, "thr 64", layer=0)) == (
        ThresholdMask(64),
        BitsMask(frozenset({BitChoice("R", 3)})),
    )
    pair = add_layer(add_layer(_open(), InvertMask()), ThresholdMask(64))
    assert masks(set_mask_text(pair, "gray", layer=1)) == (InvertMask(), GrayscaleMask())
    assert masks(set_mask_text(pair, "gray", layer=0)) == (GrayscaleMask(), ThresholdMask(64))


def test_adding_a_line_stacks_it_instead_of_rewriting_the_layer_in_hand() -> None:
    """Shift+Enter's line: the operation in hand keeps its command and its place."""
    state = add_layer(_open(), ThresholdMask(200))
    assert masks(add_mask_text(state, "thr 128")) == (ThresholdMask(200), ThresholdMask(128))
    bits = add_layer(state, BitsMask(frozenset({BitChoice("R", 3)})))
    assert masks(add_mask_text(bits, "b.0")) == (
        ThresholdMask(200),
        BitsMask(frozenset({BitChoice("R", 3)})),
        BitsMask(frozenset({BitChoice("B", 0)})),
    )  # a second selection layer, the one below untouched
    assert masks(add_mask_text(bits, "b > r")) == (
        ThresholdMask(200),
        BitsMask(frozenset({BitChoice("R", 3)})),
        RegionMask("b > r"),
    )


def test_adding_nothing_adds_nothing() -> None:
    state = add_layer(_open(), InvertMask())
    assert add_mask_text(state, "") is state
    assert add_mask_text(state, "b>r and b.") is state  # still being typed
    with pytest.raises(commands.CommandError):
        add_mask_text(state, "thr 1x")
    assert masks(state) == (InvertMask(),)


def test_a_half_typed_line_changes_nothing() -> None:
    """No prefix of a command may touch the stack: ``b>r and b.`` is not a filter."""
    state = add_layer(_open(), RegionMask("R > 0"))
    assert set_mask_text(state, "b>r and b.", layer=0) is state
    assert set_mask_text(state, "R >= ", layer=0) is state


def test_writing_an_empty_text_drops_the_mask_it_edits() -> None:
    state = add_layer(add_layer(_open(), InvertMask()), RegionMask("R > 0"))
    assert masks(set_mask_text(state, "  ", layer=1)) == (InvertMask(),)
    empty = _open()
    assert set_mask_text(empty, "") is empty  # nothing to drop, nothing happens


def test_a_command_the_box_cannot_read_changes_nothing() -> None:
    state = add_layer(_open(), InvertMask())
    with pytest.raises(commands.CommandError):
        set_mask_text(state, "thr 1x", layer=0)
    assert masks(state) == (InvertMask(),)


def test_a_region_mask_can_be_edited_by_its_index_or_by_what_is_selected() -> None:
    state = add_layer(add_layer(_open(), RegionMask("R > 0")), RegionMask("G > 0"))
    assert masks(set_mask_text(state, "B > 0", layer=0)) == (
        RegionMask("B > 0"),
        RegionMask("G > 0"),
    )


def test_stepping_the_focus_stays_inside_the_plane() -> None:
    state = select_only(_open(), "R", 0)
    assert step_focus_bit(state, -1).focus == BitChoice("R", 0)
    top = step_focus_bit(select_only(state, "R", 7), 1)
    assert top.focus == BitChoice("R", 7)
    assert bits_of(top) == frozenset({BitChoice("R", 7)})


def test_move_cursor_starts_at_the_origin_and_clamps() -> None:
    state = move_cursor(_open(), 10, 10)
    assert state.cursor == PixelCoord(0, 0)
    state = move_cursor(state, 10, 10)
    assert state.cursor == PixelCoord(1, 1)
    state = move_cursor(state, -5, -5)
    assert state.cursor == PixelCoord(0, 0)


def test_extract_encoding_switches_and_survives_an_image_switch() -> None:
    state = _open()
    assert state.extract_encoding is ExtractEncoding.ASCII
    utf8 = set_extract_encoding(state, ExtractEncoding.UTF8)
    assert utf8.extract_encoding is ExtractEncoding.UTF8
    assert set_extract_encoding(utf8, ExtractEncoding.UTF8) is utf8
    assert open_image(utf8, _image()).extract_encoding is ExtractEncoding.UTF8


def test_stepping_planes_walks_the_display_order() -> None:
    state = _open()
    # A wide view enters the ladder at the head: the first step lands on R7.
    first = step_plane(state, 1)
    assert bits_of(first) == frozenset({BitChoice("R", 7)})
    assert bits_of(step_plane(first, 1)) == frozenset({BitChoice("R", 6)})
    backwards = step_plane(first, -1)
    assert bits_of(backwards) == frozenset({BitChoice("B", 0)})  # wrapped past the head
    assert bits_of(step_plane(backwards, -1)) == frozenset({BitChoice("B", 1)})
    assert bits_of(step_plane(state, -1)) == frozenset({BitChoice("B", 0)})
    # Down the ladder: R0 is followed by G7, and B0 wraps back to R7.
    assert bits_of(step_plane(select_only(state, "R", 0), 1)) == frozenset({BitChoice("G", 7)})
    assert bits_of(step_plane(select_only(state, "B", 0), 1)) == frozenset({BitChoice("R", 7)})
    empty = ViewerState()
    assert step_plane(empty, 1) is empty  # no image: nothing to step


def test_a_bits_edit_falls_back_when_the_named_layer_is_not_a_bits_mask() -> None:
    state = add_layer(add_layer(_open(), InvertMask()), BitsMask(frozenset({BitChoice("R", 0)})))
    assert bits_position(state.layers, 0) == 1
    assert bits_position(state.layers, 1) == 1
    assert bits_position(state.layers, 9) == 1
    assert bits_position(()) is None
    stepped = step_plane(state, 1, layer=0)  # layer 0 is the invert mask
    assert mask_of(stepped, BitsMask).selection == frozenset({BitChoice("G", 7)})
    assert len(masks(stepped)) == 2


def test_stepping_planes_rewrites_the_mask_it_was_pointed_at() -> None:
    state = add_layer(_open(), BitsMask(frozenset({BitChoice("R", 0)})))
    stepped = step_plane(state, 1, layer=0)
    assert len(masks(stepped)) == 1
    assert mask_of(stepped, BitsMask).selection == frozenset({BitChoice("G", 7)})


def test_stepping_planes_survives_a_mask_that_names_another_images_channel() -> None:
    """A mask kept across an image change can show a plane the new one does not have."""
    deep = tuple(SamplePlane(plane.name, plane.index, 16, plane.origin) for plane in planes_rgb())
    state = open_image(ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), deep))
    carried = open_image(select_only(state, "R", 12), _image(path="b.png"))
    assert mask_of(carried, BitsMask).selection == frozenset({BitChoice("R", 12)})
    stepped = step_plane(carried, 1)  # R12 is no plane of the 8-bit image
    assert bits_of(stepped) == frozenset({BitChoice("R", 7)})  # the ladder head
    assert bits_of(step_plane(carried, -1)) == frozenset({BitChoice("B", 0)})
    assert len(masks(stepped)) == 1  # the mask is rewritten, not duplicated


def test_stepping_channels_shows_every_bit_of_one_channel() -> None:
    state = _open()
    red = step_channel(state, 1)
    assert bits_of(red) == frozenset({BitChoice("R", bit) for bit in range(8)})
    assert red.focus == BitChoice("R", 0)
    green = step_channel(red, 1)
    assert bits_of(green) == frozenset({BitChoice("G", bit) for bit in range(8)})
    blue = step_channel(green, 1)
    assert bits_of(blue) == frozenset({BitChoice("B", bit) for bit in range(8)})
    assert bits_of(step_channel(blue, 1)) == bits_of(red)  # wraps back to R
    assert bits_of(step_channel(red, -1)) == bits_of(blue)
    assert bits_of(step_channel(state, -1)) == bits_of(blue)
    empty = ViewerState()
    assert step_channel(empty, 1) is empty


def test_missing_plane_shortcut_leaves_the_state_alone() -> None:
    state = _open()
    assert select_lsb(state, "L") is state


def test_the_first_bit_edit_adds_a_mask_holding_every_other_bit() -> None:
    """A shortcut narrows one bit off the whole image instead of starting from LSBs."""
    state = toggle_bit(_open(), "R", 7)
    image = state.image
    assert image is not None
    assert bits_of(state) == all_bits(image.planes) - {BitChoice("R", 7)}
    assert len(masks(state)) == 1


def test_a_bit_edit_rewrites_the_topmost_bits_mask() -> None:
    state = add_layer(add_layer(_open(), BitsMask(frozenset({BitChoice("R", 0)}))), InvertMask())
    assert mask_of(toggle_bit(state, "R", 3), BitsMask).selection == frozenset(
        {BitChoice("R", 0), BitChoice("R", 3)}
    )
    assert bits_of(select_only(state, "G", 2)) == frozenset({BitChoice("G", 2)})
    assert len(masks(select_only(state, "G", 2))) == 2


def test_the_bits_presets_replace_the_selection_whole() -> None:
    state = select_only(_open(), "R", 3)
    assert bits_of(select_lsbs(state)) == frozenset(
        {BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)}
    )
    image = state.image
    assert image is not None
    assert bits_of(select_all_bits(state)) == all_bits(image.planes)


def test_select_only_rejects_a_bit_past_the_plane() -> None:
    with pytest.raises(ValueError, match=r"outside 0\.\.7"):
        select_only(_open(), "R", 8)


def test_channel_and_column_group_switches() -> None:
    state = _open()
    image = state.image
    assert image is not None
    off = set_channel(state, "R", on=False)
    assert bits_of(off) == all_bits(image.planes) - {BitChoice("R", bit) for bit in range(8)}
    assert bits_of(set_channel(off, "R", on=True)) == all_bits(image.planes)
    column = set_column(state, 0, on=False)
    assert bits_of(column) == all_bits(image.planes) - {
        BitChoice("R", 0),
        BitChoice("G", 0),
        BitChoice("B", 0),
    }


def test_a_group_switch_only_touches_the_mask_it_was_pointed_at() -> None:
    state = add_layer(add_layer(_open(), InvertMask()), BitsMask(frozenset({BitChoice("R", 0)})))
    switched = set_column(state, 0, on=True, layer=1)
    assert mask_of(switched, BitsMask).selection == frozenset(
        {BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)}
    )


def test_the_crop_mask_is_part_of_the_stack() -> None:
    state = add_layer(_open(), CropMask())
    assert masks(state) == (CropMask(),)
    assert masks(remove_layer(state, 0)) == ()


def test_zoom_and_format_cycle() -> None:
    state = _open()
    assert cycle_format(state).value_format is DisplayFormat.BINARY
    assert cycle_format(cycle_format(state)).value_format is DisplayFormat.DECIMAL
    assert step_zoom(set_zoom(state, MAX_ZOOM), 1).zoom == MAX_ZOOM
    with pytest.raises(ValueError, match="outside"):
        set_zoom(state, 0)


def test_the_extract_order_survives_opening_another_image() -> None:
    state = set_channel_order(ViewerState(), ("B", "G", "R"))
    state = set_bit_order(state, BitOrder.LSB)
    state = set_scan_order(state, ScanOrder.YZ)
    expected = ExtractOrder(planes=("B", "G", "R"), bit_order=BitOrder.LSB, scan=ScanOrder.YZ)
    assert open_image(state, _image()).extract_order == expected
    assert open_image(state, _image(path="b.png")).extract_order == expected


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


def test_set_frame_keeps_the_stack_the_selection_and_the_cursor() -> None:
    image = make_image(np.full((2, 2, 3), 7, dtype=np.uint16), _PLANES, path=Path("a.png"))
    other = make_image(np.full((2, 2, 3), 9, dtype=np.uint16), _PLANES, path=Path("a.png"))
    state = set_cursor(select_only(open_image(ViewerState(), image), "G", 2), PixelCoord(1, 1))
    frame = set_frame(state, other)
    assert frame.image is other
    assert bits_of(frame) == frozenset({BitChoice("G", 2)})
    assert frame.cursor == PixelCoord(1, 1)
    assert frame.focus == BitChoice("G", 2)
    assert masks(frame) == masks(state)


def test_set_frame_drops_a_focus_the_new_frame_does_not_offer() -> None:
    """A 16-bit frame view over an 8-bit one keeps a mask that has nothing to select."""
    wide = make_image(
        np.zeros((2, 2, 1), dtype=np.uint16),
        (SamplePlane("L", 0, 16, SampleOrigin.RAW),),
    )
    narrow = make_image(
        np.zeros((2, 2, 1), dtype=np.uint16),
        (SamplePlane("L", 0, 8, SampleOrigin.RAW),),
    )
    state = select_only(open_image(ViewerState(), wide), "L", 12)
    frame = set_frame(state, narrow)
    assert frame.focus == BitChoice("L", 0)
    assert bits_of(frame) == frozenset({BitChoice("L", 12)})  # the mask keeps its own bits


def test_set_frame_without_an_open_image_opens_it() -> None:
    state = set_frame(ViewerState(), _image())
    assert state.image is not None
    assert state.focus == BitChoice("R", 0)

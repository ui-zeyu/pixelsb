import numpy as np
import pytest

from pixelsb.domain.labels import (
    font_pixel_size,
    label_fits,
    pixel_text,
    region_texts,
    widest_text,
    zoom_required,
)
from pixelsb.domain.models import BitChoice, DisplayFormat, PixelCoord, ViewerState
from pixelsb.domain.samples import render_rgb
from pixelsb.domain.transitions import (
    open_image,
    select_all_bits,
    select_lsbs,
    select_only,
    set_cursor,
    set_format,
    toggle_bit,
)
from tests.support import make_image, planes_rgb


def _rgb_state() -> ViewerState:
    samples = np.array([[[0b00001001, 0b00000001, 0]]], dtype=np.uint16)
    return open_image(ViewerState(), make_image(samples, planes_rgb()))


def test_one_bit_is_zero_or_one_and_renders_black_or_white() -> None:
    state = select_only(_rgb_state(), "R", 0)
    image = state.image
    assert image is not None
    rendered = render_rgb(image, state.selection)
    assert int(rendered[0, 0, 0]) == 255
    high = select_only(state, "R", 3)
    assert high.image is not None
    assert int(render_rgb(high.image, high.selection)[0, 0, 0]) == 255
    off = select_only(state, "R", 1)
    assert off.image is not None
    assert int(render_rgb(off.image, off.selection)[0, 0, 0]) == 0


def test_mask_keeps_original_weights_for_several_bits() -> None:
    state = select_only(_rgb_state(), "R", 0)
    state = toggle_bit(state, "R", 3)
    image = state.image
    assert image is not None
    assert state.selection == frozenset({BitChoice("R", 0), BitChoice("R", 3)})
    rendered = render_rgb(image, state.selection)
    assert int(rendered[0, 0, 0]) == 255
    assert int(rendered[0, 0, 1]) == 0


def test_presets_cover_every_lsb_and_the_original() -> None:
    state = select_lsbs(_rgb_state())
    assert state.selection == frozenset({BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)})
    restored = select_all_bits(state)
    assert restored.selection is None
    assert restored.image is not None
    assert int(render_rgb(restored.image, restored.selection).sum()) > 0


def test_select_only_rejects_a_bit_past_the_plane() -> None:
    with pytest.raises(ValueError):
        select_only(_rgb_state(), "R", 8)


def test_unchecking_a_bit_from_the_original_materializes_the_selection() -> None:
    from pixelsb.domain.selection import all_bits

    state = toggle_bit(_rgb_state(), "R", 7)
    image = state.image
    assert image is not None
    expected = all_bits(image) - {BitChoice("R", 7)}
    assert state.selection == expected
    restored = toggle_bit(state, "R", 7)
    assert restored.selection is None


def test_numbers_follow_the_readout_layer() -> None:
    from pixelsb.domain.transitions import (
        select_only_readout,
        set_cursor,
        set_detached,
        toggle_readout_bit,
    )

    state = set_detached(set_cursor(select_only(_rgb_state(), "B", 0), PixelCoord(0, 0)), True)
    assert pixel_text(state, PixelCoord(0, 0)) == "R:09\nG:01\nB:00"
    assert widest_text(state) == "R:FF\nG:FF\nB:FF"
    numbers = toggle_readout_bit(state, "R", 0)
    assert pixel_text(numbers, PixelCoord(0, 0)) == "R:08\nG:01\nB:00"
    assert widest_text(numbers) == "R:FE\nG:FF\nB:FF"
    both = toggle_readout_bit(numbers, "R", 3)
    assert pixel_text(both, PixelCoord(0, 0)) == "R:00\nG:01\nB:00"
    binary = set_format(both, DisplayFormat.BINARY)
    assert pixel_text(binary, PixelCoord(0, 0)) == "R:00000000\nG:00000001\nB:00000000"
    single = select_only_readout(both, "B", 0)
    assert pixel_text(single, PixelCoord(0, 0)) == "0"
    attached = set_detached(single, False)
    assert pixel_text(attached, PixelCoord(0, 0)) == "0"


def test_labels_use_the_cell_bigness_instead_of_tiny_fonts() -> None:
    assert font_pixel_size(20, "FF") == 13
    assert font_pixel_size(128, "1") == 92
    assert zoom_required("1") == 9
    assert zoom_required("FF") == 9
    assert label_fits(9, "1")
    assert not label_fits(8, "1")
    assert label_fits(9, "FF")
    assert not label_fits(8, "FF")


def test_channel_lines_stack_so_the_font_stays_big() -> None:
    rgb_hex = "R FF\nG FF\nB FF"
    assert zoom_required(rgb_hex) == 21
    assert font_pixel_size(63, rgb_hex) == 16
    assert label_fits(21, rgb_hex)
    assert not label_fits(20, rgb_hex)
    two_lines = "R1\nG0"
    assert zoom_required(two_lines) == 15
    assert label_fits(15, two_lines)


def test_region_texts_matches_pixel_text_and_is_clipped() -> None:
    state = set_cursor(toggle_bit(select_only(_rgb_state(), "R", 0), "R", 3), PixelCoord(0, 0))
    image = state.image
    assert image is not None
    labels = region_texts(state, 0, 0, image.width, image.height)
    assert labels == [
        pixel_text(state, PixelCoord(x, y)) for y in range(image.height) for x in range(image.width)
    ]
    assert region_texts(state, 5, 5, 9, 9) == []


def test_region_texts_formats_offsets_against_the_anchor() -> None:
    from pixelsb.domain.models import ValueMode
    from pixelsb.domain.transitions import (
        select_only_readout,
        set_anchor,
        set_detached,
        set_value_mode,
    )

    samples = np.array([[[0b1001, 0, 0], [0b0000, 0, 0]]], dtype=np.uint16)
    image = make_image(samples, planes_rgb())
    state = set_cursor(select_only(open_image(ViewerState(), image), "R", 0), PixelCoord(1, 0))
    state = set_detached(
        set_value_mode(set_anchor(state, PixelCoord(0, 0)), ValueMode.OFFSET), True
    )
    assert pixel_text(state, PixelCoord(0, 0)) == "R:0\nG:0\nB:0"
    assert pixel_text(state, PixelCoord(1, 0)) == "R:-09\nG:0\nB:0"
    narrowed = select_only_readout(state, "R", 0)
    assert pixel_text(narrowed, PixelCoord(1, 0)) == "-1"

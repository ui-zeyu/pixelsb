"""Property tests: the compiler, the stack, and the renderers against naive references."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from pixelsb.domain.extract import extract_bytes
from pixelsb.domain.geometry import SLIDER_STEPS, slider_position, slider_zoom
from pixelsb.domain.models import (
    MAX_ZOOM,
    BitChoice,
    BitOrder,
    BitsMask,
    CropMask,
    ExtractOrder,
    LoadedImage,
    Raster,
    ScanOrder,
)
from pixelsb.domain.predicate import PredicateError, compile_filter
from pixelsb.domain.samples import render_raster
from pixelsb.domain.selection import all_bits
from pixelsb.domain.stack import apply_mask, match_span
from tests.support import make_image, planes_rgb, raster

_BITS = st.sets(st.integers(min_value=0, max_value=7), min_size=1, max_size=8)
_COORDS = st.integers(min_value=0, max_value=8)
_STEPS = st.integers(min_value=1, max_value=6)
_SIZES = st.integers(min_value=1, max_value=6)
_PATTERNS = st.integers(min_value=0, max_value=0xFFFF)


def _image(width: int, height: int, seed: int = 0) -> LoadedImage:
    rng = np.random.default_rng(seed)
    return make_image(rng.integers(0, 256, size=(height, width, 3)).astype(np.uint16), planes_rgb())


def _bits_of(image: LoadedImage, bits: set[int]) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane.name, bit) for plane in image.planes for bit in bits)


def _selected(image: LoadedImage, bits: set[int]) -> Raster:
    """The raster that image makes with those bits of every channel selected."""
    return raster(image, BitsMask(_bits_of(image, bits)))


def _whole_channel(image: LoadedImage, name: str) -> Raster:
    """The raster that image makes with all eight bits of one channel selected."""
    return raster(image, BitsMask(frozenset(BitChoice(name, bit) for bit in range(8))))


def _select_in(raster: Raster, bits: set[int]) -> Raster:
    """That raster with those bits of every channel selected instead."""
    chosen = frozenset(BitChoice(plane.name, bit) for plane in raster.planes for bit in bits)
    return apply_mask(raster, BitsMask(chosen))


def _marked(image: LoadedImage, live: NDArray[np.bool_]) -> Raster:
    """A raster of that image with those pixels standing as the ones that survived."""
    return Raster(
        samples=image.samples,
        planes=image.planes,
        selection=all_bits(image.planes),
        live=live,
    )


def _pattern(side: int, pattern: int) -> NDArray[np.bool_]:
    """The 4x4 (or ``side``-square) bit pattern an integer spells out."""
    return np.array(
        [
            [bool((pattern | 1) >> (row * side + column) & 1) for column in range(side)]
            for row in range(side)
        ]
    )


def _mask(expression: str, image: LoadedImage) -> NDArray[np.bool_]:
    return compile_filter(expression, image.planes).evaluate(raster(image))


@given(width=_SIZES, height=_SIZES, x=_COORDS, y=_COORDS, step_x=_STEPS, step_y=_STEPS)
def test_grid_is_the_lattice_it_describes(
    width: int, height: int, x: int, y: int, step_x: int, step_y: int
) -> None:
    image = _image(width, height)
    expected = np.array(
        [
            [
                column >= x and row >= y and (column - x) % step_x == 0 and (row - y) % step_y == 0
                for column in range(width)
            ]
            for row in range(height)
        ]
    )
    assert np.array_equal(_mask(f"grid({x}, {y}, {step_x}, {step_y})", image), expected)


@given(width=_SIZES, height=_SIZES, x0=_COORDS, y0=_COORDS, x1=_COORDS, y1=_COORDS)
def test_rect_is_the_box_its_bounds_describe(
    width: int, height: int, x0: int, y0: int, x1: int, y1: int
) -> None:
    image = _image(width, height)
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    expected = np.array(
        [
            [left <= column < right and top <= row < bottom for column in range(width)]
            for row in range(height)
        ]
    )
    assert np.array_equal(_mask(f"rect({x0}, {y0}, {x1}, {y1})", image), expected)


@given(x=st.integers(min_value=-(2**31), max_value=-1), y=_COORDS)
def test_grid_rejects_a_negative_anchor(x: int, y: int) -> None:
    with pytest.raises(PredicateError, match="不能为负"):
        _mask(f"grid({x}, {y}, 2, 2)", _image(4, 4))


@given(step_x=st.integers(min_value=-(2**31), max_value=0), step_y=_STEPS)
def test_grid_rejects_a_step_below_one(step_x: int, step_y: int) -> None:
    with pytest.raises(PredicateError, match="至少为 1"):
        _mask(f"grid(0, 0, {step_x}, {step_y})", _image(4, 4))


@given(huge=st.one_of(st.integers(min_value=2**63), st.integers(max_value=-(2**63) - 1)))
def test_a_literal_no_field_could_hold_is_rejected(huge: int) -> None:
    """However odd the expression, it fails as a filter error and nothing else."""
    with pytest.raises(PredicateError, match="整数超出范围"):
        _mask(f"left >= {huge}", _image(4, 4))


@given(bit=st.integers(min_value=0, max_value=7))
def test_a_bit_field_is_that_bit_of_the_stored_channel(bit: int) -> None:
    """A channel's bit index reads the stored sample's bit, whatever it holds."""
    image = _image(4, 3, seed=7)
    expected = ((image.samples[:, :, 0] >> np.uint16(bit)) & np.uint16(1)) == 1
    assert np.array_equal(_mask(f"R.{bit} == 1", image), expected)


@given(bits=_BITS)
def test_the_render_is_the_divide_reference(bits: set[int]) -> None:
    """Whatever the selected bits, scaling them to bytes goes by the exact field."""
    image = _image(5, 4, seed=11)
    field = sum(1 << bit for bit in bits)
    expected = np.stack(
        [
            ((image.samples[:, :, index] & field).astype(np.uint32) * 255 // field).astype(np.uint8)
            for index in range(3)
        ],
        axis=-1,
    )
    assert np.array_equal(render_raster(_selected(image, bits)), expected)


@given(bits=_BITS)
def test_the_extract_stream_packs_the_selected_bits_msb_first(bits: set[int]) -> None:
    image = _image(3, 2, seed=5)
    stream = [
        (int(image.samples[row, column, plane.index]) >> bit) & 1
        for row in range(image.height)
        for column in range(image.width)
        for plane in image.planes
        for bit in range(8)
        if bit in bits
    ]
    assert extract_bytes(_selected(image, bits)) == np.packbits(stream).tobytes()


@given(pattern=_PATTERNS)
def test_cropping_crops_to_the_rows_and_columns_its_pixels_touch(pattern: int) -> None:
    side = 4
    live = _pattern(side, pattern)  # its first bit is always set, so something survives
    cropped = apply_mask(_marked(_image(side, side), live), CropMask())
    span = match_span(live)
    assert span is not None
    rows, columns = span
    assert cropped.rows is not None
    assert cropped.columns is not None
    assert cropped.rows.tolist() == rows.tolist()
    assert cropped.columns.tolist() == columns.tolist()
    mask = cropped.live
    assert mask is not None
    for dy in range(cropped.height):
        for dx in range(cropped.width):
            coord = cropped.source_at(dx, dy)
            # Every cell is its own source pixel; which ones take part is `live`.
            assert cropped.cell_of(coord) == (dx, dy)
            assert bool(mask[dy, dx]) == bool(live[coord.y, coord.x])


@given(zoom=st.integers(min_value=1, max_value=int(MAX_ZOOM)))
def test_the_slider_round_trips_every_whole_zoom(zoom: int) -> None:
    position = slider_position(zoom)
    assert 0 <= position <= SLIDER_STEPS
    assert slider_zoom(position) == zoom


def _transposed(image: LoadedImage) -> LoadedImage:
    return make_image(image.samples.transpose(1, 0, 2), planes_rgb())


@given(width=_SIZES, height=_SIZES, bits=_BITS, seed=st.integers(0, 5))
def test_the_column_order_is_the_transposed_image_read_by_rows(
    width: int, height: int, bits: set[int], seed: int
) -> None:
    image = _image(width, height, seed=seed)
    columns = ExtractOrder(scan=ScanOrder.YZ)
    assert extract_bytes(_selected(image, bits), columns) == extract_bytes(
        _selected(_transposed(image), bits)
    )


@given(pattern=_PATTERNS, bits=_BITS)
def test_the_column_order_carries_the_live_mask_along(pattern: int, bits: set[int]) -> None:
    side = 4
    image = _image(side, side, seed=7)
    live = _pattern(side, pattern)
    columns = ExtractOrder(scan=ScanOrder.YZ)
    assert extract_bytes(_select_in(_marked(image, live), bits), columns) == extract_bytes(
        _select_in(_marked(_transposed(image), live.T), bits)
    )


@given(width=_SIZES, height=_SIZES, seed=st.integers(0, 5))
def test_low_first_returns_the_stored_bytes_of_a_whole_channel(
    width: int, height: int, seed: int
) -> None:
    image = _image(width, height, seed=seed)
    whole_red = _whole_channel(image, "R")
    stored = image.samples[:, :, 0].astype(np.uint8).tobytes()
    low_first = ExtractOrder(bit_order=BitOrder.LSB)
    assert extract_bytes(whole_red, low_first) == stored
    # High first packs bit 0 into the top of the byte, which mirrors it.
    mirrored = bytes(int(f"{byte:08b}"[::-1], 2) for byte in stored)
    assert extract_bytes(whole_red) == mirrored


@given(width=_SIZES, height=_SIZES, seed=st.integers(0, 5))
def test_a_channel_order_permutes_the_byte_groups_of_every_pixel(
    width: int, height: int, seed: int
) -> None:
    every_channel = _selected(_image(width, height, seed=seed), set(range(8)))
    every = extract_bytes(every_channel)
    swapped = extract_bytes(every_channel, ExtractOrder(planes=("B", "G", "R")))
    groups = [every[start : start + 3] for start in range(0, len(every), 3)]
    assert swapped == b"".join(group[::-1] for group in groups)

"""Property tests: the compiler and the renderers against naive references."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pixelsb.domain.extract import extract_bytes
from pixelsb.domain.geometry import SLIDER_STEPS, slider_position, slider_zoom
from pixelsb.domain.match_view import match_span, match_view
from pixelsb.domain.models import (
    MAX_ZOOM,
    BitChoice,
    BitOrder,
    ExtractOrder,
    LoadedImage,
    ScanOrder,
)
from pixelsb.domain.predicate import PredicateError, compile_filter
from pixelsb.domain.samples import render_rgb
from tests.support import make_image, planes_rgb

_BITS = st.sets(st.integers(min_value=0, max_value=7), min_size=1, max_size=8)
_COORDS = st.integers(min_value=0, max_value=8)
_STEPS = st.integers(min_value=1, max_value=6)
_SIZES = st.integers(min_value=1, max_value=6)


def _image(width: int, height: int, seed: int = 0) -> LoadedImage:
    rng = np.random.default_rng(seed)
    return make_image(rng.integers(0, 256, size=(height, width, 3)).astype(np.uint16), planes_rgb())


def _mask(expression: str, image: LoadedImage) -> np.ndarray:
    return compile_filter(expression, image.planes).evaluate(image, None)


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


@given(bits=_BITS)
def test_the_render_is_the_divide_reference(bits: set[int]) -> None:
    """Whatever the selected bits, scaling them to bytes goes by the exact field."""
    image = _image(5, 4, seed=11)
    chosen = frozenset(BitChoice(name, bit) for name in ("R", "G", "B") for bit in bits)
    field = sum(1 << bit for bit in bits)
    expected = np.stack(
        [
            ((image.samples[:, :, index] & field).astype(np.uint32) * 255 // field).astype(np.uint8)
            for index in range(3)
        ],
        axis=-1,
    )
    assert np.array_equal(render_rgb(image, chosen), expected)


@given(bits=_BITS)
def test_the_extract_stream_packs_the_selected_bits_msb_first(bits: set[int]) -> None:
    image = _image(3, 2, seed=5)
    chosen = frozenset(BitChoice(name, bit) for name in ("R", "G", "B") for bit in bits)
    stream = [
        (int(image.samples[row, column, plane.index]) >> bit) & 1
        for row in range(image.height)
        for column in range(image.width)
        for plane in image.planes
        for bit in range(8)
        if bit in bits
    ]
    assert extract_bytes(image, chosen) == np.packbits(stream).tobytes()


@given(pattern=st.integers(min_value=0, max_value=0xFFFF))
def test_a_compacted_view_is_the_span_of_the_match(pattern: int) -> None:
    side = 4
    # Bit zero is always set, so the view always has something to show.
    match = np.array(
        [
            [bool((pattern | 1) >> (row * side + column) & 1) for column in range(side)]
            for row in range(side)
        ]
    )
    span = match_span(match)
    assert span is not None
    ys, xs = span
    view = match_view(np.zeros((ys.size, xs.size, 3), dtype=np.uint8), match, ys, xs, (1, 2, 3))
    # Only the rows and columns the match touches are in the view at all.
    assert view.xs.tolist() == [column for column in range(side) if match[:, column].any()]
    assert view.ys.tolist() == [row for row in range(side) if match[row, :].any()]
    for dy in range(view.height):
        for dx in range(view.width):
            coord = view.source_at(dx, dy)
            # Each cell is its own source pixel, and carries that pixel's bit.
            assert view.display_of(coord) == (dx, dy)
            assert bool(view.match[dy, dx]) == bool(match[coord.y, coord.x])


@given(zoom=st.integers(min_value=1, max_value=int(MAX_ZOOM)))
def test_the_slider_round_trips_every_whole_zoom(zoom: int) -> None:
    position = slider_position(zoom)
    assert 0 <= position <= SLIDER_STEPS
    assert slider_zoom(position) == zoom


def _chosen(image: LoadedImage, bits: set[int]) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane.name, bit) for plane in image.planes for bit in bits)


def _transposed(image: LoadedImage) -> LoadedImage:
    return make_image(image.samples.transpose(1, 0, 2), planes_rgb())


@given(width=_SIZES, height=_SIZES, bits=_BITS, seed=st.integers(0, 5))
def test_the_column_order_is_the_transposed_image_read_by_rows(
    width: int, height: int, bits: set[int], seed: int
) -> None:
    image = _image(width, height, seed=seed)
    columns = ExtractOrder(scan=ScanOrder.YZ)
    assert extract_bytes(image, _chosen(image, bits), order=columns) == extract_bytes(
        _transposed(image), _chosen(image, bits)
    )


@given(pattern=st.integers(min_value=0, max_value=0xFFFF), bits=_BITS)
def test_the_column_order_carries_the_filter_along(pattern: int, bits: set[int]) -> None:
    side = 4
    image = _image(side, side, seed=7)
    mask = np.array(
        [
            [bool(pattern >> (row * side + column) & 1) for column in range(side)]
            for row in range(side)
        ]
    )
    columns = ExtractOrder(scan=ScanOrder.YZ)
    assert extract_bytes(image, _chosen(image, bits), mask, order=columns) == extract_bytes(
        _transposed(image), _chosen(image, bits), mask.T
    )


@given(width=_SIZES, height=_SIZES, seed=st.integers(0, 5))
def test_low_first_returns_the_stored_bytes_of_a_whole_channel(
    width: int, height: int, seed: int
) -> None:
    image = _image(width, height, seed=seed)
    chosen = frozenset(BitChoice("R", bit) for bit in range(8))
    stored = image.samples[:, :, 0].astype(np.uint8).tobytes()
    low_first = ExtractOrder(bit_order=BitOrder.LSB)
    assert extract_bytes(image, chosen, order=low_first) == stored
    # High first packs bit 0 into the top of the byte, which mirrors it.
    mirrored = bytes(int(f"{byte:08b}"[::-1], 2) for byte in stored)
    assert extract_bytes(image, chosen) == mirrored


@given(width=_SIZES, height=_SIZES, seed=st.integers(0, 5))
def test_a_channel_order_permutes_the_byte_groups_of_every_pixel(
    width: int, height: int, seed: int
) -> None:
    image = _image(width, height, seed=seed)
    chosen = frozenset(BitChoice(name, bit) for name in ("R", "G", "B") for bit in range(8))
    every = extract_bytes(image, chosen)
    reversed_channels = ExtractOrder(planes=("B", "G", "R"))
    swapped = extract_bytes(image, chosen, order=reversed_channels)
    groups = [every[start : start + 3] for start in range(0, len(every), 3)]
    assert swapped == b"".join(group[::-1] for group in groups)

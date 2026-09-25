import numpy as np
import pytest

from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    InvertMask,
    RegionMask,
    SampleOrigin,
    SamplePlane,
)
from pixelsb.domain.samples import (
    bit_plane,
    composite_on_checkerboard,
    render_export,
    render_raster,
)
from tests.support import make_image, planes_rgb, planes_rgba, raster


def _plane() -> SamplePlane:
    return SamplePlane("L", 0, 8, SampleOrigin.RAW)


def _selection(*choices: tuple[str, int]) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane, bit) for plane, bit in choices)


def test_bit_plane_treats_bit_zero_as_the_lsb() -> None:
    samples = np.array(
        [
            [[0b00000001], [0b10000000]],
            [[0b00000000], [0b11111111]],
        ],
        dtype=np.uint16,
    )
    plane = _plane()
    low = bit_plane(samples, plane, 0)
    high = bit_plane(samples, plane, 7)
    assert low.tolist() == [[255, 0], [0, 255]]
    assert high.tolist() == [[0, 255], [0, 255]]


def test_bit_plane_rejects_a_bit_outside_the_plane() -> None:
    samples = np.zeros((1, 1, 1), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"outside 0\.\.7"):
        bit_plane(samples, _plane(), 8)


def test_checkerboard_composite_uses_integer_alpha() -> None:
    clear = np.zeros((8, 8, 4), dtype=np.uint8)
    assert composite_on_checkerboard(clear, cell=8)[0, 0].tolist() == [232, 232, 232]

    wide = np.zeros((1, 16, 4), dtype=np.uint8)
    composed = composite_on_checkerboard(wide, cell=8)
    assert composed[0, 0].tolist() == [232, 232, 232]
    assert composed[0, 8].tolist() == [255, 255, 255]

    opaque = np.zeros((1, 1, 4), dtype=np.uint8)
    opaque[0, 0] = (255, 0, 0, 255)
    assert composite_on_checkerboard(opaque)[0, 0].tolist() == [255, 0, 0]

    half = np.zeros((1, 1, 4), dtype=np.uint8)
    half[0, 0] = (255, 0, 0, 128)
    assert composite_on_checkerboard(half, cell=8)[0, 0].tolist() == [243, 115, 115]


def test_the_original_render_uses_the_channel_samples() -> None:
    image = make_image(np.array([[[1, 2, 3]]], dtype=np.uint16), planes_rgb())
    assert render_raster(raster(image))[0, 0].tolist() == [1, 2, 3]


def test_one_selected_bit_renders_as_a_bitmap() -> None:
    image = make_image(np.array([[[0b00000001, 0, 0]]], dtype=np.uint16), planes_rgb())
    bitmap = render_raster(raster(image, BitsMask(_selection(("R", 0)))))
    assert bitmap[0, 0].tolist() == [255, 255, 255]


def test_several_selected_bits_keep_their_original_weights() -> None:
    image = make_image(np.array([[[0b00001001, 0, 0]]], dtype=np.uint16), planes_rgb())
    weighted = render_raster(raster(image, BitsMask(_selection(("R", 0), ("R", 3)))))
    assert weighted[0, 0].tolist() == [255, 0, 0]


def test_what_a_mask_left_shows_through_the_render() -> None:
    image = make_image(np.array([[[0, 0, 0]]], dtype=np.uint16), planes_rgb())
    assert render_raster(raster(image, InvertMask()))[0, 0].tolist() == [255, 255, 255]


def test_a_cropped_raster_renders_the_pixels_the_full_one_holds_there() -> None:
    samples = np.arange(2 * 4 * 3, dtype=np.uint16).reshape(2, 4, 3) * 8
    image = make_image(samples, planes_rgb())
    full = render_raster(raster(image))
    cropped = raster(image, RegionMask("left >= 1 and left <= 2"), CropMask())
    assert np.array_equal(render_raster(cropped), full[:, 1:3])


def test_a_cropped_raster_keeps_the_checkerboard_phase() -> None:
    samples = np.zeros((4, 5, 4), dtype=np.uint16)
    samples[..., :3] = 200
    samples[..., 3] = 128
    image = make_image(samples, planes_rgba())
    full = render_raster(raster(image))
    cropped = raster(image, RegionMask("left >= 1"), CropMask())
    assert np.array_equal(render_raster(cropped), full[:, 1:])


def test_export_keeps_the_alpha_channel_the_canvas_paints_over_a_board() -> None:
    """Transparency is a viewing aid on screen and a channel in the saved file."""
    samples = np.zeros((2, 2, 4), dtype=np.uint16)
    samples[..., :3] = 100
    samples[..., 3] = [[0, 128], [255, 64]]
    image = make_image(samples, planes_rgba())
    board = render_raster(raster(image))
    assert board[0, 0].tolist() == [232, 232, 232]  # fully clear: the board shows
    exported = render_export(raster(image))
    assert exported[..., :3].tolist() == [[[100, 100, 100], [100, 100, 100]]] * 2
    assert exported[..., 3].tolist() == [[0, 128], [255, 64]]


def test_export_without_alpha_is_the_plain_picture() -> None:
    image = make_image(np.array([[[1, 2, 3]]], dtype=np.uint16), planes_rgb())
    plain = render_export(raster(image))
    assert plain.shape == (1, 1, 3)
    assert plain[0, 0].tolist() == [1, 2, 3]
    # A selection that leaves the alpha channel out keeps three channels too.
    colors = BitsMask(_selection(("R", 7), ("G", 7), ("B", 7)))
    assert render_export(raster(image, colors)).shape == (1, 1, 3)

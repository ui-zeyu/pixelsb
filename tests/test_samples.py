import numpy as np

from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    InvertMask,
    RegionMask,
)
from pixelsb.domain.samples import (
    composite_on_checkerboard,
    render_export,
    render_raster,
)
from tests.support import make_image, planes_rgb, planes_rgba, raster


def _selection(*choices: tuple[str, int]) -> BitsMask:
    return BitsMask(frozenset(BitChoice(plane, bit) for plane, bit in choices))


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


def test_a_projected_single_bit_renders_as_a_bitmap() -> None:
    """One plane alone reads as gray: the bit plane the projection made."""
    image = make_image(np.array([[[0b00000001, 0, 0]]], dtype=np.uint16), planes_rgb())
    bitmap = render_raster(raster(image, _selection(("R", 0))))
    assert bitmap[0, 0].tolist() == [255, 255, 255]


def test_a_projected_pair_of_bits_keeps_their_relative_weights() -> None:
    image = make_image(np.array([[[0b00000010, 0, 0]]], dtype=np.uint16), planes_rgb())
    weighted = render_raster(raster(image, _selection(("R", 0), ("R", 1))))
    assert weighted[0, 0].tolist() == [170, 170, 170]  # 2 of 0..3, scaled to a byte


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
    # A projection that names the colors keeps three channels and no alpha.
    colors = _selection(("R", 7), ("G", 7), ("B", 7))
    assert render_export(raster(image, colors)).shape == (1, 1, 3)

import numpy as np
import pytest

from pixelsb.domain.models import BitChoice, SampleOrigin, SamplePlane
from pixelsb.domain.samples import bit_plane, composite_on_checkerboard, render_rgb, render_rgb_at
from tests.support import make_image


def _plane() -> SamplePlane:
    return SamplePlane("L", 0, 8, SampleOrigin.RAW)


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


def test_original_render_uses_channel_samples_and_one_bit_is_grayscale() -> None:
    samples = np.array([[[0b00000001, 0, 0]]], dtype=np.uint16)
    image = make_image(samples, _rgb())
    original = render_rgb(image, None)
    assert original[0, 0].tolist() == [1, 0, 0]
    plane = render_rgb(image, frozenset({BitChoice("R", 0)}))
    assert plane[0, 0].tolist() == [255, 255, 255]


def _rgb() -> tuple[SamplePlane, SamplePlane, SamplePlane]:
    return (
        SamplePlane("R", 0, 8, SampleOrigin.RAW),
        SamplePlane("G", 1, 8, SampleOrigin.RAW),
        SamplePlane("B", 2, 8, SampleOrigin.RAW),
    )


def test_render_rgb_at_matches_the_gathered_full_render() -> None:
    samples = np.arange(2 * 4 * 3, dtype=np.uint16).reshape(2, 4, 3) * 8
    image = make_image(samples, _rgb())
    ys = np.array([1, 0, 1])
    xs = np.array([3, 0, 2])
    selection = frozenset({BitChoice("R", bit) for bit in range(8)})
    at = render_rgb_at(image, selection, ys, xs)
    assert np.array_equal(at, render_rgb(image, selection)[np.ix_(ys, xs)])
    assert np.array_equal(
        render_rgb_at(image, None, ys, xs), render_rgb(image, None)[np.ix_(ys, xs)]
    )


def test_every_bit_mask_scales_exactly_like_the_division() -> None:
    """The multiply fast path must agree with ``value * 255 // maximum``."""
    rng = np.random.default_rng(7)
    samples = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint16)
    image = make_image(samples, _rgb())
    for mask in range(1, 256):
        bits = tuple(bit for bit in range(8) if mask >> bit & 1)
        chosen = frozenset(BitChoice(name, bit) for name in ("R", "G", "B") for bit in bits)
        expected = np.stack(
            [
                ((samples[:, :, index] & mask).astype(np.uint32) * 255 // mask).astype(np.uint8)
                for index in range(3)
            ],
            axis=-1,
        )
        assert np.array_equal(render_rgb(image, chosen), expected), f"mask {mask:#04x}"


def test_render_rgb_at_keeps_the_checkerboard_phase() -> None:
    planes = (
        SamplePlane("R", 0, 8, SampleOrigin.RAW),
        SamplePlane("G", 1, 8, SampleOrigin.RAW),
        SamplePlane("B", 2, 8, SampleOrigin.RAW),
        SamplePlane("A", 3, 8, SampleOrigin.RAW),
    )
    samples = np.zeros((4, 5, 4), dtype=np.uint16)
    samples[..., :3] = 200
    samples[..., 3] = 128
    image = make_image(samples, planes)
    alpha = frozenset({BitChoice("A", bit) for bit in range(8)})
    ys = np.array([1, 3])
    xs = np.array([0, 2, 3])
    at = render_rgb_at(image, alpha, ys, xs)
    assert np.array_equal(at, render_rgb(image, alpha)[np.ix_(ys, xs)])

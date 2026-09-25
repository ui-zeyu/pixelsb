import numpy as np
import pytest

from pixelsb.domain.adjust import apply_adjust
from pixelsb.domain.models import ViewAdjust


def rgb(pixels: list[list[tuple[int, int, int]]]) -> np.ndarray:
    return np.array(pixels, dtype=np.uint8)


def test_no_adjustment_keeps_the_pixels_and_the_object() -> None:
    pixels = rgb([[(10, 128, 250)]])
    assert apply_adjust(pixels, ViewAdjust()) is pixels


def test_invert_is_255_minus_the_channel() -> None:
    pixels = rgb([[(10, 128, 250)]])
    inverted = apply_adjust(pixels, ViewAdjust(invert=True))
    assert inverted.tolist() == [[[245, 127, 5]]]


def test_grayscale_mixes_channels_by_luma() -> None:
    pixels = rgb([[(255, 0, 0), (0, 255, 0), (0, 0, 255)]])
    gray = apply_adjust(pixels, ViewAdjust(grayscale=True))
    assert gray.tolist() == [[[54, 54, 54], [182, 182, 182], [18, 18, 18]]]


def test_threshold_binarizes_around_the_level() -> None:
    pixels = rgb([[(10, 128, 129)]])
    cut = apply_adjust(pixels, ViewAdjust(threshold=True, level=128))
    assert cut.tolist() == [[[0, 0, 255]]]


def test_the_adjustments_compose_in_order() -> None:
    """Grayscale, then threshold, then invert: red's luma 54 falls under 64 to
    black, and the invert turns it white."""
    pixels = rgb([[(255, 0, 0)]])
    combined = ViewAdjust(grayscale=True, threshold=True, invert=True, level=64)
    assert apply_adjust(pixels, combined).tolist() == [[[255, 255, 255]]]


def test_the_level_must_be_a_byte_value() -> None:
    with pytest.raises(ValueError, match=r"0\.\.255"):
        ViewAdjust(level=256)
    with pytest.raises(ValueError, match=r"0\.\.255"):
        ViewAdjust(level=-1)

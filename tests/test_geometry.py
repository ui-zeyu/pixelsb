from itertools import pairwise

from pixelsb.domain.geometry import (
    SLIDER_STEPS,
    initial_zoom,
    pixel_at,
    slider_position,
)
from pixelsb.domain.models import PixelCoord


def test_pixel_at_uses_integer_zoom() -> None:
    assert pixel_at(0, 0, 10, 2, 2) == PixelCoord(0, 0)
    assert pixel_at(19, 19, 10, 2, 2) == PixelCoord(1, 1)
    assert pixel_at(20, 0, 10, 2, 2) is None
    assert pixel_at(-1, 0, 10, 2, 2) is None
    assert pixel_at(0, 0, 0, 2, 2) is None


def test_initial_zoom_fits_and_stays_in_range() -> None:
    assert initial_zoom((100, 100), (400, 300)) == 3.0
    assert initial_zoom((1000, 1000), (100, 100)) == 1.0
    assert initial_zoom((10, 10), (1000, 1000)) == 16.0
    assert initial_zoom((10, 10), (1000, 1000), cap=100) == 100.0
    assert initial_zoom((10, 10), (0, 100)) == 1.0


def test_initial_zoom_fills_one_axis_with_fractional_zoom() -> None:
    fit = initial_zoom((360, 240), (1055, 851))
    assert fit == 1055 / 360
    assert fit > 2
    assert fit < 3


def test_the_slider_spends_equal_travel_per_doubling() -> None:
    """Each doubling takes the same slider travel, so the scale reads as a ruler."""
    positions = [slider_position(zoom) for zoom in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert positions[0] == 0
    assert positions[-1] == SLIDER_STEPS
    gaps = [later - earlier for earlier, later in pairwise(positions)]
    assert max(gaps) - min(gaps) <= 1

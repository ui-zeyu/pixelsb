from pixelsb.domain.geometry import initial_zoom, pixel_at
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
    assert fit > 2 and fit < 3

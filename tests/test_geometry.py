"""Zoom arithmetic: the pixel mapping, the fit, and the slider's ruler."""

from itertools import pairwise

import pytest

from pixelsb.domain.geometry import (
    SLIDER_STEPS,
    initial_zoom,
    pixel_at,
    slider_position,
    slider_zoom,
)
from pixelsb.domain.models import MIN_ZOOM, PixelCoord


def test_pixel_at_truncates_the_scaled_position() -> None:
    assert pixel_at(0, 0, 10, 2, 2) == PixelCoord(0, 0)
    assert pixel_at(19, 19, 10, 2, 2) == PixelCoord(1, 1)
    assert pixel_at(20, 0, 10, 2, 2) is None
    assert pixel_at(-1, 0, 10, 2, 2) is None
    assert pixel_at(0, 0, 0, 2, 2) is None


def test_pixel_at_maps_several_widget_pixels_to_one_below_one_to_one() -> None:
    assert pixel_at(0.9, 0.9, 0.5, 2, 2) == PixelCoord(1, 1)
    assert pixel_at(0.4, 0.9, 0.5, 2, 2) == PixelCoord(0, 1)
    assert pixel_at(1.9, 1.9, 0.5, 2, 2) is None


def test_initial_zoom_fits_and_stays_in_range() -> None:
    assert initial_zoom((100, 100), (400, 300)) == 3.0
    assert initial_zoom((1000, 1000), (100, 100)) == 0.1
    assert initial_zoom((4000, 3000), (800, 600)) == 0.2
    assert initial_zoom((10, 10), (1000, 1000)) == 16.0
    assert initial_zoom((10, 10), (1000, 1000), cap=100) == 100.0
    assert initial_zoom((10, 10), (0, 100)) == MIN_ZOOM


def test_initial_zoom_fills_one_axis_with_fractional_zoom() -> None:
    fit = initial_zoom((360, 240), (1055, 851))
    assert fit == 1055 / 360
    assert fit > 2
    assert fit < 3


def test_the_slider_spends_equal_travel_per_doubling() -> None:
    """Each doubling takes the same slider travel, so the scale reads as a ruler."""
    ladder = [MIN_ZOOM * 2.0**octave for octave in range(12)]
    positions = [slider_position(zoom) for zoom in ladder]
    assert positions[0] == 0
    assert positions[-1] == SLIDER_STEPS
    gaps = [later - earlier for earlier, later in pairwise(positions)]
    assert max(gaps) - min(gaps) <= 1


def test_the_slider_reads_whole_ladder_steps_exactly() -> None:
    for octave in range(12):
        zoom = MIN_ZOOM * 2.0**octave
        assert slider_zoom(slider_position(zoom)) == pytest.approx(zoom, rel=0.01)

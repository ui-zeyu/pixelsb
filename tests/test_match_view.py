import numpy as np

from pixelsb.domain.match_view import match_span, match_view
from pixelsb.domain.models import PixelCoord

_RGB = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
_BACKGROUND = (235, 237, 240)


def test_grid_mask_spans_the_strided_lattice() -> None:
    match = np.zeros((2, 4), dtype=bool)
    match[::2, ::2] = True
    span = match_span(match)
    assert span is not None
    ys, xs = span
    assert xs.tolist() == [0, 2]
    assert ys.tolist() == [0]
    view = match_view(_RGB[0:1, 0:4:2], match, ys, xs, _BACKGROUND)
    assert (view.width, view.height) == (2, 1)
    assert np.array_equal(view.rgb, _RGB[0:1, 0:4:2])
    assert view.match.all()


def test_rect_mask_compacts_to_the_crop() -> None:
    match = np.zeros((2, 4), dtype=bool)
    match[:, 1:3] = True
    span = match_span(match)
    assert span is not None
    ys, xs = span
    view = match_view(_RGB[:, 1:3], match, ys, xs, _BACKGROUND)
    assert np.array_equal(view.rgb, _RGB[:, 1:3])
    assert view.match.all()


def test_scattered_mask_keeps_holes_on_the_background() -> None:
    match = np.array([[True, False], [False, True]])
    span = match_span(match)
    assert span is not None
    ys, xs = span
    view = match_view(_RGB[np.ix_(ys, xs)], match, ys, xs, _BACKGROUND)
    assert np.array_equal(view.rgb[0, 0], _RGB[0, 0])
    assert tuple(view.rgb[0, 1]) == _BACKGROUND
    assert tuple(view.rgb[1, 0]) == _BACKGROUND
    assert np.array_equal(view.rgb[1, 1], _RGB[1, 1])
    assert view.match.tolist() == [[True, False], [False, True]]


def test_coordinate_mappings_round_trip() -> None:
    match = np.zeros((2, 4), dtype=bool)
    match[1, ::2] = True
    span = match_span(match)
    assert span is not None
    ys, xs = span
    view = match_view(_RGB[1:2, ::2], match, ys, xs, _BACKGROUND)
    assert (view.width, view.height) == (2, 1)
    assert view.source_at(1, 0) == PixelCoord(2, 1)
    assert view.display_of(PixelCoord(2, 1)) == (1, 0)
    assert view.display_of(PixelCoord(1, 1)) is None
    assert view.display_of(PixelCoord(2, 0)) is None


def test_empty_match_yields_no_span() -> None:
    match = np.zeros((2, 4), dtype=bool)
    assert match_span(match) is None

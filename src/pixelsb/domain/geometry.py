"""Widget-to-pixel mapping and zoom helpers. Zoom is fractional."""

import math

from pixelsb.domain.models import MAX_ZOOM, MIN_ZOOM, PixelCoord

GRID_ZOOM = 8
SLIDER_STEPS = 1000
_ZOOM_RATIO = MAX_ZOOM / MIN_ZOOM


def pixel_at(
    widget_x: float,
    widget_y: float,
    zoom: float,
    width: int,
    height: int,
) -> PixelCoord | None:
    """Map a logical widget position to a pixel. ``zoom`` is the canvas scale."""
    if zoom < 1 or width < 1 or height < 1 or widget_x < 0 or widget_y < 0:
        return None
    x = int(widget_x / zoom)
    y = int(widget_y / zoom)
    if x >= width or y >= height:
        return None
    return PixelCoord(x, y)


def initial_zoom(
    image: tuple[int, int],
    viewport: tuple[int, int],
    *,
    cap: float = 16.0,
) -> float:
    """Largest zoom that keeps the image fully visible. Fills one axis exactly."""
    image_w, image_h = image
    view_w, view_h = viewport
    if image_w < 1 or image_h < 1 or view_w < 1 or view_h < 1:
        return MIN_ZOOM
    fit = min(view_w / image_w, view_h / image_h)
    return min(max(fit, MIN_ZOOM), cap, MAX_ZOOM)


def slider_zoom(position: int) -> int:
    """The whole-number zoom at a slider position; each doubling takes equal travel."""
    return max(int(MIN_ZOOM), round(MIN_ZOOM * _ZOOM_RATIO ** (position / SLIDER_STEPS)))


def slider_position(zoom: float) -> int:
    """Where a zoom sits on the slider, the inverse of :func:`slider_zoom`."""
    return round(SLIDER_STEPS * math.log(zoom / MIN_ZOOM) / math.log(_ZOOM_RATIO))

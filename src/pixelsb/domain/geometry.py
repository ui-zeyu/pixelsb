"""Integer zoom and widget-to-pixel mapping."""

from pixelsb.domain.models import MAX_ZOOM, MIN_ZOOM, PixelCoord

GRID_ZOOM = 8


def pixel_at(
    widget_x: int,
    widget_y: int,
    zoom: int,
    width: int,
    height: int,
) -> PixelCoord | None:
    """Map a logical widget position to a pixel. ``zoom`` is the canvas scale."""
    if zoom < 1 or width < 1 or height < 1 or widget_x < 0 or widget_y < 0:
        return None
    x = widget_x // zoom
    y = widget_y // zoom
    if x >= width or y >= height:
        return None
    return PixelCoord(x, y)


def initial_zoom(
    image: tuple[int, int],
    viewport: tuple[int, int],
    *,
    cap: int = 16,
) -> int:
    image_w, image_h = image
    view_w, view_h = viewport
    if image_w < 1 or image_h < 1 or view_w < 1 or view_h < 1:
        return MIN_ZOOM
    fit = min(view_w // image_w, view_h // image_h)
    return min(max(fit, MIN_ZOOM), min(cap, MAX_ZOOM))


def spatial_offset(origin: PixelCoord, target: PixelCoord) -> tuple[int, int]:
    """Return ``target - origin`` as ``(dx, dy)``."""
    return target.x - origin.x, target.y - origin.y

"""The picture a raster paints: scaled channels, checkerboards, exports.

The samples already are what the viewer sees — the bits projection lives in the
stack, so painting reads the planes as they are and scales each by its own
maximum. There is no second notion of "the shown value" anywhere: the readout,
the histogram, and the extract stream all read these same samples.
"""

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    COLOR_SLOTS,
    GRAY_SLOTS,
    ChannelArray,
    IndexArray,
    Raster,
    RgbaArray,
    RgbArray,
)

_CHECKER_DARK = 232
_CHECKER_LIGHT = 255


def render_raster(raster: Raster) -> RgbArray:
    """Compose the raster into the RGB picture the canvas paints.

    Transparency is painted as a checkerboard, which is a viewing aid: the file a
    dump is saved to keeps the alpha channel instead (:func:`render_export`).
    """
    painted, alpha = _painted(raster)
    if alpha is None:
        return painted
    return composite_on_checkerboard(
        np.dstack([painted, alpha]), rows=raster.rows, columns=raster.columns
    )


def painted_rgb(raster: Raster) -> RgbArray:
    """The colors the canvas paints, with no viewing aid layered on.

    The checkerboard behind a transparency is a screen aid, and the file a dump
    is saved to keeps the alpha channel beside these colors instead
    (:func:`render_export`); this is the picture both of them start from.
    """
    return _painted(raster)[0]


def render_export(raster: Raster) -> RgbArray | RgbaArray:
    """The composed picture as a saved file holds it: alpha kept, no checkerboard.

    The same rule the region dimming follows — what is only there to look at is
    not written into the file — so a picture carrying an alpha channel is
    written with four channels and stays transparent.
    """
    painted, alpha = _painted(raster)
    return painted if alpha is None else np.dstack([painted, alpha])


def composite_on_checkerboard(
    rgba: NDArray[np.uint8],
    cell: int = 8,
    rows: IndexArray | None = None,
    columns: IndexArray | None = None,
) -> RgbArray:
    """Composite straight RGBA over a light checkerboard. Returns HxWx3 uint8.

    ``rows``/``columns`` are the source indices of the block's rows and columns;
    the pattern keeps its original phase when rendering a gathered sub-block.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an HxWx4 array")
    if cell < 1:
        raise ValueError("cell size must be positive")
    height, width = rgba.shape[:2]
    row_index = np.arange(height) if rows is None else rows
    column_index = np.arange(width) if columns is None else columns
    light = ((row_index[:, None] // cell + column_index[None, :] // cell) & 1) == 1
    background = np.where(light, np.uint8(_CHECKER_LIGHT), np.uint8(_CHECKER_DARK)).astype(
        np.uint16
    )
    foreground = rgba[..., :3].astype(np.uint16)
    alpha = rgba[..., 3].astype(np.uint16)[..., None]
    mixed = (foreground * alpha + background[..., None] * (np.uint16(255) - alpha)) // np.uint16(
        255
    )
    return mixed.astype(np.uint8)


def _painted(raster: Raster) -> tuple[RgbArray, ChannelArray | None]:
    """The raster's color planes, and its alpha channel when it has one.

    One plane alone reads as gray, whatever its name — a projected single
    channel reads as gray on screen, and alpha by itself reads as gray rather
    than compositing with itself. Otherwise the color trio paints in their
    slots, a gray plane stands in for the whole image, and alpha travels back
    beside the colors, so a caller can either layer it over a checkerboard or
    keep it in the file.
    """
    scaled = _scaled_planes(raster)
    if not scaled:
        return np.zeros((raster.height, raster.width, 3), dtype=np.uint8), None
    if len(scaled) == 1:
        only = next(iter(scaled.values()))
        return np.stack([only, only, only], axis=-1), None
    alpha = scaled.get("A")
    colors = [scaled.get(slot) for slot in COLOR_SLOTS]
    if any(channel is not None for channel in colors):
        blank = np.zeros((raster.height, raster.width), dtype=np.uint8)
        painted = np.stack([blank if channel is None else channel for channel in colors], axis=-1)
        return painted, alpha
    gray = next((scaled[slot] for slot in GRAY_SLOTS if slot in scaled), None)
    if gray is not None:
        # A gray plane stands in for the whole image, as it does on screen.
        return np.stack([gray, gray, gray], axis=-1), alpha
    if alpha is not None:
        return np.stack([alpha, alpha, alpha], axis=-1), None
    return np.zeros((raster.height, raster.width, 3), dtype=np.uint8), None


def _scaled_planes(raster: Raster) -> dict[str, ChannelArray]:
    """One scaled byte array per plane of the raster, by the plane's own maximum."""
    return {
        plane.name: scale_to_byte(raster.samples[:, :, plane.index], plane.maximum)
        for plane in raster.planes
    }


def scale_to_byte(value: NDArray[np.uint16], maximum: int) -> ChannelArray:
    """Map the range 0..maximum onto 0..255."""
    if maximum <= 0:
        return np.zeros(value.shape, dtype=np.uint8)
    factor = 255 // maximum
    if factor == 1 and maximum == 255:
        return value.astype(np.uint8)
    if factor * maximum == 255:
        # Full bytes and the maxima 1, 3 and 15 divide 255 exactly, so one
        # multiply replaces a whole uint32 divide pass over the image.
        return (value * np.uint16(factor)).astype(np.uint8)
    return (value.astype(np.uint32) * np.uint32(255) // np.uint32(maximum)).astype(np.uint8)

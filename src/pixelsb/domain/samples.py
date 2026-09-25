"""Bit planes and the picture the selected ones compose."""

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    COLOR_SLOTS,
    ChannelArray,
    IndexArray,
    Raster,
    RgbaArray,
    RgbArray,
    SampleArray,
    SamplePlane,
    plane_named,
)
from pixelsb.domain.selection import bits_for, mask_of

_CHECKER_DARK = 232
_CHECKER_LIGHT = 255
# A gray plane standing in for the whole image, in the order they are tried.
_GRAY_SLOTS = ("L", "Index")


def bit_plane(
    samples: SampleArray,
    plane: SamplePlane,
    bit: int,
) -> ChannelArray:
    """Return an HxW image of 0 and 255. Bit 0 is the least significant bit."""
    if not 0 <= bit < plane.bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{plane.bit_depth - 1}")
    if samples.ndim != 3 or not 0 <= plane.index < samples.shape[2]:
        raise ValueError("plane index is outside the sample array")
    channel = samples[:, :, plane.index]
    selected = ((channel >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8)
    return selected * np.uint8(255)


def render_raster(raster: Raster) -> RgbArray:
    """Compose the selected bits into the RGB picture the canvas paints.

    Transparency is painted as a checkerboard, which is a viewing aid: the file a
    dump is saved to keeps the alpha channel instead (:func:`render_export`).
    """
    painted, alpha = _painted(raster)
    if alpha is None:
        return painted
    return composite_on_checkerboard(
        np.dstack([painted, alpha]), rows=raster.rows, columns=raster.columns
    )


def render_export(raster: Raster) -> RgbArray | RgbaArray:
    """The composed picture as a saved file holds it: alpha kept, no checkerboard.

    The same rule the region dimming follows — what is only there to look at is
    not written into the file — so a picture whose selection carries an alpha
    channel is written with four channels and stays transparent.
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
    """The color planes the selection paints, and its alpha channel when it has one.

    The first group that carries anything paints the picture: the color triplet
    (a channel the selection misses stays black), then a gray plane standing in
    for the whole image, then alpha, which alone reads as gray rather than
    compositing with itself. An alpha channel travels back beside the colors, so
    a caller can either layer it over a checkerboard or keep it in the file.
    """
    chosen = raster.selection
    if not chosen:
        return np.zeros((raster.height, raster.width, 3), dtype=np.uint8), None
    if len(chosen) == 1:
        choice = next(iter(chosen))
        gray = bit_plane(raster.samples, plane_named(raster.planes, choice.plane), choice.bit)
        return np.stack([gray, gray, gray], axis=-1), None
    scaled = _scaled_planes(raster)
    alpha = scaled.get("A")
    colors = [scaled.get(slot) for slot in COLOR_SLOTS]
    gray = next((scaled[slot] for slot in _GRAY_SLOTS if slot in scaled), None)
    if any(channel is not None for channel in colors):
        blank = np.zeros((raster.height, raster.width), dtype=np.uint8)
        painted = np.stack([blank if channel is None else channel for channel in colors], axis=-1)
        return painted, alpha
    if gray is not None:
        # A gray plane stands in for the whole image, as it does on screen.
        return np.stack([gray, gray, gray], axis=-1), alpha
    if alpha is not None:
        # Alpha alone reads as a gray image rather than compositing with itself.
        return np.stack([alpha, alpha, alpha], axis=-1), None
    return np.zeros((raster.height, raster.width, 3), dtype=np.uint8), None


def _scaled_planes(raster: Raster) -> dict[str, ChannelArray]:
    """One scaled byte array per plane the selection touches."""
    return {
        plane.name: _scale_to_byte(*_masked_channel(raster.samples[:, :, plane.index], bits))
        for plane in raster.planes
        if (bits := bits_for(raster.selection, plane.name))
    }


def _masked_channel(
    channel: NDArray[np.uint16],
    bits: tuple[int, ...],
) -> tuple[NDArray[np.uint16], int]:
    """The selected bits shifted down to start at bit zero, and their maximum.

    Shifting loses no information: the masked value is the shifted one times
    ``2 ** low``, and so is the maximum, so the scaled byte is identical. A
    contiguous run then has a maximum of ``2 ** width - 1``, which is what lets
    the byte scaling multiply instead of divide for the common selections.
    """
    low, high = bits[0], bits[-1]
    if bits == tuple(range(low, high + 1)):
        width = high - low + 1
        run = channel if low == 0 else channel >> np.uint16(low)
        return run & np.uint16((1 << width) - 1), (1 << width) - 1
    mask = mask_of(bits)
    return channel & np.uint16(mask), mask


def _scale_to_byte(value: NDArray[np.uint16], maximum: int) -> ChannelArray:
    """Map the masked range 0..maximum onto 0..255."""
    if maximum <= 0:
        return np.zeros(value.shape, dtype=np.uint8)
    factor = 255 // maximum
    if factor == 1 and maximum == 255:
        return value.astype(np.uint8)
    if factor * maximum == 255:
        # Full bytes and the masks 1, 3 and 15 divide 255 exactly, so one
        # multiply replaces a whole uint32 divide pass over the image.
        return (value * np.uint16(factor)).astype(np.uint8)
    return (value.astype(np.uint32) * np.uint32(255) // np.uint32(maximum)).astype(np.uint8)

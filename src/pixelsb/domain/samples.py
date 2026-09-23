"""Bit planes and preview compositing."""

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    BitChoice,
    LoadedImage,
    RgbArray,
    SampleArray,
    SamplePlane,
)
from pixelsb.domain.selection import bits_for, effective_selection, mask_of

_CHECKER_DARK = 232
_CHECKER_LIGHT = 255


def bit_plane(
    samples: SampleArray,
    plane: SamplePlane,
    bit: int,
) -> NDArray[np.uint8]:
    """Return an HxW image of 0 and 255. Bit 0 is the least significant bit."""
    if not 0 <= bit < plane.bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{plane.bit_depth - 1}")
    if samples.ndim != 3 or not 0 <= plane.index < samples.shape[2]:
        raise ValueError("plane index is outside the sample array")
    channel = samples[:, :, plane.index]
    selected = ((channel >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8)
    return selected * np.uint8(255)


def composite_on_checkerboard(rgba: NDArray[np.uint8], cell: int = 8) -> RgbArray:
    """Composite straight RGBA over a light checkerboard. Returns HxWx3 uint8."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an HxWx4 array")
    if cell < 1:
        raise ValueError("cell size must be positive")
    height, width = rgba.shape[:2]
    light = ((np.arange(height)[:, None] // cell + np.arange(width)[None, :] // cell) & 1) == 1
    background = np.where(light, np.uint8(_CHECKER_LIGHT), np.uint8(_CHECKER_DARK)).astype(
        np.uint16
    )
    foreground = rgba[..., :3].astype(np.uint16)
    alpha = rgba[..., 3].astype(np.uint16)[..., None]
    mixed = (foreground * alpha + background[..., None] * (np.uint16(255) - alpha)) // np.uint16(
        255
    )
    return mixed.astype(np.uint8)


def render_rgb(
    image: LoadedImage,
    selection: frozenset[BitChoice] | None,
) -> RgbArray:
    """Selected bits keep their original channel and weight. One bit renders as a bitmap."""
    chosen = effective_selection(image, selection)
    if not chosen:
        return np.zeros((image.height, image.width, 3), dtype=np.uint8)
    if len(chosen) == 1:
        choice = next(iter(chosen))
        plane = image.plane(choice.plane)
        gray = bit_plane(image.samples, plane, choice.bit)
        return np.stack([gray, gray, gray], axis=-1)
    return _mask_rgb(image, chosen)


def _mask_rgb(image: LoadedImage, chosen: frozenset[BitChoice]) -> RgbArray:
    color = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    alpha = np.full((image.height, image.width), 255, dtype=np.uint8)
    wrote_color = False
    luma: NDArray[np.uint8] | None = None
    wrote_alpha = False
    for plane in image.planes:
        bits = bits_for(chosen, plane.name)
        if not bits:
            continue
        scaled = _scale_to_byte(
            _masked_channel(image.samples[:, :, plane.index], bits),
            mask_of(bits),
        )
        if plane.name == "R":
            color[:, :, 0] = scaled
            wrote_color = True
        elif plane.name == "G":
            color[:, :, 1] = scaled
            wrote_color = True
        elif plane.name == "B":
            color[:, :, 2] = scaled
            wrote_color = True
        elif plane.name == "A":
            alpha = scaled
            wrote_alpha = True
        elif plane.name == "L" or (plane.name == "Index" and luma is None):
            luma = scaled
    if not wrote_color and luma is not None:
        color = np.stack([luma, luma, luma], axis=-1)
    elif not wrote_color and wrote_alpha:
        color = np.stack([alpha, alpha, alpha], axis=-1)
    if wrote_alpha and (wrote_color or luma is not None):
        rgba = np.dstack([color, alpha])
        return composite_on_checkerboard(rgba)
    return color


def _masked_channel(
    channel: NDArray[np.uint16],
    bits: tuple[int, ...],
) -> NDArray[np.uint16]:
    mask = np.uint16(mask_of(bits))
    return channel & mask


def _scale_to_byte(
    value: NDArray[np.uint16] | NDArray[np.uint32] | NDArray[np.uint64],
    maximum: int,
) -> NDArray[np.uint8]:
    if maximum <= 0:
        return np.zeros(value.shape, dtype=np.uint8)
    if maximum == 255 and value.dtype == np.uint8:
        return value.astype(np.uint8)
    return (value.astype(np.uint32) * np.uint32(255) // np.uint32(maximum)).astype(np.uint8)

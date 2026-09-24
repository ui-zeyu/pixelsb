"""Bit planes and preview compositing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    BitChoice,
    LoadedImage,
    RgbArray,
    SampleArray,
    SamplePlane,
)
from pixelsb.domain.selection import bits_for, effective_selection

type IndexArray = NDArray[np.intp]
type ChannelArray = NDArray[np.uint8]

_CHECKER_DARK = 232
_CHECKER_LIGHT = 255
# Where a plane's bytes land in the rendered image.
_COLOR_SLOTS = ("R", "G", "B")
_GRAY_SLOTS = ("L", "Index")


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """One render: which bits of which planes, over which pixels.

    ``rows`` and ``columns`` are the source indices when the render is of a
    gathered sub-block rather than the whole image; the checkerboard pattern
    needs them to keep its phase.
    """

    samples: SampleArray
    planes: tuple[SamplePlane, ...]
    chosen: frozenset[BitChoice]
    height: int
    width: int
    rows: IndexArray | None = None
    columns: IndexArray | None = None


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


def render_rgb(
    image: LoadedImage,
    selection: frozenset[BitChoice] | None,
) -> RgbArray:
    """Selected bits keep their original channel and weight. One bit renders as a bitmap."""
    return _render(
        RenderRequest(
            samples=image.samples,
            planes=image.planes,
            chosen=effective_selection(image, selection),
            height=image.height,
            width=image.width,
        )
    )


def render_rgb_at(
    image: LoadedImage,
    selection: frozenset[BitChoice] | None,
    ys: IndexArray,
    xs: IndexArray,
) -> RgbArray:
    """Render only the pixels at the given row/column indices.

    Compacted views need a fraction of the pixels — gathering the samples first
    keeps the bit-plane work proportional to the view, not the image.
    """
    return _render(
        RenderRequest(
            samples=image.samples[np.ix_(ys, xs)],
            planes=image.planes,
            chosen=effective_selection(image, selection),
            height=ys.size,
            width=xs.size,
            rows=ys,
            columns=xs,
        )
    )


def _render(request: RenderRequest) -> RgbArray:
    if not request.chosen:
        return np.zeros((request.height, request.width, 3), dtype=np.uint8)
    if len(request.chosen) == 1:
        choice = next(iter(request.chosen))
        gray = bit_plane(request.samples, _plane(request.planes, choice.plane), choice.bit)
        return np.stack([gray, gray, gray], axis=-1)
    return _compose(request)


def _compose(request: RenderRequest) -> RgbArray:
    """Assemble the channels the selection touches, in plane order."""
    scaled = _scaled_planes(request)
    color = scaled.get("R"), scaled.get("G"), scaled.get("B")
    # A gray plane stands in for the whole image, as it does on screen.
    gray = next((value for name, value in scaled.items() if name in _GRAY_SLOTS), None)
    alpha = scaled.get("A")
    if any(channel is not None for channel in color):
        blank = np.zeros((request.height, request.width), dtype=np.uint8)
        painted = np.stack([blank if channel is None else channel for channel in color], axis=-1)
    elif gray is not None:
        painted = np.stack([gray, gray, gray], axis=-1)
    elif alpha is not None:
        # Alpha alone reads as a gray image rather than compositing with itself.
        painted = np.stack([alpha, alpha, alpha], axis=-1)
    else:
        return np.zeros((request.height, request.width, 3), dtype=np.uint8)
    if alpha is None or (gray is None and not any(channel is not None for channel in color)):
        return painted
    return composite_on_checkerboard(
        np.dstack([painted, alpha]), rows=request.rows, columns=request.columns
    )


def _scaled_planes(request: RenderRequest) -> dict[str, ChannelArray]:
    """One scaled byte array per plane the selection touches."""
    return {
        plane.name: _scale_to_byte(*_masked_channel(request.samples[:, :, plane.index], bits))
        for plane in request.planes
        if (bits := bits_for(request.chosen, plane.name))
    }


def _plane(planes: tuple[SamplePlane, ...], name: str) -> SamplePlane:
    for plane in planes:
        if plane.name == name:
            return plane
    raise KeyError(name)


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
    mask = sum(1 << bit for bit in bits)
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

"""Text drawn inside each image pixel once the zoom can hold it."""

import math
from functools import cache

import numpy as np

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.models import (
    MIN_ZOOM,
    DisplayFormat,
    Raster,
    SampleArray,
)
from pixelsb.domain.selection import bits_for, mask_of

LABEL_PAD = 2
FONT_FILL = 0.72
ADVANCE = 0.68
LINE_SPACING = 1.25
MIN_FONT = 5


def font_pixel_size(zoom: float, text: str) -> int:
    if not text:
        return MIN_FONT
    lines = text.split("\n")
    longest = max(len(line) for line in lines)
    cap = max(int(zoom * FONT_FILL), 1)
    by_width = max(int((zoom - LABEL_PAD) / (ADVANCE * longest) + 1e-9), 1)
    by_height = max(int((zoom - LABEL_PAD) / (LINE_SPACING * len(lines)) + 1e-9), 1)
    return min(cap, by_width, by_height)


def label_fits(zoom: float, text: str) -> bool:
    if not text:
        return False
    return font_pixel_size(zoom, text) >= MIN_FONT


def zoom_required(text: str) -> int:
    if not text:
        return int(MIN_ZOOM)
    lines = text.split("\n")
    longest = max(len(line) for line in lines)
    by_width = math.ceil(MIN_FONT * ADVANCE * longest) + LABEL_PAD
    by_height = math.ceil(MIN_FONT * LINE_SPACING * len(lines)) + LABEL_PAD
    cap = math.ceil(MIN_FONT / FONT_FILL)
    return int(max(MIN_ZOOM, by_width, by_height, cap))


def region_texts(
    raster: Raster,
    fmt: DisplayFormat,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[str]:
    """Labels for a rectangle of the raster's cells, row-major. One line per channel.

    The coordinates are cells, not source pixels: a cropped raster's labels read
    what it drew, which is the state the canvas paints from.
    """
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, raster.width)
    y1 = min(y1, raster.height)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return []
    return _texts(raster, fmt, raster.samples[y0:y1, x0:x1])


def _texts(raster: Raster, fmt: DisplayFormat, region: SampleArray) -> list[str]:
    """Rendered labels for one gathered block of pixels, row-major."""
    if not raster.selection:
        return [""] * (region.shape[0] * region.shape[1])
    active = [plane for plane in raster.planes if bits_for(raster.selection, plane.name)]
    joined: list[str] | None = None
    for plane in active:
        bits = bits_for(raster.selection, plane.name)
        channel = region[:, :, plane.index]
        if len(bits) == 1:
            shown = (channel >> np.uint16(bits[0])) & np.uint16(1)
            depth = 1
        else:
            shown = channel & np.uint16(mask_of(bits))
            depth = max(bits) + 1
        table = _format_table(depth, fmt)
        strings = list(map(table.__getitem__, shown.ravel().tolist()))
        if len(active) > 1:
            name = plane.name + ":"
            strings = [name + text for text in strings]
        joined = (
            strings
            if joined is None
            else [a + "\n" + b for a, b in zip(joined, strings, strict=True)]
        )
    assert joined is not None
    return joined


@cache
def _format_table(depth: int, fmt: DisplayFormat) -> tuple[str, ...]:
    """One rendered string per value of a ``depth``-bit field, reused across paints."""
    return tuple(format_sample(value, depth, fmt) for value in range(1 << depth))


def widest_text(raster: Raster, fmt: DisplayFormat) -> str:
    """The longest label the selection can produce, used to fit the font."""
    if not raster.selection:
        return ""
    active = [plane for plane in raster.planes if bits_for(raster.selection, plane.name)]
    parts: list[str] = []
    for plane in active:
        shown, depth = _largest_value(bits_for(raster.selection, plane.name))
        rendered = format_sample(shown, depth, fmt)
        parts.append(rendered if len(active) == 1 else f"{plane.name}:{rendered}")
    return "\n".join(parts)


def _largest_value(bits: tuple[int, ...]) -> tuple[int, int]:
    """The largest number the selected bits can show, and how many bits it spans."""
    if len(bits) == 1:
        return 1, 1
    return mask_of(bits), max(bits) + 1

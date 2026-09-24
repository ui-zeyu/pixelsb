"""Text drawn inside each image pixel once the zoom can hold it."""

import math
from functools import cache

import numpy as np

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.models import MIN_ZOOM, DisplayFormat, PixelCoord, ViewerState
from pixelsb.domain.selection import bits_for, effective_selection, mask_of, number_bits

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


def pixel_text(state: ViewerState, coord: PixelCoord) -> str:
    texts = region_texts(state, coord.x, coord.y, coord.x + 1, coord.y + 1)
    return texts[0] if texts else ""


def region_texts(state: ViewerState, x0: int, y0: int, x1: int, y1: int) -> list[str]:
    """Labels for a rectangle of pixels, row-major. One line per active channel."""
    image = state.image
    if image is None:
        return []
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image.width)
    y1 = min(y1, image.height)
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return []
    chosen = effective_selection(image, number_bits(state))
    if not chosen:
        return [""] * (width * height)
    region = image.samples[y0:y1, x0:x1]
    active = [plane for plane in image.planes if bits_for(chosen, plane.name)]
    joined: list[str] | None = None
    for plane in active:
        bits = bits_for(chosen, plane.name)
        channel = region[:, :, plane.index]
        if len(bits) == 1:
            shown = (channel >> np.uint16(bits[0])) & np.uint16(1)
            depth = 1
        else:
            shown = channel & np.uint16(mask_of(bits))
            depth = max(bits) + 1
        table = _format_table(depth, state.value_format)
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


def widest_text(state: ViewerState) -> str:
    """The longest label the number layer can produce, used to fit the font."""
    image = state.image
    if image is None:
        return ""
    chosen = effective_selection(image, number_bits(state))
    if not chosen:
        return ""
    active = [plane for plane in image.planes if bits_for(chosen, plane.name)]
    parts: list[str] = []
    for plane in active:
        shown, depth = _largest_value(bits_for(chosen, plane.name))
        rendered = format_sample(shown, depth, state.value_format)
        parts.append(rendered if len(active) == 1 else f"{plane.name}:{rendered}")
    return "\n".join(parts)


def _largest_value(bits: tuple[int, ...]) -> tuple[int, int]:
    """The largest number the selected bits can show, and how many bits it spans."""
    if len(bits) == 1:
        return 1, 1
    return mask_of(bits), max(bits) + 1

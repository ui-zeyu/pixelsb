"""Text drawn inside each image pixel once the zoom can hold it."""

import math
from collections.abc import Callable

import numpy as np

from pixelsb.domain.formatting import format_delta, format_sample
from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    BitChoice,
    DisplayFormat,
    LoadedImage,
    PixelCoord,
    ValueMode,
    ViewerState,
)
from pixelsb.domain.selection import bits_for, effective_selection, mask_of

LABEL_PAD = 2
FONT_FILL = 0.72
ADVANCE = 0.68
LINE_SPACING = 1.25
MIN_FONT = 5


def font_pixel_size(zoom: int, text: str) -> int:
    if not text:
        return MIN_FONT
    lines = text.split("\n")
    longest = max(len(line) for line in lines)
    cap = max(int(zoom * FONT_FILL), 1)
    by_width = max(int((zoom - LABEL_PAD) / (ADVANCE * longest)), 1)
    by_height = max(int((zoom - LABEL_PAD) / (LINE_SPACING * len(lines))), 1)
    return min(cap, by_width, by_height)


def label_fits(zoom: int, text: str) -> bool:
    if not text:
        return False
    return font_pixel_size(zoom, text) >= MIN_FONT


def zoom_required(text: str) -> int:
    if not text:
        return MIN_ZOOM
    lines = text.split("\n")
    longest = max(len(line) for line in lines)
    by_width = math.ceil(MIN_FONT * ADVANCE * longest) + LABEL_PAD
    by_height = math.ceil(MIN_FONT * LINE_SPACING * len(lines)) + LABEL_PAD
    cap = math.ceil(MIN_FONT / FONT_FILL)
    return max(MIN_ZOOM, by_width, by_height, cap)


def zoom_to_fit(text: str, *, cap: int = MAX_ZOOM) -> int:
    return min(zoom_required(text), cap)


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
    chosen = effective_selection(image, state.selection)
    if not chosen:
        return [""] * (width * height)
    offset = state.value_mode is ValueMode.OFFSET and state.anchor is not None
    anchor_row = _row(image, state.anchor) if offset and state.anchor is not None else None
    region = image.samples[y0:y1, x0:x1]
    active = [plane for plane in image.planes if bits_for(chosen, plane.name)]
    joined: list[str] | None = None
    for plane in active:
        bits = bits_for(chosen, plane.name)
        mask = mask_of(bits)
        depth = max(bits) + 1 if len(bits) > 1 else 1
        shown = region[:, :, plane.index].astype(np.uint32) & np.uint32(mask)
        if offset and anchor_row is not None:
            anchor_shown = int(anchor_row[plane.index]) & mask
            deltas = shown.astype(np.int64) - anchor_shown
            memo = _FormatMemo(depth, state.value_format, offset=True)
            strings = [memo[value] for value in deltas.ravel().tolist()]
        else:
            memo = _FormatMemo(depth, state.value_format, offset=False)
            strings = [memo[value] for value in shown.ravel().tolist()]
        if len(active) > 1:
            name = plane.name
            strings = [name + text for text in strings]
        joined = (
            strings
            if joined is None
            else [a + "\n" + b for a, b in zip(joined, strings, strict=True)]
        )
    assert joined is not None
    return joined


class _FormatMemo(dict[int, str]):
    def __init__(self, depth: int, fmt: DisplayFormat, *, offset: bool) -> None:
        super().__init__()
        self._depth = depth
        self._fmt = fmt
        self._offset = offset

    def __missing__(self, value: int) -> str:
        if self._offset:
            text = format_delta(value, self._depth, self._fmt)
        else:
            text = format_sample(value, self._depth, self._fmt)
        self[value] = text
        return text


def widest_text(state: ViewerState) -> str:
    image = state.image
    if image is None:
        return ""
    chosen = effective_selection(image, state.selection)
    if not chosen:
        return ""
    offset = state.value_mode is ValueMode.OFFSET and state.anchor is not None
    return _join_channels(
        image,
        chosen,
        state.value_format,
        offset=offset,
        value_at=_maximum_channel_value,
    )


def _join_channels(
    image: LoadedImage,
    chosen: frozenset[BitChoice],
    fmt: DisplayFormat,
    *,
    offset: bool,
    value_at: Callable[[int, tuple[int, ...]], tuple[int, int]],
) -> str:
    active = [plane for plane in image.planes if bits_for(chosen, plane.name)]
    parts: list[str] = []
    for plane in active:
        bits = bits_for(chosen, plane.name)
        shown, depth = value_at(plane.index, bits)
        rendered = _format_value(shown, depth, fmt, offset=offset)
        if len(active) == 1:
            parts.append(rendered)
        else:
            parts.append(f"{plane.name}{rendered}")
    return "\n".join(parts)


def _maximum_channel_value(plane_index: int, bits: tuple[int, ...]) -> tuple[int, int]:
    del plane_index
    if len(bits) == 1:
        return 1, 1
    return mask_of(bits), max(bits) + 1


def _format_value(shown: int, depth: int, fmt: DisplayFormat, *, offset: bool) -> str:
    if offset:
        return format_delta(shown, depth, fmt)
    return format_sample(shown, depth, fmt)


def _row(image: LoadedImage, coord: PixelCoord) -> tuple[int, ...]:
    return tuple(int(sample) for sample in image.samples[coord.y, coord.x])

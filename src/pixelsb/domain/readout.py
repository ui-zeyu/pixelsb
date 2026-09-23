"""Cursor and anchor readout. Pixel numbers live on the image itself."""

from dataclasses import dataclass

from pixelsb.domain.formatting import format_delta, format_sample
from pixelsb.domain.geometry import spatial_offset
from pixelsb.domain.labels import pixel_text, widest_text, zoom_required
from pixelsb.domain.models import BitChoice, LoadedImage, PixelCoord, SampleOrigin, ViewerState
from pixelsb.domain.selection import bits_for, effective_selection, number_bits, shown_channel_value


@dataclass(frozen=True, slots=True)
class ChannelReadout:
    name: str
    absolute: str
    relative: str | None
    origin: SampleOrigin


@dataclass(frozen=True, slots=True)
class Readout:
    cursor: PixelCoord
    anchor: PixelCoord | None
    dx: int | None
    dy: int | None
    channels: tuple[ChannelReadout, ...]
    summary: str


def layer_summary(state: ViewerState) -> str:
    """One line saying which bits drive the canvas and which drive the numbers."""
    image = state.image
    if image is None:
        return ""
    canvas = _layer_text(image, state.selection, original="原图")
    numbers = _layer_text(image, state.readout, original="原始值")
    return f"画面 {canvas} · 数字 {numbers}"


def _layer_text(
    image: LoadedImage,
    chosen: frozenset[BitChoice] | None,
    *,
    original: str,
) -> str:
    if chosen is None:
        return original
    if not chosen:
        return "未选择"
    parts: list[str] = []
    for plane in image.planes:
        for bit in bits_for(chosen, plane.name):
            parts.append(f"{plane.name}{bit}")
    return " ".join(parts)


def build_readout(state: ViewerState) -> Readout | None:
    image = state.image
    cursor = state.cursor
    if image is None or cursor is None:
        return None
    anchor = state.anchor
    dx: int | None = None
    dy: int | None = None
    if anchor is not None:
        dx, dy = spatial_offset(anchor, cursor)
    return Readout(
        cursor=cursor,
        anchor=anchor,
        dx=dx,
        dy=dy,
        channels=_channels(state),
        summary=layer_summary(state),
    )


def cursor_label(state: ViewerState) -> str:
    if state.cursor is None:
        return ""
    return pixel_text(state, state.cursor)


def label_zoom(state: ViewerState) -> int | None:
    text = widest_text(state)
    if not text:
        return None
    return zoom_required(text)


def _channels(state: ViewerState) -> tuple[ChannelReadout, ...]:
    image = state.image
    cursor = state.cursor
    if image is None or cursor is None:
        return ()
    chosen = effective_selection(image, number_bits(state))
    if not chosen:
        return ()
    cursor_row = tuple(int(sample) for sample in image.samples[cursor.y, cursor.x])
    anchor_row = None
    if state.anchor is not None:
        anchor_row = tuple(int(sample) for sample in image.samples[state.anchor.y, state.anchor.x])
    rows: list[ChannelReadout] = []
    for plane in image.planes:
        bits = bits_for(chosen, plane.name)
        if not bits:
            continue
        shown, depth = shown_channel_value(cursor_row[plane.index], bits)
        relative = None
        if anchor_row is not None:
            anchor_shown, _depth = shown_channel_value(anchor_row[plane.index], bits)
            relative = format_delta(shown - anchor_shown, depth, state.value_format)
        rows.append(
            ChannelReadout(
                name=plane.name,
                absolute=format_sample(shown, depth, state.value_format),
                relative=relative,
                origin=plane.origin,
            )
        )
    return tuple(rows)

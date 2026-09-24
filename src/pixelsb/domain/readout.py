"""Cursor readout. Pixel numbers live on the image itself."""

from dataclasses import dataclass

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.labels import pixel_text, widest_text, zoom_required
from pixelsb.domain.models import BitChoice, LoadedImage, PixelCoord, SampleOrigin, ViewerState
from pixelsb.domain.selection import bits_for, effective_selection, shown_channel_value


@dataclass(frozen=True, slots=True)
class ChannelReadout:
    name: str
    absolute: str
    origin: SampleOrigin


@dataclass(frozen=True, slots=True)
class Readout:
    cursor: PixelCoord
    channels: tuple[ChannelReadout, ...]
    summary: str


def layer_summary(state: ViewerState) -> str:
    """One line saying which bits drive the canvas and its numbers."""
    image = state.image
    if image is None:
        return ""
    return f"画面 {_layer_text(image, state.selection, original='原图')}"


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
    return " ".join(
        f"{plane.name}{bit}" for plane in image.planes for bit in bits_for(chosen, plane.name)
    )


def build_readout(state: ViewerState) -> Readout | None:
    image = state.image
    cursor = state.cursor
    if image is None or cursor is None:
        return None
    return Readout(
        cursor=cursor,
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
    chosen = effective_selection(image, state.selection)
    if not chosen:
        return ()
    cursor_row = tuple(image.samples[cursor.y, cursor.x].tolist())
    rows: list[ChannelReadout] = []
    for plane in image.planes:
        bits = bits_for(chosen, plane.name)
        if not bits:
            continue
        shown, depth = shown_channel_value(cursor_row[plane.index], bits)
        rows.append(
            ChannelReadout(
                name=plane.name,
                absolute=format_sample(shown, depth, state.value_format),
                origin=plane.origin,
            )
        )
    return tuple(rows)

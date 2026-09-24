"""Cursor readout: the numbers of the pixel under the cursor."""

from dataclasses import dataclass

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.labels import widest_text, zoom_required
from pixelsb.domain.models import BitChoice, LoadedImage, PixelCoord, ViewerState
from pixelsb.domain.selection import bits_for, effective_selection, shown_channel_value


@dataclass(frozen=True, slots=True)
class ChannelValue:
    """One plane's number for the cursor pixel, already formatted."""

    name: str
    text: str


@dataclass(frozen=True, slots=True)
class Readout:
    cursor: PixelCoord
    channels: tuple[ChannelValue, ...]
    bits: tuple[BitChoice, ...] | None  # None: every bit, that is the original image


def build_readout(state: ViewerState) -> Readout | None:
    """The cursor pixel's numbers, or ``None`` while no pixel is under the cursor."""
    image = state.image
    cursor = state.cursor
    if image is None or cursor is None:
        return None
    chosen = effective_selection(image, state.selection)
    return Readout(
        cursor=cursor,
        channels=_channels(image, cursor, chosen, state),
        bits=None if state.selection is None else active_bits(image, chosen),
    )


def active_bits(image: LoadedImage, chosen: frozenset[BitChoice]) -> tuple[BitChoice, ...]:
    """The selected bits in plane order, then low to high."""
    return tuple(
        BitChoice(plane.name, bit) for plane in image.planes for bit in bits_for(chosen, plane.name)
    )


def label_zoom(state: ViewerState) -> int | None:
    """The zoom at which pixel numbers become legible, or ``None`` when none do."""
    text = widest_text(state)
    if not text:
        return None
    return zoom_required(text)


def _channels(
    image: LoadedImage,
    cursor: PixelCoord,
    chosen: frozenset[BitChoice],
    state: ViewerState,
) -> tuple[ChannelValue, ...]:
    sample = tuple(image.samples[cursor.y, cursor.x].tolist())
    rows: list[ChannelValue] = []
    for plane in image.planes:
        bits = bits_for(chosen, plane.name)
        if not bits:
            continue
        shown, depth = shown_channel_value(sample[plane.index], bits)
        rows.append(ChannelValue(plane.name, format_sample(shown, depth, state.value_format)))
    return tuple(rows)

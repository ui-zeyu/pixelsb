"""Cursor readout: the numbers of the pixel under the cursor."""

from dataclasses import dataclass

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.labels import widest_text, zoom_required
from pixelsb.domain.models import BitChoice, DisplayFormat, PixelCoord, Raster, ViewerState
from pixelsb.domain.selection import active_bits, bits_for, shown_channel_value, whole


@dataclass(frozen=True, slots=True)
class ChannelValue:
    """One plane's number for the cursor pixel, already formatted."""

    name: str
    text: str


@dataclass(frozen=True, slots=True)
class Readout:
    cursor: PixelCoord
    channels: tuple[ChannelValue, ...]
    bits: tuple[BitChoice, ...] | None  # None: every bit of every plane


def build_readout(state: ViewerState, raster: Raster) -> Readout | None:
    """The cursor pixel's numbers, or ``None`` while the raster does not show it."""
    cursor = state.cursor
    if cursor is None:
        return None
    cell = raster.cell_of(cursor)
    if cell is None:
        return None
    column, row = cell
    return Readout(
        cursor=cursor,
        channels=_channels(raster, row, column, state.value_format),
        bits=None
        if whole(raster.planes, raster.selection)
        else active_bits(raster.planes, raster.selection),
    )


def label_zoom(state: ViewerState, raster: Raster) -> int | None:
    """The zoom at which pixel numbers become legible, or ``None`` when none do."""
    text = widest_text(raster, state.value_format)
    if not text:
        return None
    return zoom_required(text)


def _channels(
    raster: Raster,
    row: int,
    column: int,
    fmt: DisplayFormat,
) -> tuple[ChannelValue, ...]:
    sample = tuple(raster.samples[row, column].tolist())
    values: list[ChannelValue] = []
    for plane in raster.planes:
        bits = bits_for(raster.selection, plane.name)
        if not bits:
            continue
        shown, depth = shown_channel_value(sample[plane.index], bits)
        values.append(ChannelValue(plane.name, format_sample(shown, depth, fmt)))
    return tuple(values)

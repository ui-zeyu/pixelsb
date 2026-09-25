"""StegSolve-style bit extraction: selected bits packed into bytes."""

import codecs
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import batched, permutations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain import bytes_text
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    ExtractEncoding,
    ExtractOrder,
    LoadedImage,
    SampleArray,
    SamplePlane,
    ScanOrder,
)
from pixelsb.domain.selection import bits_for, effective_selection

DISPLAY_LINES = 4096
BYTES_PER_ROW = 16
HEX_WIDTH = BYTES_PER_ROW * 3 - 1
ASCII_START = HEX_WIDTH + 2
_LOOKBACK = 4  # bytes the longest character of a supported encoding can span

# The order every caller gets unless it asks for another one. Shared rather than
# built per default: it is frozen, so one instance serves every call.
DEFAULT_ORDER = ExtractOrder()

_PACKBIT_ORDER: dict[BitOrder, Literal["big", "little"]] = {
    BitOrder.MSB: "big",
    BitOrder.LSB: "little",
}
_MAX_ORDER_PLANES = 3  # StegSolve's set: every arrangement of three channels


@dataclass(frozen=True, slots=True)
class ExtractRow:
    """One dump row: hex and ASCII, with the offset kept beside the text.

    ``data`` is kept so the panel can re-read the row in another encoding, and
    note rows (no data, truncated output) carry no offset and no bytes.
    """

    offset: int | None
    text: str
    data: bytes = b""

    @property
    def offset_text(self) -> str:
        return "" if self.offset is None else f"{self.offset:08x}"

    @property
    def hex_text(self) -> str:
        """The hex column; a note row keeps its whole text in the hex pane."""
        return self.text if self.offset is None else self.text[:HEX_WIDTH]


def extract_bytes(
    image: LoadedImage,
    chosen: frozenset[BitChoice] | None,
    match: NDArray[np.bool_] | None = None,
    *,
    order: ExtractOrder = DEFAULT_ORDER,
) -> bytes:
    """Pack the selected bits into bytes, in the requested order.

    The stream walks the pixels in ``order.scan``, each pixel's channels in
    ``order.planes``, and each channel's selected bits from low to high; every
    eight bits become a byte, the first of them the high or low end of it.
    ``match`` (HxW bool) limits the stream to the pixels that pass the display
    filter; surviving pixels keep their bit order and are re-packed densely.
    """
    selection = effective_selection(image, chosen)
    if not selection:
        return b""
    # Select the surviving pixels first: expanding every bit plane of a large
    # image just to discard most of it is slow and allocation heavy.
    rows = _pixel_rows(image.samples, match, order.scan)
    columns = [
        _bit_column(rows, plane.index, bit)
        for plane in ordered_planes(image.planes, order)
        for bit in bits_for(selection, plane.name)
    ]
    if not columns:
        return b""
    stream = np.stack(columns, axis=-1)
    return np.packbits(stream.reshape(-1), bitorder=_PACKBIT_ORDER[order.bit_order]).tobytes()


def ordered_planes(
    planes: tuple[SamplePlane, ...],
    order: ExtractOrder,
) -> tuple[SamplePlane, ...]:
    """The planes in the order the stream reads them: the ranked names first.

    Names the order does not mention sort into the image's own order after every
    ranked one, which keeps a preference like ``("B", "G", "R")`` meaningful for
    images that hold other planes as well.
    """
    if not order.planes:
        return planes
    rank = {name: position for position, name in enumerate(order.planes)}
    return tuple(sorted(planes, key=lambda plane: rank.get(plane.name, len(rank) + plane.index)))


def applied_order(
    planes: tuple[SamplePlane, ...],
    chosen: frozenset[BitChoice],
    order: ExtractOrder,
) -> tuple[str, ...]:
    """The channel order in effect: the planes carrying bits, as the stream reads them."""
    return tuple(
        plane.name for plane in ordered_planes(planes, order) if bits_for(chosen, plane.name)
    )


def order_choices(
    planes: tuple[SamplePlane, ...],
    chosen: frozenset[BitChoice],
) -> tuple[tuple[str, ...], ...]:
    """The channel orders worth offering: the arrangements of the planes in use.

    Three channels or fewer come in every arrangement, which is StegSolve's set;
    beyond that the order itself and its reverse keep the list readable.
    """
    used = tuple(plane.name for plane in planes if bits_for(chosen, plane.name))
    if not used:
        return ()
    if len(used) > _MAX_ORDER_PLANES:
        return used, tuple(reversed(used))
    return tuple(permutations(used))


def _pixel_rows(
    samples: SampleArray,
    match: NDArray[np.bool_] | None,
    scan: ScanOrder,
) -> SampleArray:
    """The pixels to read, in scan order: rows first (XY) or columns first (YZ).

    A transposed view reads the same samples with the axes swapped, and the
    filter travels with it, so the surviving pixels keep column order. The
    reshape of that view is the one copy this order costs.
    """
    if scan is ScanOrder.YZ:
        samples = samples.transpose(1, 0, 2)
        match = None if match is None else match.T
    if match is None:
        return samples.reshape(-1, samples.shape[2])
    return samples[match]


def _bit_column(rows: SampleArray, index: int, bit: int) -> NDArray[np.uint8]:
    """One bit plane of one channel, as a stream of 0 and 1 bytes."""
    return ((rows[:, index] >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8)


def format_extract(data: bytes, limit: int = DISPLAY_LINES, base: int = 0) -> list[ExtractRow]:
    """Rows of hex and ASCII; the offset stays out of ``text`` for the gutter.

    ``base`` is the stream position of the first byte, so a dump of one block
    shows the offsets that byte has inside the file.
    """
    shown = data[: limit * BYTES_PER_ROW]
    rows = [
        ExtractRow(base + index * BYTES_PER_ROW, _row_text(bytes(chunk)), bytes(chunk))
        for index, chunk in enumerate(batched(shown, BYTES_PER_ROW, strict=False))
    ]
    if not rows:
        rows.append(ExtractRow(None, "（无数据）"))
    elif len(data) > len(shown):
        rows.append(ExtractRow(None, f"…已截断，共 {len(data)} 字节"))
    return rows


def decode_row(row: ExtractRow, encoding: ExtractEncoding) -> str:
    """One row's bytes as text: printable characters, a dot for everything else.

    Each row decodes on its own, which is right for a byte-per-cell encoding and
    for a row read in isolation: a multi-byte character split by the row
    boundary shows as a replacement there. ``decode_rows`` is what a whole dump
    uses, so a boundary does not break a character.
    """
    if not row.data:
        return ""
    if encoding is ExtractEncoding.ASCII:
        text = bytes_text.as_text(row.data)
    else:
        text = row.data.decode(encoding.value, errors="replace")
    return "".join(char if char.isprintable() else "." for char in text)


def decode_rows(rows: Sequence[ExtractRow], encoding: ExtractEncoding) -> list[str]:
    """One text line per row, characters read across the rows' own boundaries.

    A character whose bytes straddle two rows is shown whole, in the row that
    finishes it, so a comment in the dump reads as written; only a genuinely
    invalid sequence becomes a replacement. Rows are only treated as neighbours
    when their offsets really are adjacent, so a search that hides rows cannot
    glue together bytes that were never next to each other. ASCII needs none of
    this: one byte is one cell, row or no row.
    """
    if encoding is ExtractEncoding.ASCII:
        return [decode_row(row, encoding) for row in rows]
    decoder = codecs.getincrementaldecoder(encoding.value)
    return [
        _decode_joined(decoder, row, rows[index - 1] if index else None)
        for index, row in enumerate(rows)
    ]


def _decode_joined(
    decoder: Callable[..., codecs.IncrementalDecoder],
    row: ExtractRow,
    previous: ExtractRow | None,
) -> str:
    """One row of a dump: what its bytes add to the text the row above left off."""
    if not row.data or row.offset is None:
        return ""
    incremental = decoder(errors="replace")
    preceding = _preceding(previous, row)
    if preceding:
        # The lookback only finishes characters the row above already showed.
        incremental.decode(preceding[-_LOOKBACK:])
    text = incremental.decode(row.data)
    return "".join(char if char.isprintable() else "." for char in text)


def _preceding(previous: ExtractRow | None, row: ExtractRow) -> bytes:
    """The bytes the row above ends with, when it really is the row above.

    A row with no offset is a note, and rows whose offsets do not touch are not
    neighbours at all — a search that hides rows is the usual reason.
    """
    if previous is None or previous.offset is None or not previous.data:
        return b""
    if previous.offset + len(previous.data) != row.offset:
        return b""
    return previous.data


def _row_text(chunk: bytes) -> str:
    hex_part = chunk.hex(" ")
    return f"{hex_part:<{HEX_WIDTH}}  {bytes_text.as_text(chunk)}"


def filter_extract(rows: list[ExtractRow], query: str) -> list[ExtractRow]:
    """Rows whose hex, ASCII, or offset text contains ``query``."""
    query = query.strip().lower()
    if not query:
        return rows
    return [row for row in rows if query in row.text.lower() or query in row.offset_text]

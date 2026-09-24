"""StegSolve-style bit extraction: selected bits packed into bytes."""

from dataclasses import dataclass
from itertools import batched, permutations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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

# The order every caller gets unless it asks for another one. Shared rather than
# built per default: it is frozen, so one instance serves every call.
DEFAULT_ORDER = ExtractOrder()

_PACKBIT_ORDER: dict[BitOrder, Literal["big", "little"]] = {
    BitOrder.MSB: "big",
    BitOrder.LSB: "little",
}
_MAX_ORDER_PLANES = 3  # StegSolve's set: every arrangement of three channels

# Byte value -> itself when printable, a dot otherwise, for the ASCII column.
_DOT_TABLE = bytes(byte if 32 <= byte <= 126 else ord(".") for byte in range(256))


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


def format_extract(data: bytes, limit: int = DISPLAY_LINES) -> list[ExtractRow]:
    """Rows of hex and ASCII; the offset stays out of ``text`` for the gutter."""
    shown = data[: limit * BYTES_PER_ROW]
    rows = [
        ExtractRow(index * BYTES_PER_ROW, _row_text(bytes(chunk)), bytes(chunk))
        for index, chunk in enumerate(batched(shown, BYTES_PER_ROW, strict=False))
    ]
    if not rows:
        rows.append(ExtractRow(None, "（无数据）"))
    elif len(data) > len(shown):
        rows.append(ExtractRow(None, f"…已截断，共 {len(data)} 字节"))
    return rows


def decode_row(row: ExtractRow, encoding: ExtractEncoding) -> str:
    """One row's bytes as text: printable characters, a dot for everything else.

    Each row decodes on its own, like a hex editor's text column: a multi-byte
    character split by the row boundary shows as a replacement there.
    """
    if not row.data:
        return ""
    if encoding is ExtractEncoding.ASCII:
        text = row.data.translate(_DOT_TABLE).decode("ascii")
    else:
        text = row.data.decode(encoding.value, errors="replace")
    return "".join(char if char.isprintable() else "." for char in text)


def _row_text(chunk: bytes) -> str:
    hex_part = chunk.hex(" ")
    ascii_part = chunk.translate(_DOT_TABLE).decode("ascii")
    return f"{hex_part:<{HEX_WIDTH}}  {ascii_part}"


def filter_extract(rows: list[ExtractRow], query: str) -> list[ExtractRow]:
    """Rows whose hex, ASCII, or offset text contains ``query``."""
    query = query.strip().lower()
    if not query:
        return rows
    return [row for row in rows if query in row.text.lower() or query in row.offset_text]

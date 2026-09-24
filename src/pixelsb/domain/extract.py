"""StegSolve-style bit extraction: selected bits packed into bytes."""

from dataclasses import dataclass
from itertools import batched

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import BitChoice, ExtractEncoding, LoadedImage
from pixelsb.domain.selection import effective_selection

DISPLAY_LINES = 4096
BYTES_PER_ROW = 16
HEX_WIDTH = BYTES_PER_ROW * 3 - 1
ASCII_START = HEX_WIDTH + 2

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
) -> bytes:
    """Pack the selected bits into bytes, raster order, first bit into the MSB.

    ``match`` (HxW bool) limits the stream to the pixels that pass the display
    filter; surviving pixels keep their bit order and are re-packed densely.
    """
    selection = effective_selection(image, chosen)
    if not selection:
        return b""
    # Select the surviving pixels first: expanding every bit plane of a large
    # image just to discard most of it is slow and allocation heavy.
    samples = image.samples
    rows = samples[match] if match is not None else samples.reshape(-1, samples.shape[2])
    columns = [
        ((rows[:, plane.index] >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8)
        for plane in image.planes
        for bit in range(plane.bit_depth)
        if BitChoice(plane.name, bit) in selection
    ]
    stream = np.stack(columns, axis=-1)
    return np.packbits(stream.reshape(-1)).tobytes()


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

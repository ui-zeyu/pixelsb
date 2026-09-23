"""StegSolve-style bit extraction: selected bits packed into bytes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import BitChoice, LoadedImage
from pixelsb.domain.selection import effective_selection

DISPLAY_LINES = 4096
BYTES_PER_ROW = 16
HEX_WIDTH = BYTES_PER_ROW * 3 - 1
ASCII_START = HEX_WIDTH + 2


@dataclass(frozen=True, slots=True)
class ExtractRow:
    """One dump row: hex and ASCII, with the offset kept beside the text.

    Note rows (no data, truncated output) carry no offset.
    """

    offset: int | None
    text: str

    @property
    def offset_text(self) -> str:
        return "" if self.offset is None else f"{self.offset:08x}"


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
    planes: list[np.ndarray] = []
    for plane in image.planes:
        for bit in range(plane.bit_depth):
            if BitChoice(plane.name, bit) not in selection:
                continue
            channel = image.samples[:, :, plane.index]
            planes.append(((channel >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8))
    stream = np.stack(planes, axis=-1)
    if match is not None:
        stream = stream[match]
    return np.packbits(stream.reshape(-1)).tobytes()


def format_extract(data: bytes, limit: int = DISPLAY_LINES) -> list[ExtractRow]:
    """Rows of hex and ASCII; the offset stays out of ``text`` for the gutter."""
    rows: list[ExtractRow] = []
    for offset in range(0, len(data), BYTES_PER_ROW):
        chunk = data[offset : offset + BYTES_PER_ROW]
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in chunk)
        rows.append(ExtractRow(offset, f"{hex_part:<{HEX_WIDTH}}  {ascii_part}"))
        if len(rows) >= limit:
            rows.append(ExtractRow(None, f"…已截断，共 {len(data)} 字节"))
            return rows
    if not rows:
        rows.append(ExtractRow(None, "（无数据）"))
    return rows


def filter_extract(rows: list[ExtractRow], query: str) -> list[ExtractRow]:
    """Rows whose hex, ASCII, or offset text contains ``query``."""
    query = query.strip().lower()
    if not query:
        return rows
    return [row for row in rows if query in row.text.lower() or query in row.offset_text]

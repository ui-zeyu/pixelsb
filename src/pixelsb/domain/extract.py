"""StegSolve-style bit extraction: selected bits packed into bytes."""

import numpy as np

from pixelsb.domain.models import BitChoice, LoadedImage
from pixelsb.domain.selection import effective_selection

DISPLAY_LINES = 4096


def extract_bytes(image: LoadedImage, chosen: frozenset[BitChoice] | None) -> bytes:
    """Pack the selected bits into bytes, raster order, first bit into the MSB."""
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
    stream = np.stack(planes, axis=-1).reshape(-1)
    return np.packbits(stream).tobytes()


def format_extract(data: bytes, limit: int = DISPLAY_LINES) -> list[str]:
    """xxd-style rows: offset, hex bytes, ASCII. Long outputs are truncated."""
    lines: list[str] = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in chunk)
        lines.append(f"{offset:08x}  {hex_part:<47}  {ascii_part}")
        if len(lines) >= limit:
            lines.append(f"…已截断，共 {len(data)} 字节")
            return lines
    if not lines:
        lines.append("（无数据）")
    return lines


def filter_extract(lines: list[str], query: str) -> list[str]:
    query = query.strip().lower()
    if not query:
        return lines
    return [line for line in lines if query in line.lower()]

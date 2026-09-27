"""The pixels a PNG container's blocks would show: a tolerant block decoder.

The census (:mod:`pixelsb.domain.container`) answers what a file is built of;
this module answers what its bytes would look like rendered. The decoder is
deliberately forgiving, because it is a viewer and not a validator: short
streams are padded so a half-smuggled image still shows, an unknown filter byte
reconstructs as if no filter had been named, and a second zlib stream after the
first is ignored — the fake-IDAT shape, where the first stream is what a real
renderer would have shown.
"""

import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.container import CHANNELS, PngHeader
from pixelsb.domain.models import RgbaArray, RgbArray


def render_blocks(
    payloads: Sequence[bytes], header: PngHeader, palette: bytes = b""
) -> RgbArray | RgbaArray:
    """Render selected block payloads as the pixels they would show.

    The payloads are read in file order as one pixel stream: a zlib stream is
    decompressed (raw scanline bytes work too), short streams are padded so a
    half-smuggled image still shows, and the scanlines are reconstructed under
    the file's own IHDR geometry. Interlaced data is declined. A color type that
    carries alpha renders as HxWx4, so a transparent stream stays transparent.
    """
    if header.interlace != 0:
        raise ValueError("interlaced pixel data is not supported")
    data = b"".join(payloads)
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(data)
    except zlib.error:
        raw = data  # already-raw scanline bytes work too
    # A second stream after the first is the fake-IDAT shape: render stops at
    # the first stream, which is what the real image was made of.
    stride = header.pixel_stride()
    budget = header.pixel_budget()
    if stride is None or budget is None:
        raise ValueError("this color type has no single-stride layout")
    raw = raw[:budget].ljust(budget, b"\x00")
    rows = _unfilter(raw, header.height, stride, header.bytes_per_pixel())
    return _to_pixels(rows, header, palette)


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> NDArray[np.uint8]:
    """PNG filter reconstruction, one row at a time per the spec's five filters.

    Every reconstructed byte is taken modulo 256 as the spec says; without it a
    long Sub chain climbs past the accumulator and real images crash. Sub and Up
    reconstruct with a prefix sum — a chain that only ever adds is the same sum
    taken at once, modulo 256 — while Average and Paeth predict from the byte
    this row has just reconstructed, so those rows are walked byte by byte.
    """
    scanlines = np.frombuffer(raw, np.uint8, count=height * (stride + 1)).reshape(
        height, stride + 1
    )
    filters = scanlines[:, 0]
    recon = np.zeros((height, stride), dtype=np.uint8)
    for y in range(height):
        kind = int(filters[y])
        line = scanlines[y, 1:]
        above = recon[y - 1] if y else None
        if kind == 0 or kind > 4:
            recon[y] = line  # an unknown filter byte renders as None: a viewer, not a validator
        elif kind == 1:
            recon[y] = _sub(line, bpp)
        elif kind == 2:
            recon[y] = line if above is None else line + above  # uint8 adds modulo 256
        else:
            recon[y] = _predict(line, above, bpp, average=kind == 3)
    return recon


def _sub(line: NDArray[np.uint8], bpp: int) -> NDArray[np.uint8]:
    """Filter 1: each byte adds the reconstructed byte ``bpp`` to its left.

    Those chains never cross, so the row is one prefix sum per chain: the byte
    at ``x`` is the sum of its chain up to ``x``, modulo 256, which is exactly
    what adding one at a time lands on. Only a crafted geometry whose stride
    does not divide into whole chains needs the walk.
    """
    if bpp == 1:
        return np.cumsum(line, dtype=np.uint32).astype(np.uint8)
    if line.size % bpp:
        row = line.tolist()
        for x in range(bpp, len(row)):
            row[x] = (row[x] + row[x - bpp]) & 0xFF
        return np.asarray(row, dtype=np.uint8)
    return np.cumsum(line.reshape(-1, bpp), axis=0, dtype=np.uint32).astype(np.uint8).reshape(-1)


def _predict(
    line: NDArray[np.uint8],
    above: NDArray[np.uint8] | None,
    bpp: int,
    *,
    average: bool,
) -> NDArray[np.uint8]:
    """Filters 3 and 4: each byte predicts from its left and up neighbours.

    The left neighbour is this row's own previous byte, so the row is a chain
    that has to be walked in order; it is walked on plain ints, where a step
    costs a fraction of what an array cell does.
    """
    row = line.tolist()
    up = None if above is None else above.tolist()
    for x, byte in enumerate(row):
        left = row[x - bpp] if x >= bpp else 0
        high = up[x] if up is not None else 0
        if average:
            row[x] = (byte + (left + high) // 2) & 0xFF
        else:
            corner = up[x - bpp] if x >= bpp and up is not None else 0
            row[x] = (byte + _paeth(left, high, corner)) & 0xFF
    return np.asarray(row, dtype=np.uint8)


def _paeth(left: int, up: int, up_left: int) -> int:
    """The neighbour closest to the linear estimate, ties going left, then up."""
    estimate = left + up - up_left
    to_left = abs(estimate - left)
    to_up = abs(estimate - up)
    to_corner = abs(estimate - up_left)
    if to_left <= to_up and to_left <= to_corner:
        return left
    return up if to_up <= to_corner else up_left


def _to_pixels(rows: NDArray[np.uint8], header: PngHeader, palette: bytes) -> RgbArray | RgbaArray:
    """Packed scanlines in, HxWx3 or HxWx4 bytes out; extra channels drop, alpha stays."""
    width, depth, color = header.width, header.bit_depth, header.color_type
    channels = CHANNELS[color]
    if depth == 8:
        pixels = rows.reshape(header.height, width, channels)
    elif depth == 16:
        pixels = rows.reshape(header.height, -1).view(">u2")[:, : width * channels]
        pixels = (pixels.reshape(header.height, width, channels) >> 8).astype(np.uint8)
    else:
        bits = np.unpackbits(rows, axis=1)[:, : width * channels]
        pixels = bits.reshape(header.height, width, channels) * (255 // (2**depth - 1))
    match color:
        case 0:  # gray
            return _bytes(np.repeat(pixels, 3, axis=2))
        case 2:  # RGB
            return _bytes(pixels[..., :3])
        case 3:  # palette entries, which carry no alpha of their own
            entries = np.frombuffer(palette, np.uint8).reshape(-1, 3)
            if not len(entries):
                return _bytes(np.zeros((header.height, width, 3), np.uint8))
            indexes = np.clip(pixels[..., 0], 0, len(entries) - 1)
            return _bytes(entries[indexes])
        case 4:  # gray + alpha
            gray = np.repeat(pixels[..., :1], 3, axis=2)
            return _bytes(np.concatenate([gray, pixels[..., -1:]], axis=2))
        case _:  # RGBA
            return _bytes(np.concatenate([pixels[..., :3], pixels[..., -1:]], axis=2))


def _bytes(pixels: NDArray[np.uint8]) -> RgbArray | RgbaArray:
    return np.ascontiguousarray(pixels.astype(np.uint8))

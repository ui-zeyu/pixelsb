"""Container census: the blocks a file is built from, and what looks planted.

The census answers a structural question, not a semantic one: which bytes does
a renderer need, and what is merely there. Required sets come from the format
spec — for PNG, the spec's own critical/ancillary bit — so an unknown chunk is
still classified, and a smuggled second IDAT stream is still exposed, without
this module knowing what any chunk means. The census reads one container: the
first end marker ends the structure, and whatever follows travels as one tail
block, whatever it turns out to be. The census also keeps what a renderer would
need to show the blocks' bytes as pixels: the PNG header and palette.
"""

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain import bytes_text
from pixelsb.domain.models import RgbArray

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class BlockRole(StrEnum):
    """Whether a renderer needs a block: the spec decides, not the name."""

    REQUIRED = "required"
    ANCILLARY = "ancillary"


@dataclass(frozen=True, slots=True)
class PngHeader:
    """The IHDR parameters a block render needs for its geometry."""

    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace: int

    def pixel_budget(self) -> int | None:
        """How many decompressed bytes the image needs; None when it cannot be told."""
        stride = self.pixel_stride()
        if stride is None or self.interlace != 0:
            return None  # interlaced scans do not follow the single-stride size
        return self.height * (stride + 1)

    def pixel_stride(self) -> int | None:
        """Byte width of one reconstructed scanline, the filter byte excluded."""
        channels = _CHANNELS.get(self.color_type)
        if channels is None:
            return None
        return (self.width * self.bit_depth * channels + 7) // 8

    def bytes_per_pixel(self) -> int:
        """The filter unit: how many whole bytes one pixel occupies pre-unpacking."""
        channels = _CHANNELS.get(self.color_type, 1)
        return max(1, self.bit_depth * channels // 8)


@dataclass(frozen=True, slots=True)
class Block:
    """One structural unit: where it starts, what it holds, and the bytes themselves.

    Every block carries a short printable ``preview`` of its payload — the
    census does not interpret it, but an eye reading the panel can tell a
    comment from compressed pixels at a glance. Blocks that serve one stream
    share a ``group`` number, so a split IDAT reads as one unit instead of
    three coincidences. ``payload_at`` is where the payload starts in the file
    — behind a chunk's header, or at the block itself for raw residual.
    """

    offset: int
    label: str
    length: int
    role: BlockRole
    payload: bytes = b""
    preview: str = ""
    group: int = 0
    payload_at: int = 0


@dataclass(frozen=True, slots=True)
class Finding:
    """An anomaly: bytes no renderer will show, or structure no reader expects.

    ``payload`` carries the suspicious bytes when they are contiguous, so the
    panel can preview them and scan them for signatures without re-reading;
    ``decoded`` holds a nested stream (the usual fake-IDAT shape) unfolded, so
    what the renderer skips can still be read.
    """

    kind: str
    offset: int
    length: int = 0
    payload: bytes = b""
    decoded: bytes = b""


@dataclass(frozen=True, slots=True)
class ContainerReport:
    """The census of one container: every block, and everything suspicious.

    For PNG the ``header`` and ``palette`` are kept so the blocks' bytes can be
    rendered as pixels without touching the file again.
    """

    kind: str
    blocks: tuple[Block, ...] = ()
    findings: tuple[Finding, ...] = ()
    header: PngHeader | None = None
    palette: bytes = b""


def scan_container(data: bytes) -> ContainerReport:
    """Census a file's bytes; formats without a scanner come back empty."""
    if data.startswith(_PNG_SIGNATURE):
        return _scan_png(data)
    if data.startswith(_JPEG_SIGNATURE):
        return _scan_jpeg(data)
    return ContainerReport(kind="")


def _scan_png(data: bytes) -> ContainerReport:
    blocks: list[Block] = []
    header: PngHeader | None = None
    palette = b""
    pos = len(_PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        label = data[pos + 4 : pos + 8].decode("latin-1", errors="replace")
        total = min(12 + length, len(data) - pos)
        role = _png_role(label)
        body = data[pos + 8 : pos + 8 + length]
        blocks.append(
            Block(
                pos,
                label,
                total,
                role,
                body,
                _preview(body),
                payload_at=pos + 8,
            )
        )
        if label == "IHDR" and header is None and len(body) >= 13:
            width, height, bit_depth, color, _compression, _filters, interlace = struct.unpack(
                ">IIBBBBB", body[:13]
            )
            header = PngHeader(width, height, bit_depth, color, interlace)
        if label == "PLTE":
            palette = body
        pos += total
        if label == "IEND":
            break  # the real structure ends here; everything after is residual
    findings, stream_starts = _png_idat_findings(blocks, header)
    grouped = _group_idats(blocks, stream_starts)
    if pos < len(data):
        residual = data[pos:]
        findings.extend(_png_residual_findings(data, pos))
        grouped.append(
            Block(
                pos,
                "残留数据",
                len(residual),
                BlockRole.ANCILLARY,
                residual,
                _preview(residual),
                payload_at=pos,
            )
        )
    return ContainerReport("png", tuple(grouped), tuple(findings), header, palette)


def _png_role(label: str) -> BlockRole:
    """The spec's own rule: a lowercase first letter marks an ancillary chunk."""
    return BlockRole.REQUIRED if label[:1].isupper() else BlockRole.ANCILLARY


def _preview(body: bytes, limit: int = 32) -> str:
    """The block's bytes as printable text; unprintable ones become dots."""
    return bytes_text.preview(body, limit)


def _png_idat_findings(
    blocks: list[Block],
    header: PngHeader | None,
) -> tuple[list[Finding], list[int]]:
    """The IDAT audit: every stream found, whole, contiguous, just big enough.

    Tolerant readers render only the first stream and silently drop the rest —
    which is exactly where fake IDATs hide — so each stream beyond the first
    is reported where it starts, and its start doubles as a group number for
    the census. Stream start offsets come back for that grouping.
    """
    findings: list[Finding] = []
    idat_blocks = [block for block in blocks if block.label == "IDAT"]
    if not idat_blocks:
        return findings, []
    positions = [index for index, block in enumerate(blocks) if block.label == "IDAT"]
    if any(later - earlier != 1 for earlier, later in pairwise(positions)):
        findings.append(Finding("idat-gap", blocks[positions[1]].offset))
    data = b"".join(block.payload for block in idat_blocks)
    base = idat_blocks[0].offset + 8
    streams: list[tuple[int, int, bytes]] = []
    cursor = 0
    while cursor < len(data):
        decompressor = zlib.decompressobj()
        try:
            out = decompressor.decompress(data[cursor:])
        except zlib.error:
            if not streams:  # not even one whole stream: the audit can say no more
                findings.append(Finding("stream-truncated", base))
            break
        if not decompressor.eof:
            findings.append(Finding("stream-truncated", base + len(data)))
            break
        consumed = len(data) - cursor - len(decompressor.unused_data)
        streams.append((base + cursor, base + cursor + consumed, out))
        cursor += consumed
        if not decompressor.unused_data:
            break
    for _number, (start, end, decoded) in enumerate(streams[1:], start=2):
        findings.append(
            Finding("idat-extra", start, end - start, data[start - base : end - base], decoded)
        )
    if streams and header is not None and (expected := header.pixel_budget()) is not None:
        real = len(streams[0][2])
        if real != expected:
            surplus = streams[0][2][expected:] if real > expected else b""
            findings.append(Finding("idat-oversize", streams[0][1], abs(real - expected), surplus))
    return findings, [start for start, _end, _out in streams]


def _group_idats(blocks: list[Block], stream_starts: list[int]) -> list[Block]:
    """Number each IDAT chunk by the zlib stream it serves, counting from one.

    Stream 1 is the one a decoder renders, so a lone IDAT chunk is stream 1
    too. A stream spread over several chunks reads as one group: N smuggled
    streams read as N extra groups instead of twins of the first. Chunks past
    every stream stay with the last one, and IDATs of a stream the audit could
    not read whole get no number at all.
    """
    grouped = list(blocks)
    if not stream_starts:
        return grouped
    group, next_stream = 1, 1
    for index, block in enumerate(blocks):
        if block.label != "IDAT":
            continue
        while next_stream < len(stream_starts) and block.offset + 8 >= stream_starts[next_stream]:
            group += 1
            next_stream += 1
        grouped[index] = replace(block, group=group)
    return grouped


def _reveal(payload: bytes) -> bytes:
    """A nested zlib stream (the usual fake-IDAT shape) comes back unfolded."""
    if not payload.startswith(b"\x78"):
        return b""
    try:
        return zlib.decompress(payload)
    except zlib.error:
        return b""


def _png_residual_findings(data: bytes, pos: int) -> list[Finding]:
    """Past the first IEND everything is residual; a stitched IEND gets named.

    The residual is walked best-effort, so an IEND chunk welded on after the
    real one (the Acropalypse tell) is reported at its own offset; the bytes
    themselves travel as the 残留数据 block the census appends.
    """
    findings: list[Finding] = []
    cursor = pos
    while cursor + 8 <= len(data):
        (length,) = struct.unpack(">I", data[cursor : cursor + 4])
        if data[cursor + 4 : cursor + 8] == b"IEND":
            findings.append(Finding("duplicate-eof", cursor))
        cursor += 12 + length
    return findings


def render_blocks(payloads: Sequence[bytes], header: PngHeader, palette: bytes = b"") -> RgbArray:
    """Render selected block payloads as the pixels they would show.

    The payloads are read in file order as one pixel stream: a zlib stream is
    decompressed (raw scanline bytes work too), short streams are padded so a
    half-smuggled image still shows, and the scanlines are reconstructed under
    the file's own IHDR geometry. Interlaced data is declined.
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
    return _to_rgb(rows, header, palette)


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


def _to_rgb(rows: NDArray[np.uint8], header: PngHeader, palette: bytes) -> RgbArray:
    """Packed scanlines in, an RGB array out; alpha and extra channels drop."""
    width, depth, color = header.width, header.bit_depth, header.color_type
    channels = _CHANNELS[color]
    if depth == 8:
        pixels = rows.reshape(header.height, width, channels)
    elif depth == 16:
        pixels = rows.reshape(header.height, -1).view(">u2")[:, : width * channels]
        pixels = (pixels.reshape(header.height, width, channels) >> 8).astype(np.uint8)
    else:
        bits = np.unpackbits(rows, axis=1)[:, : width * channels]
        pixels = bits.reshape(header.height, width, channels) * (255 // (2**depth - 1))
    if color == 0:
        rgb = np.repeat(pixels, 3, axis=2)
    elif color == 2:
        rgb = pixels[..., :3]
    elif color == 3:
        entries = np.frombuffer(palette, np.uint8).reshape(-1, 3)
        indexes = np.clip(pixels[..., 0], 0, max(len(entries) - 1, 0))
        rgb = entries[indexes] if len(entries) else np.zeros((header.height, width, 3), np.uint8)
    elif color == 4:
        rgb = np.repeat(pixels[..., :1], 3, axis=2)
    else:
        rgb = pixels[..., :3]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _scan_jpeg(data: bytes) -> ContainerReport:
    blocks: list[Block] = [Block(0, "SOI", 2, BlockRole.REQUIRED)]
    findings: list[Finding] = []
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            findings.append(Finding("stream-truncated", pos, len(data) - pos, data[pos:]))
            break
        while pos + 1 < len(data) and data[pos] == 0xFF and data[pos + 1] == 0xFF:
            pos += 1  # fill bytes the spec allows between segments
        if pos + 1 >= len(data):
            break
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            blocks.append(Block(pos, "EOI", 2, BlockRole.REQUIRED))
            pos += 2
            break
        if pos + 4 > len(data):
            findings.append(Finding("stream-truncated", pos, len(data) - pos, data[pos:]))
            break
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        label = _JPEG_MARKERS.get(marker, f"FF{marker:02X}")
        role = (
            BlockRole.ANCILLARY if marker == 0xFE or 0xE0 <= marker <= 0xEF else BlockRole.REQUIRED
        )
        if marker == 0xDA:  # SOS: the segment header is followed by entropy data
            end = _jpeg_entropy_end(data, pos + 2 + length)
            blocks.append(Block(pos, label, end - pos, role))
            pos = end
            continue
        payload = data[pos + 4 : pos + 2 + length]
        blocks.append(
            Block(pos, label, 2 + length, role, payload, _preview(payload), payload_at=pos + 4)
        )
        pos += 2 + length
    if pos < len(data):
        residual = data[pos:]
        blocks.append(
            Block(
                pos,
                "残留数据",
                len(residual),
                BlockRole.ANCILLARY,
                residual,
                _preview(residual),
                payload_at=pos,
            )
        )
    return ContainerReport("jpeg", tuple(blocks), tuple(findings))


def _jpeg_entropy_end(data: bytes, start: int) -> int:
    """Entropy-coded data runs to the next marker that is not byte stuffing."""
    cursor = start
    while cursor + 1 < len(data):
        marker = data[cursor + 1]
        if data[cursor] == 0xFF and marker != 0x00 and not 0xD0 <= marker <= 0xD7:
            return cursor  # RSTn belongs to the scan; anything else ends it
        cursor += 1
    return len(data)


_JPEG_MARKERS = {
    0xC0: "SOF0",
    0xC1: "SOF1",
    0xC2: "SOF2",
    0xC4: "DHT",
    0xDB: "DQT",
    0xDD: "DRI",
    0xDA: "SOS",
    0xFE: "COM",
    **{0xE0 + index: f"APP{index}" for index in range(16)},
}

"""Container census: the blocks a file is built from, and what looks planted.

The census answers a structural question, not a semantic one: which bytes does
a renderer need, and what is merely there. Required sets come from the format
spec — for PNG, the spec's own critical/ancillary bit — so an unknown chunk is
still classified, and a smuggled second IDAT stream is still exposed, without
this module knowing what any chunk means. The census reads one container: the
first end marker ends the structure, and whatever follows travels as one tail
block, whatever it turns out to be. The census also keeps what a renderer would
need to show the blocks' bytes as pixels: the PNG header and palette. BMP has
no chunk framing to audit, so its census is the frame the spec draws — file
header, DIB header, color table, pixel region — and the sizes its pixels fill
when the declared pair does not hold them.
"""

import math
import struct
import zlib
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise

from pixelsb.domain import bytes_text

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"
_BMP_SIGNATURE = b"BM"
_FILE_HEADER = 14  # BITMAPFILEHEADER: magic, declared size, two reserved, pixel offset
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_SIZE_HINTS = 6  # candidate geometries shown for a doctored IHDR or DIB, at most


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
        channels = CHANNELS.get(self.color_type)
        if channels is None:
            return None
        return (self.width * self.bit_depth * channels + 7) // 8

    def bytes_per_pixel(self) -> int:
        """The filter unit: how many whole bytes one pixel occupies pre-unpacking."""
        channels = CHANNELS.get(self.color_type, 1)
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
class SizeHint:
    """A (width, height) the pixel data on hand would fill exactly."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Finding:
    """An anomaly: bytes no renderer will show, or structure no reader expects.

    ``payload`` carries the suspicious bytes when they are contiguous, so the
    panel can preview them and scan them for signatures without re-reading;
    ``decoded`` holds a nested stream (the usual fake-IDAT shape) unfolded, so
    what the renderer skips can still be read, and ``sizes`` names the
    geometries those decoded bytes would fill exactly.
    """

    kind: str
    offset: int
    length: int = 0
    payload: bytes = b""
    decoded: bytes = b""
    sizes: tuple[SizeHint, ...] = ()


@dataclass(frozen=True, slots=True)
class ContainerReport:
    """The census of one container: every block, and everything suspicious.

    For PNG the ``header`` and ``palette`` are kept so the blocks' bytes can be
    rendered as pixels without touching the file again. ``sizes`` names the
    geometries the pixel data would fill exactly, when that is not the
    declared one — a doctored IHDR or DIB header is the usual reason, and one
    of them is what the picture was made with.
    """

    kind: str
    blocks: tuple[Block, ...] = ()
    findings: tuple[Finding, ...] = ()
    header: PngHeader | None = None
    palette: bytes = b""
    sizes: tuple[SizeHint, ...] = ()


def scan_container(data: bytes) -> ContainerReport:
    """Census a file's bytes; formats without a scanner come back empty."""
    if data.startswith(_PNG_SIGNATURE):
        return _scan_png(data)
    if data.startswith(_JPEG_SIGNATURE):
        return _scan_jpeg(data)
    if data.startswith(_BMP_SIGNATURE):
        report = _scan_bmp(data)
        if report is not None:
            return report
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
                bytes_text.preview(body),
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
    findings, stream_starts, first_stream = _png_idat_findings(blocks, header)
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
                bytes_text.preview(residual),
                payload_at=pos,
            )
        )
    return ContainerReport(
        "png",
        tuple(grouped),
        tuple(findings),
        header,
        palette,
        _size_candidates(header, first_stream),
    )


def _png_role(label: str) -> BlockRole:
    """The spec's own rule: a lowercase first letter marks an ancillary chunk."""
    return BlockRole.REQUIRED if label[:1].isupper() else BlockRole.ANCILLARY


def _png_idat_findings(
    blocks: list[Block],
    header: PngHeader | None,
) -> tuple[list[Finding], list[int], int]:
    """The IDAT audit: every stream found, whole, contiguous, just big enough.

    Tolerant readers render only the first stream and silently drop the rest —
    which is exactly where fake IDATs hide — so each stream beyond the first
    is reported where it starts, its start doubles as a group number for the
    census, and its finding carries the geometries its own bytes would fill.
    Stream start offsets come back for that grouping, and the first stream's
    decompressed size for the geometry candidates.
    """
    findings: list[Finding] = []
    positions = [index for index, block in enumerate(blocks) if block.label == "IDAT"]
    if not positions:
        return findings, [], 0
    if any(later - earlier != 1 for earlier, later in pairwise(positions)):
        findings.append(Finding("idat-gap", blocks[positions[1]].offset))
    data = b"".join(blocks[index].payload for index in positions)
    view = memoryview(data)  # a slice of the view is free; a slice of the bytes copies
    base = blocks[positions[0]].offset + 8
    streams: list[tuple[int, int, bytes]] = []
    cursor = 0
    while cursor < len(data):
        decompressor = zlib.decompressobj()
        try:
            out = decompressor.decompress(view[cursor:])
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
    for start, end, decoded in streams[1:]:
        findings.append(
            Finding(
                "idat-extra",
                start,
                end - start,
                data[start - base : end - base],
                decoded,
                sizes=_size_candidates(header, len(decoded)),
            )
        )
    if streams and header is not None and (expected := header.pixel_budget()) is not None:
        real = len(streams[0][2])
        if real != expected:
            surplus = streams[0][2][expected:] if real > expected else b""
            findings.append(Finding("idat-oversize", streams[0][1], abs(real - expected), surplus))
    return findings, [start for start, _end, _out in streams], len(streams[0][2]) if streams else 0


def _size_candidates(header: PngHeader | None, actual: int) -> tuple[SizeHint, ...]:
    """The geometries the stream's own byte count fills exactly, closest aspect first.

    Only a *mismatch* gets candidates: when the declared pair already fits the
    data there is nothing to repair, and a divisor walk of any byte count can
    always spell out alternative (wrong) geometries. A doctored IHDR is the
    classic reason for a mismatch — the data is another picture's — so the walk
    asks one question per divisor: does this stride, with this many rows, use
    the data up? The declared pair is the first candidate to drop out; what is
    left is worth a try.
    """
    if header is None or header.interlace != 0 or actual <= 0:
        return ()
    expected = header.pixel_budget()
    if expected is None or actual == expected:
        return ()
    unit = header.bit_depth * CHANNELS.get(header.color_type, 0)
    if unit <= 0:
        return ()
    declared = (header.width, header.height)
    found: list[SizeHint] = []
    for stride in _divisors(actual):
        height = actual // stride
        width = (stride - 1) * 8 // unit
        if width < 1 or (width * unit + 7) // 8 != stride - 1:
            continue
        if (width, height) != declared:
            found.append(SizeHint(width, height))
    truth = math.log(declared[0] / declared[1])
    found.sort(key=lambda hint: abs(math.log(hint.width / hint.height) - truth))
    return tuple(found[:_SIZE_HINTS])


def _divisors(number: int) -> list[int]:
    """Every divisor of ``number``, smallest first; a number has few of them."""
    small: list[int] = []
    large: list[int] = []
    cursor = 1
    while cursor * cursor <= number:
        if number % cursor == 0:
            small.append(cursor)
            if cursor * cursor != number:
                large.append(number // cursor)
        cursor += 1
    return small + large[::-1]


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
            Block(
                pos,
                label,
                2 + length,
                role,
                payload,
                bytes_text.preview(payload),
                payload_at=pos + 4,
            )
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
                bytes_text.preview(residual),
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


def _scan_bmp(data: bytes) -> ContainerReport | None:
    """The census of a BMP: the two headers, the palette, the pixels, the rest.

    ``None`` says the bytes only pretend to start with ``BM``: the DIB header
    size — the one field every BMP carries — is not a size the spec names, so
    this is no bitmap and stays anonymous. A declared file size past the
    actual end is the one structural lie this walk reports; a doctored width
    or height is reported the PNG way, as the geometries the pixels fill.
    """
    if len(data) < _FILE_HEADER + 4:
        return None
    (dib_size,) = struct.unpack_from("<I", data, _FILE_HEADER)
    if not 12 <= dib_size <= 124:
        return None
    (bf_size,) = struct.unpack_from("<I", data, 2)
    (off_bits,) = struct.unpack_from("<I", data, 10)
    width, height, bit_count = _bmp_geometry(data, dib_size)
    findings: list[Finding] = []
    pixel_from = max(_FILE_HEADER + dib_size, min(off_bits, len(data)))
    if bf_size > len(data):
        findings.append(Finding("stream-truncated", pixel_from))
    pixel_end = bf_size if pixel_from < bf_size <= len(data) else len(data)
    pixel = data[pixel_from:pixel_end]
    head = data[:_FILE_HEADER]
    body = data[_FILE_HEADER : _FILE_HEADER + dib_size]
    blocks = [
        Block(
            0,
            "文件头",
            len(head),
            BlockRole.REQUIRED,
            head,
            bytes_text.preview(head),
            payload_at=0,
        ),
        Block(
            _FILE_HEADER,
            "DIB 头",
            len(body),
            BlockRole.REQUIRED,
            body,
            bytes_text.preview(body),
            payload_at=_FILE_HEADER,
        ),
    ]
    if pixel_from > _FILE_HEADER + dib_size:
        gap = data[_FILE_HEADER + dib_size : pixel_from]
        palette = bit_count <= 8  # the color table is the one thing a gap can be
        blocks.append(
            Block(
                _FILE_HEADER + dib_size,
                "调色板" if palette else "间隔",
                len(gap),
                BlockRole.REQUIRED if palette else BlockRole.ANCILLARY,
                gap,
                bytes_text.preview(gap),
                payload_at=_FILE_HEADER + dib_size,
            )
        )
    if pixel:
        blocks.append(
            Block(
                pixel_from,
                "像素数据",
                len(pixel),
                BlockRole.REQUIRED,
                pixel,
                bytes_text.preview(pixel),
                payload_at=pixel_from,
            )
        )
    if pixel_end < len(data):
        residual = data[pixel_end:]
        blocks.append(
            Block(
                pixel_end,
                "残留数据",
                len(residual),
                BlockRole.ANCILLARY,
                residual,
                bytes_text.preview(residual),
                payload_at=pixel_end,
            )
        )
    return ContainerReport(
        "bmp",
        tuple(blocks),
        tuple(findings),
        sizes=_bmp_size_candidates(width, abs(height), bit_count, len(pixel)),
    )


def _bmp_geometry(data: bytes, dib_size: int) -> tuple[int, int, int]:
    """The DIB header's width, height, and bit count; zeros where bytes stop short.

    The core header keeps 16-bit fields; every later member widened them to
    32 bits and moved the bit count along. Height keeps its sign — negative
    reads top-down — and only its size matters to this census.
    """
    core = dib_size == 12
    if len(data) < _FILE_HEADER + (26 if core else 30):
        return 0, 0, 0
    if core:
        width, height = struct.unpack_from("<HH", data, 18)
        (bit_count,) = struct.unpack_from("<H", data, 24)
    else:
        width, height = struct.unpack_from("<ii", data, 18)
        (bit_count,) = struct.unpack_from("<H", data, 28)
    return width, height, bit_count


def _bmp_row_bytes(width: int, bit_count: int) -> int:
    """One row's bytes: a BMP pads every scanline to whole 4-byte units."""
    return (width * bit_count + 31) // 32 * 4


def _bmp_size_candidates(
    width: int, height: int, bit_count: int, actual: int
) -> tuple[SizeHint, ...]:
    """The geometries the pixel region fills exactly, closest aspect first.

    Only a *mismatch* gets candidates, and the declared pair is the first to
    drop out: a doctored DIB header over another picture's pixels is the
    reason this walk exists, so each padded stride a divisor names yields the
    widths that fit it and nothing else does.
    """
    if actual <= 0 or bit_count <= 0 or width < 1 or height < 1:
        return ()
    if actual == _bmp_row_bytes(width, bit_count) * height:
        return ()
    truth = math.log(width / height)
    found: list[SizeHint] = []
    for stride in _divisors(actual):
        candidate = stride * 8 // bit_count
        while candidate >= 1 and _bmp_row_bytes(candidate, bit_count) == stride:
            found.append(SizeHint(candidate, actual // stride))
            candidate -= 1
    found.sort(key=lambda hint: abs(math.log(hint.width / hint.height) - truth))
    return tuple(found[:_SIZE_HINTS])

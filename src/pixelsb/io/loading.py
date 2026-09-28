"""Decode image files into read-only sample planes."""

import io
import struct
import zlib
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pixelsb.domain.container import ContainerReport, SizeHint, scan_container
from pixelsb.domain.models import FrameGeometry, LoadedImage, SampleArray, SampleOrigin, SamplePlane

type Decoded = tuple[SampleArray, tuple[SamplePlane, ...]]
type Decoder = Callable[[Image.Image], Decoded]
type RepairPlan = tuple[bytes, int, tuple[SizeHint, ...], str]

# Recent files' repair plans: the census behind one costs a pass over the whole
# pixel stream, and the info page's size row and its buttons ask for it again
# and again, where sharing one plan costs nothing but memory for the bytes.
_REPAIR_PLANS = 8


class ImageLoadError(Exception):
    """The file could not be decoded as a viewable image."""


def load_image(path: Path) -> LoadedImage:
    """Load the first frame of an image file."""
    return load_frame(path, 0)


def load_frame(path: Path, index: int) -> LoadedImage:
    """Load one frame of a (possibly multi-frame) image.

    Samples are native-endian uint16. A palette picture carries the colors its
    palette looks up, and 16-bit values keep their numeric samples. A PNG
    whose declared size does not fit its own data — the doctored-IHDR trick —
    is retried under the geometries its byte count implies, closest to the
    declared aspect first; the file on disk stays exactly as it was, and the
    file-info page tells the whole story.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ImageLoadError(f"file not found: {file_path}")
    try:
        with Image.open(file_path) as image:
            delays = _frame_delays(image)
            if not 0 <= index < max(len(delays), 1):
                raise ImageLoadError(f"no frame {index}: the image has {max(len(delays), 1)}")
            if delays:  # the delay pass seeks; put the reader back on the frame
                image.seek(index)
            image.load()
            return _decode(file_path, image, index, delays)
    except ImageLoadError:
        raise
    except Exception as exc:
        repaired = _repaired(file_path, index)
        if repaired is not None:
            return repaired
        raise ImageLoadError(str(exc)) from exc


def _repair_hints(path: Path) -> RepairPlan | None:
    """A doctored header's bytes, its width field's offset, the sizes to try, the kind.

    ``None`` when the file cannot be read, hides no size candidates, or is a
    kind whose header this cannot rewrite; each caller turns that into its own
    failure. The plan is remembered per file stamp — size and mtime — so the
    page's size row and a click on one of its buttons share one census instead
    of re-reading and re-decompressing the file for each.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    return _repair_plan(path, (info.st_mtime_ns, info.st_size))


@lru_cache(maxsize=_REPAIR_PLANS)
def _repair_plan(path: Path, _stamp: tuple[int, int]) -> RepairPlan | None:
    """The plan for one file at one stamp: the stamp is what keeps the cache true."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    report = scan_container(data)
    if not report.sizes:
        return None
    at = _width_field(report)
    if at is None:
        return None
    return data, at, report.sizes, report.kind


def _width_field(report: ContainerReport) -> int | None:
    """Where the declared width starts: the IHDR payload, or the DIB body.

    Only the modern DIB headers — 40 bytes and up — carry 32-bit width and
    height this can rewrite; the museum-piece core header keeps 16-bit fields
    and gets no repair.
    """
    if report.kind == "png":
        if report.header is None:
            return None
        return next((block.payload_at for block in report.blocks if block.label == "IHDR"), None)
    if report.kind == "bmp":
        dib = next((block for block in report.blocks if block.label == "DIB 头"), None)
        if dib is None or len(dib.payload) < 4:
            return None
        (dib_size,) = struct.unpack_from("<I", dib.payload, 0)
        if dib_size < 40:
            return None
        return dib.payload_at + 4  # past the size field, where width and height live
    return None


def _patched(data: bytes, at: int, hint: SizeHint, kind: str) -> bytes:
    """The bytes rewritten to the hint's geometry: PNG renews its CRC, BMP is bare.

    Every kind ``_width_field`` can locate a width field for has its patch
    here, and a plan only hands out such kinds, so this is total over what the
    callers reach; a patch that decodes wrong fails the decode every hint must
    survive anyway.
    """
    if kind == "bmp":
        patched = bytearray(data)
        patched[at : at + 8] = struct.pack("<ii", hint.width, hint.height)
        return bytes(patched)
    return patched_ihdr(data, at, hint)


def _repaired(path: Path, index: int) -> LoadedImage | None:
    """The image a doctored size header was hiding, read under its own geometry.

    The census's size hints are tried in their aspect order; the first one PIL
    accepts wins, and a file whose data fits nothing comes back as failure, as
    it always did. The bytes on disk are never touched.
    """
    found = _repair_hints(path)
    if found is None:
        return None
    data, at, hints, kind = found
    for hint in hints:
        patched = _patched(data, at, hint, kind)
        try:
            with Image.open(io.BytesIO(patched)) as image:
                if index and index >= max(int(getattr(image, "n_frames", 1) or 1), 1):
                    continue
                image.seek(index)
                image.load()
                return _decode(path, image, index, ())
        except Exception:
            continue
    return None


def openable_repairs(path: Path) -> tuple[SizeHint, ...]:
    """The census's size hints that Pillow can actually decode, best first.

    A byte count that matches a candidate geometry is necessary, never
    sufficient: a stride can disagree with a renderer in ways the byte count
    cannot see, so every hint is proved by a real decode before it earns a
    button.
    """
    found = _repair_hints(path)
    if found is None:
        return ()
    data, at, hints, kind = found
    good: list[SizeHint] = []
    for hint in hints:
        patched = _patched(data, at, hint, kind)
        try:
            with Image.open(io.BytesIO(patched)) as image:
                image.load()
        except Exception:
            continue
        good.append(hint)
    return tuple(good)


def patched_size(path: Path, hint: SizeHint) -> bytes | None:
    """The file's bytes rewritten to the hint's geometry, checksums and all.

    The repair row's own patch, so the copy it hands out is sound by anyone's
    reading: a PNG's IHDR gets a fresh CRC, a BMP's DIB header is bare.
    ``None`` when the header cannot be found or rewritten.
    """
    found = _repair_hints(path)
    if found is None:
        return None
    data, at, _hints, kind = found
    return _patched(data, at, hint, kind)


def patched_ihdr(data: bytes, header_at: int, hint: SizeHint) -> bytes:
    """The file's bytes with the IHDR's width and height replaced, CRC and all.

    A patched chunk with the old checksum would be thrown out by the very
    decoder this is meant to convince, so the CRC is recomputed over the chunk
    type and the patched payload.
    """
    patched = bytearray(data)
    patched[header_at : header_at + 8] = struct.pack(">II", hint.width, hint.height)
    crc = zlib.crc32(bytes(patched[header_at - 4 : header_at + 13])) & 0xFFFFFFFF
    patched[header_at + 13 : header_at + 17] = struct.pack(">I", crc)
    return bytes(patched)


def image_from_pixels(path: Path, pixels: NDArray[np.uint8]) -> LoadedImage:
    """Wrap rendered pixels as a viewable image, provenance kept as ``path``.

    Three channels read as RGB, four as RGBA: a stream rendered from a block that
    carries alpha keeps its transparency rather than dropping it.
    """
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("expected an HxWx3 or HxWx4 array")
    height, width, channels = pixels.shape
    names = ("R", "G", "B", "A")[:channels]
    specs = tuple((name, 8, SampleOrigin.RAW) for name in names)
    return LoadedImage(
        path=Path(path),
        source_mode="RGB" if channels == 3 else "RGBA",
        width=width,
        height=height,
        samples=_stack(*(pixels[..., index] for index in range(channels))),
        planes=_planes(*specs),
        frame_count=1,
        frame_index=0,
    )


def frame_geometries(path: Path) -> tuple[FrameGeometry, ...]:
    """Each animated frame's own rectangle and delay, for the file-info table.

    Only the formats that place frames on a canvas — GIF — have a rectangle to
    give; anything else comes back empty. A seek reads the frame's header, not
    its pixels, so the whole table costs far less than decoding the animation.
    """
    try:
        with Image.open(Path(path).expanduser()) as image:
            count = max(int(getattr(image, "n_frames", 1) or 1), 1)
            frames: list[FrameGeometry] = []
            for index in range(count):
                image.seek(index)
                extent = getattr(image, "dispose_extent", None)
                if extent is None:
                    return ()  # frames are full pictures here: no offset to report
                x0, y0, x1, y1 = extent
                frames.append(
                    FrameGeometry(
                        index,
                        x1 - x0,
                        y1 - y0,
                        x0,
                        y0,
                        int(image.info.get("duration", 0) or 0),
                    )
                )
            return tuple(frames)
    except OSError, ValueError, EOFError, SyntaxError, KeyError:
        return ()  # a file this broken has no frame table to show


def _frame_delays(image: Image.Image) -> tuple[int, ...]:
    """Per-frame duration in milliseconds; empty for a single-frame image."""
    count = max(int(getattr(image, "n_frames", 1) or 1), 1)
    if count == 1:
        return ()
    delays = []
    for position in range(count):
        image.seek(position)
        delays.append(int(image.info.get("duration", 0) or 0))
    return tuple(delays)


def _decode(
    path: Path,
    image: Image.Image,
    index: int,
    delays: tuple[int, ...],
) -> LoadedImage:
    if image.width < 1 or image.height < 1:
        raise ImageLoadError("image has no pixels")
    samples, planes = _with_color_key(image, _DECODERS.get(image.mode, _converted)(image))
    return LoadedImage(
        path=path,
        source_mode=image.mode,
        width=image.width,
        height=image.height,
        samples=samples,
        planes=planes,
        frame_count=max(len(delays), 1),
        frame_index=index,
        frame_delays=delays,
    )


def _bilevel(image: Image.Image) -> Decoded:
    """One bit per pixel, kept as 0/1 so the bit grid and the pixels agree."""
    return _stack(np.asarray(image) != 0), _planes(("L", 1, SampleOrigin.RAW))


def _gray(image: Image.Image) -> Decoded:
    return _stack(np.asarray(image)), _planes(("L", 8, SampleOrigin.RAW))


def _gray_alpha(image: Image.Image) -> Decoded:
    array = np.asarray(image)
    return (
        _stack(array[..., 0], array[..., 1]),
        _planes(("L", 8, SampleOrigin.RAW), ("A", 8, SampleOrigin.RAW)),
    )


def _rgb(image: Image.Image) -> Decoded:
    array = np.asarray(image)
    return (
        _stack(array[..., 0], array[..., 1], array[..., 2]),
        _planes(
            ("R", 8, SampleOrigin.RAW),
            ("G", 8, SampleOrigin.RAW),
            ("B", 8, SampleOrigin.RAW),
        ),
    )


def _rgba(image: Image.Image) -> Decoded:
    array = np.asarray(image)
    return (
        _stack(array[..., 0], array[..., 1], array[..., 2], array[..., 3]),
        _planes(
            ("R", 8, SampleOrigin.RAW),
            ("G", 8, SampleOrigin.RAW),
            ("B", 8, SampleOrigin.RAW),
            ("A", 8, SampleOrigin.RAW),
        ),
    )


def _gray16(image: Image.Image) -> Decoded:
    values = np.array(image)
    if values.ndim != 2:
        raise ImageLoadError(f"expected a 2D 16-bit image, got shape {values.shape}")
    return _stack(values), _planes(("L", 16, SampleOrigin.RAW))


def _palette(image: Image.Image) -> Decoded:
    """The colors the palette looks up, with the alpha channel beside them."""
    return _rgba_channels(image), _rgba_planes(SampleOrigin.PALETTE, SampleOrigin.PALETTE)


def _palette_alpha(image: Image.Image) -> Decoded:
    """The looked-up colors, over the alpha the file itself carries."""
    return _rgba_channels(image), _rgba_planes(SampleOrigin.PALETTE, SampleOrigin.RAW)


def _with_color_key(image: Image.Image, decoded: Decoded) -> Decoded:
    """Add the alpha plane a PNG color key implies, for the pictures that lack one.

    A truecolor or gray PNG can name one transparent sample in its tRNS chunk
    instead of carrying an alpha channel; PIL keeps that key in ``info`` and hands
    the pixels over without it. Turning it into an alpha plane is what makes such
    a file behave like the others: transparency on the canvas, ``A`` to select in
    the bit grid and to extract. Formats that really carry alpha already have one,
    so they are left exactly as decoded.
    """
    samples, planes = decoded
    key = image.info.get("transparency")
    if key is None or any(plane.name == "A" for plane in planes):
        return decoded
    if isinstance(key, int):
        transparent = samples[..., 0] == np.uint16(key)
    elif isinstance(key, tuple) and len(key) == len(planes):
        transparent = np.all(samples == np.array(key, dtype=np.uint16), axis=-1)
    else:
        return decoded  # a key this decoder cannot line up with the samples
    alpha = np.where(transparent, np.uint16(0), np.uint16(255)).astype(np.uint16)
    return (
        np.concatenate([samples, alpha[..., None]], axis=-1),
        (*planes, SamplePlane("A", len(planes), 8, SampleOrigin.RAW)),
    )


def _converted(image: Image.Image) -> Decoded:
    """Anything else is decoded to RGBA, and the status bar says so."""
    return _rgba_channels(image), _rgba_planes(SampleOrigin.CONVERTED, SampleOrigin.CONVERTED)


def _rgba_channels(image: Image.Image) -> SampleArray:
    """The picture as four uint16 planes, read out of its RGBA form."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return _stack(rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3])


def _rgba_planes(rgb_origin: SampleOrigin, alpha_origin: SampleOrigin) -> tuple[SamplePlane, ...]:
    """The plane table of a four-channel picture, whose alpha may decode its own way."""
    return _planes(
        ("R", 8, rgb_origin),
        ("G", 8, rgb_origin),
        ("B", 8, rgb_origin),
        ("A", 8, alpha_origin),
    )


# One decoder per mode we keep as stored; the 16-bit spellings differ only in
# byte order, which PIL has already normalized by the time we read the pixels.
_DECODERS: dict[str, Decoder] = {
    "1": _bilevel,
    "L": _gray,
    "LA": _gray_alpha,
    "RGB": _rgb,
    "RGBA": _rgba,
    "I;16": _gray16,
    "I;16L": _gray16,
    "I;16B": _gray16,
    "I;16N": _gray16,
    "P": _palette,
    "PA": _palette_alpha,
}


def _planes(*specs: tuple[str, int, SampleOrigin]) -> tuple[SamplePlane, ...]:
    return tuple(
        SamplePlane(name, index, bit_depth, origin)
        for index, (name, bit_depth, origin) in enumerate(specs)
    )


def _stack(*channels: NDArray[np.generic]) -> SampleArray:
    stacked = np.stack([np.asarray(channel, dtype=np.uint16) for channel in channels], axis=-1)
    return np.ascontiguousarray(stacked, dtype=np.uint16)

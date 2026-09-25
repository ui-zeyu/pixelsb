"""Decode image files into read-only sample planes."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pixelsb.domain.models import LoadedImage, SampleArray, SampleOrigin, SamplePlane

type Decoded = tuple[SampleArray, tuple[SamplePlane, ...]]
type Decoder = Callable[[Image.Image], Decoded]


class ImageLoadError(Exception):
    """The file could not be decoded as a viewable image."""


def load_image(path: Path) -> LoadedImage:
    """Load the first frame of an image file."""
    return load_frame(path, 0)


def load_frame(path: Path, index: int) -> LoadedImage:
    """Load one frame of a (possibly multi-frame) image.

    Samples are native-endian uint16. Palette indexes stay separate from the
    colors they look up, and 16-bit values keep their numeric samples.
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
        raise ImageLoadError(str(exc)) from exc


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
    """The index is its own plane; the colors it looks up are four more."""
    rgba = np.asarray(image.convert("RGBA"))
    index = np.asarray(image)
    return (
        _stack(index, rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]),
        _planes(
            ("Index", 8, SampleOrigin.RAW),
            ("R", 8, SampleOrigin.PALETTE),
            ("G", 8, SampleOrigin.PALETTE),
            ("B", 8, SampleOrigin.PALETTE),
            ("A", 8, SampleOrigin.PALETTE),
        ),
    )


def _palette_alpha(image: Image.Image) -> Decoded:
    raw = np.asarray(image)
    if raw.ndim != 3 or raw.shape[2] < 2:
        raise ImageLoadError(f"expected PA samples, got shape {raw.shape}")
    rgba = np.asarray(image.convert("RGBA"))
    return (
        _stack(raw[..., 0], rgba[..., 0], rgba[..., 1], rgba[..., 2], raw[..., 1]),
        _planes(
            ("Index", 8, SampleOrigin.RAW),
            ("R", 8, SampleOrigin.PALETTE),
            ("G", 8, SampleOrigin.PALETTE),
            ("B", 8, SampleOrigin.PALETTE),
            ("A", 8, SampleOrigin.RAW),
        ),
    )


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
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return (
        _stack(rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]),
        _planes(
            ("R", 8, SampleOrigin.CONVERTED),
            ("G", 8, SampleOrigin.CONVERTED),
            ("B", 8, SampleOrigin.CONVERTED),
            ("A", 8, SampleOrigin.CONVERTED),
        ),
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

"""Decode image files into read-only sample planes."""

import contextlib
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
    """Load the first frame.

    Samples are native-endian uint16. Palette indexes stay separate from the
    colors they look up, and 16-bit values keep their numeric samples.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ImageLoadError(f"file not found: {file_path}")
    try:
        with Image.open(file_path) as image:
            with contextlib.suppress(EOFError, ValueError, AttributeError, OSError):
                image.seek(0)
            image.load()
            return _decode(file_path, image)
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(str(exc)) from exc


def _decode(path: Path, image: Image.Image) -> LoadedImage:
    if image.width < 1 or image.height < 1:
        raise ImageLoadError("image has no pixels")
    samples, planes = _DECODERS.get(image.mode, _converted)(image)
    return LoadedImage(
        path=path,
        source_mode=image.mode,
        width=image.width,
        height=image.height,
        samples=samples,
        planes=planes,
        frame_count=max(int(getattr(image, "n_frames", 1) or 1), 1),
        frame_index=0,
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

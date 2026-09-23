"""Decode image files into read-only sample planes."""

import contextlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pixelsb.domain.models import LoadedImage, PreviewArray, SampleArray, SampleOrigin, SamplePlane


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
    match image.mode:
        case "1" | "L" | "LA" | "RGB" | "RGBA":
            samples, planes, preview = _raw_color(image)
        case "I;16" | "I;16L" | "I;16B":
            samples, planes, preview = _gray16(image)
        case "P" | "PA":
            samples, planes, preview = _palette(image)
        case _:
            samples, planes, preview = _converted(image)
    frame_count = max(int(getattr(image, "n_frames", 1) or 1), 1)
    return LoadedImage(
        path=path,
        source_mode=image.mode,
        width=image.width,
        height=image.height,
        samples=samples,
        planes=planes,
        preview_rgba=preview,
        frame_count=frame_count,
        frame_index=0,
    )


def _raw_color(
    image: Image.Image,
) -> tuple[SampleArray, tuple[SamplePlane, ...], PreviewArray]:
    array = np.asarray(image)
    match image.mode:
        case "1":
            mask = array != 0
            values = mask.astype(np.uint16)
            gray = mask.astype(np.uint8) * np.uint8(255)
            return (
                _as_samples(values),
                _planes(("L", 1, SampleOrigin.RAW)),
                _opaque_gray(gray, 255),
            )
        case "L":
            values = array.astype(np.uint16)
            return (
                _as_samples(values),
                _planes(("L", 8, SampleOrigin.RAW)),
                _opaque_gray(array.astype(np.uint8), 255),
            )
        case "LA":
            values = array.astype(np.uint16)
            return (
                _as_samples(values[..., 0], values[..., 1]),
                _planes(("L", 8, SampleOrigin.RAW), ("A", 8, SampleOrigin.RAW)),
                _opaque_gray(array[..., 0].astype(np.uint8), array[..., 1].astype(np.uint8)),
            )
        case "RGB":
            color = array.astype(np.uint8)
            return (
                _as_samples(color[..., 0], color[..., 1], color[..., 2]),
                _planes(
                    ("R", 8, SampleOrigin.RAW),
                    ("G", 8, SampleOrigin.RAW),
                    ("B", 8, SampleOrigin.RAW),
                ),
                _with_alpha(color, 255),
            )
        case "RGBA":
            color = array.astype(np.uint8)
            return (
                _as_samples(color[..., 0], color[..., 1], color[..., 2], color[..., 3]),
                _planes(
                    ("R", 8, SampleOrigin.RAW),
                    ("G", 8, SampleOrigin.RAW),
                    ("B", 8, SampleOrigin.RAW),
                    ("A", 8, SampleOrigin.RAW),
                ),
                _preview(color),
            )
        case _:
            raise ImageLoadError(f"unsupported color mode: {image.mode}")


def _gray16(image: Image.Image) -> tuple[SampleArray, tuple[SamplePlane, ...], PreviewArray]:
    values = np.array(image).astype(np.uint16)
    if values.ndim != 2:
        raise ImageLoadError(f"expected a 2D 16-bit image, got shape {values.shape}")
    gray = (values.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    return (
        _as_samples(values),
        _planes(("L", 16, SampleOrigin.RAW)),
        _opaque_gray(gray, 255),
    )


def _palette(image: Image.Image) -> tuple[SampleArray, tuple[SamplePlane, ...], PreviewArray]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if image.mode == "P":
        index = np.asarray(image, dtype=np.uint16)
        samples = _as_samples(
            index,
            rgba[..., 0],
            rgba[..., 1],
            rgba[..., 2],
            rgba[..., 3],
        )
        planes = _planes(
            ("Index", 8, SampleOrigin.RAW),
            ("R", 8, SampleOrigin.PALETTE),
            ("G", 8, SampleOrigin.PALETTE),
            ("B", 8, SampleOrigin.PALETTE),
            ("A", 8, SampleOrigin.PALETTE),
        )
        return samples, planes, _preview(rgba)
    raw = np.asarray(image)
    if raw.ndim != 3 or raw.shape[2] < 2:
        raise ImageLoadError(f"expected PA samples, got shape {raw.shape}")
    samples = _as_samples(
        raw[..., 0],
        rgba[..., 0],
        rgba[..., 1],
        rgba[..., 2],
        raw[..., 1],
    )
    planes = _planes(
        ("Index", 8, SampleOrigin.RAW),
        ("R", 8, SampleOrigin.PALETTE),
        ("G", 8, SampleOrigin.PALETTE),
        ("B", 8, SampleOrigin.PALETTE),
        ("A", 8, SampleOrigin.RAW),
    )
    preview = rgba.copy()
    preview[..., 3] = raw[..., 1].astype(np.uint8)
    return samples, planes, _preview(preview)


def _converted(image: Image.Image) -> tuple[SampleArray, tuple[SamplePlane, ...], PreviewArray]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return (
        _as_samples(rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]),
        _planes(
            ("R", 8, SampleOrigin.CONVERTED),
            ("G", 8, SampleOrigin.CONVERTED),
            ("B", 8, SampleOrigin.CONVERTED),
            ("A", 8, SampleOrigin.CONVERTED),
        ),
        _preview(rgba),
    )


def _planes(*specs: tuple[str, int, SampleOrigin]) -> tuple[SamplePlane, ...]:
    return tuple(
        SamplePlane(name, index, bit_depth, origin)
        for index, (name, bit_depth, origin) in enumerate(specs)
    )


def _as_samples(*channels: NDArray[np.generic]) -> SampleArray:
    stacked = np.stack([np.asarray(channel, dtype=np.uint16) for channel in channels], axis=-1)
    return np.ascontiguousarray(stacked, dtype=np.uint16)


def _preview(rgba: NDArray[np.uint8]) -> PreviewArray:
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def _with_alpha(rgb: NDArray[np.uint8], alpha: int) -> PreviewArray:
    plane = np.full(rgb.shape[:2], alpha, dtype=np.uint8)
    return _preview(np.dstack([rgb, plane]))


def _opaque_gray(gray: NDArray[np.uint8], alpha: NDArray[np.uint8] | int) -> PreviewArray:
    if isinstance(alpha, int):
        alpha_plane = np.full(gray.shape, alpha, dtype=np.uint8)
    else:
        alpha_plane = np.asarray(alpha, dtype=np.uint8)
    return _preview(np.dstack([gray, gray, gray, alpha_plane]))

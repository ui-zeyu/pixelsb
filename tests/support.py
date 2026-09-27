"""What several test modules build their cases from: images, masks, chunks, widgets."""

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QPushButton, QWidget

from pixelsb.domain.models import (
    BitChoice,
    Layer,
    LoadedImage,
    Mask,
    Raster,
    SampleOrigin,
    SamplePlane,
    ViewerState,
)
from pixelsb.domain.stack import resolve
from pixelsb.domain.transitions import current_bits


def planes_rgb() -> tuple[SamplePlane, ...]:
    return (
        SamplePlane("R", 0, 8, SampleOrigin.RAW),
        SamplePlane("G", 1, 8, SampleOrigin.RAW),
        SamplePlane("B", 2, 8, SampleOrigin.RAW),
    )


def planes_rgba() -> tuple[SamplePlane, ...]:
    return (*planes_rgb(), SamplePlane("A", 3, 8, SampleOrigin.RAW))


def make_image(
    samples: np.ndarray,
    plane_list: tuple[SamplePlane, ...],
    *,
    source_mode: str = "RGB",
    path: Path | None = None,
    frame_count: int = 1,
) -> LoadedImage:
    array = np.asarray(samples, dtype=np.uint16)
    height, width, _channels = array.shape
    return LoadedImage(
        path=path or Path("memory.png"),
        source_mode=source_mode,
        width=width,
        height=height,
        samples=array,
        planes=plane_list,
        frame_count=frame_count,
        frame_index=0,
    )


def planes_of(image: LoadedImage) -> tuple[str, ...]:
    """The channel names an image carries, in order."""
    return tuple(plane.name for plane in image.planes)


def layers(*masks: Mask, enabled: bool = True) -> tuple[Layer, ...]:
    """A stack of masks, all enabled unless the case is about a switched-off one."""
    return tuple(Layer(mask, enabled=enabled) for mask in masks)


def raster(image: LoadedImage, *masks: Mask) -> Raster:
    """The pixels a stack of masks makes of an image."""
    return resolve(image, layers(*masks))


def board_of(state: ViewerState) -> Raster:
    """The raster a state's own stack makes of its own image."""
    assert state.image is not None
    return resolve(state.image, state.layers)


def masks(state: ViewerState) -> tuple[Mask, ...]:
    """The stack's masks, bottom to top."""
    return tuple(layer.mask for layer in state.layers)


def mask_of[Kind: Mask](state: ViewerState, kind: type[Kind]) -> Kind:
    """The one mask of that kind in the stack."""
    found = [layer.mask for layer in state.layers if isinstance(layer.mask, kind)]
    (only,) = found
    return only


def bits_of(state: ViewerState) -> frozenset[BitChoice]:
    """The bits the canvas shows: the topmost projection's, or the whole image."""
    return current_bits(state)


def chunk(label: bytes, payload: bytes) -> bytes:
    """One PNG chunk: length, type, payload, and the CRC over the last two."""
    return (
        struct.pack(">I", len(payload))
        + label
        + payload
        + struct.pack(">I", zlib.crc32(label + payload) & 0xFFFFFFFF)
    )


def button(pair: QWidget, label: str) -> QPushButton:
    """The button of a segmented pair, by the text on it."""
    return next(widget for widget in pair.findChildren(QPushButton) if widget.text() == label)


def doctored_png(path: Path, *, ramp: bool = False, declared: tuple[int, int] = (10, 10)) -> Path:
    """A PNG declared ``declared`` whose data really fills 5x1: the repair row's case.

    ``ramp`` puts nonzero bytes into the scanline, so a re-read under any other
    geometry meets invalid filter bytes: only the true 5x1 survives a decode,
    which is what the loader-repair cases want; plain zeros keep every
    candidate decodable, which is what the size-button rows want.
    """
    pixels = np.zeros((1, 5, 3), dtype=np.uint8)
    if ramp:
        pixels[0, :, 0] = np.arange(5, dtype=np.uint8) * 40
    Image.fromarray(pixels).save(path)
    data = bytearray(path.read_bytes())
    at = data.find(b"IHDR") + 4
    width, height = declared
    data[at : at + 8] = width.to_bytes(4, "big") + height.to_bytes(4, "big")
    path.write_bytes(bytes(data))
    return path

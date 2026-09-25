"""What several test modules build their cases from: images, chunks, widgets."""

import struct
import zlib
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QPushButton, QWidget

from pixelsb.domain.models import LoadedImage, SampleOrigin, SamplePlane


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

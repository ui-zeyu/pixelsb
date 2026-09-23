"""Immutable image samples and viewer state."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type SampleArray = NDArray[np.uint16]
type PreviewArray = NDArray[np.uint8]
type RgbArray = NDArray[np.uint8]

MIN_ZOOM = 1.0
MAX_ZOOM = 128.0


class SampleOrigin(StrEnum):
    RAW = "raw"
    PALETTE = "palette"
    CONVERTED = "converted"


class DisplayFormat(StrEnum):
    DECIMAL = "decimal"
    HEX = "hex"
    BINARY = "binary"


class ValueMode(StrEnum):
    ABSOLUTE = "absolute"
    OFFSET = "offset"


@dataclass(frozen=True, slots=True)
class BitChoice:
    plane: str
    bit: int

    def __post_init__(self) -> None:
        if not self.plane:
            raise ValueError("plane name is required")
        if self.bit < 0:
            raise ValueError("bit must be non-negative")


@dataclass(frozen=True, slots=True)
class PixelCoord:
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("pixel coordinates must be non-negative")


@dataclass(frozen=True, slots=True)
class SamplePlane:
    name: str
    index: int
    bit_depth: int
    origin: SampleOrigin

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plane name is required")
        if self.index < 0:
            raise ValueError("plane index must be non-negative")
        if not 1 <= self.bit_depth <= 16:
            raise ValueError("bit depth must be from 1 to 16")


@dataclass(frozen=True, slots=True, eq=False)
class LoadedImage:
    path: Path
    source_mode: str
    width: int
    height: int
    samples: SampleArray
    planes: tuple[SamplePlane, ...]
    preview_rgba: PreviewArray
    frame_count: int
    frame_index: int

    def __post_init__(self) -> None:
        if self.samples.dtype != np.uint16 or self.samples.ndim != 3:
            raise ValueError("samples must be a uint16 array of shape (height, width, planes)")
        if self.preview_rgba.dtype != np.uint8 or self.preview_rgba.ndim != 3:
            raise ValueError("preview must be a uint8 array of shape (height, width, 4)")
        samples = np.ascontiguousarray(self.samples)
        preview = np.ascontiguousarray(self.preview_rgba)
        samples.setflags(write=False)
        preview.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "preview_rgba", preview)
        if self.width < 1 or self.height < 1:
            raise ValueError("image has no pixels")
        if not self.planes:
            raise ValueError("image has no planes")
        if samples.shape != (self.height, self.width, len(self.planes)):
            raise ValueError("sample shape does not match the image size and planes")
        if preview.shape != (self.height, self.width, 4):
            raise ValueError("preview shape does not match the image size")
        if len({plane.name for plane in self.planes}) != len(self.planes):
            raise ValueError("plane names must be unique")
        for index, plane in enumerate(self.planes):
            if plane.index != index:
                raise ValueError("plane index must match its position")
        if self.frame_count < 1 or not 0 <= self.frame_index < self.frame_count:
            raise ValueError("frame index is outside the image")

    def plane(self, name: str) -> SamplePlane:
        for candidate in self.planes:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    @property
    def any_converted(self) -> bool:
        return any(plane.origin is SampleOrigin.CONVERTED for plane in self.planes)


def ensure_inside(image: LoadedImage, coord: PixelCoord) -> None:
    if not 0 <= coord.x < image.width or not 0 <= coord.y < image.height:
        raise ValueError(f"pixel ({coord.x}, {coord.y}) is outside {image.width}x{image.height}")


@dataclass(frozen=True, slots=True, eq=False)
class ViewerState:
    """``selection is None`` means every bit of every plane (the original image)."""

    image: LoadedImage | None = None
    cursor: PixelCoord | None = None
    anchor: PixelCoord | None = None
    selection: frozenset[BitChoice] | None = None
    focus: BitChoice | None = None
    readout: frozenset[BitChoice] | None = None
    detached: bool = False
    filter_expr: str = ""
    value_format: DisplayFormat = DisplayFormat.HEX
    value_mode: ValueMode = ValueMode.ABSOLUTE
    zoom: float = 1.0

    def __post_init__(self) -> None:
        if not MIN_ZOOM <= self.zoom <= MAX_ZOOM:
            raise ValueError(f"zoom {self.zoom} is outside {MIN_ZOOM}..{MAX_ZOOM}")
        if self.image is None:
            if (
                self.cursor is not None
                or self.anchor is not None
                or self.focus is not None
                or self.selection is not None
                or self.readout is not None
            ):
                raise ValueError("cursor, anchor, focus, selection, and readout require an image")
            return
        if self.selection is not None:
            for choice in self.selection:
                _require_choice(self.image, choice)
        if self.readout is not None:
            for choice in self.readout:
                _require_choice(self.image, choice)
        if self.focus is not None:
            _require_choice(self.image, self.focus)
        if self.cursor is not None:
            ensure_inside(self.image, self.cursor)
        if self.anchor is not None:
            ensure_inside(self.image, self.anchor)


def _require_choice(image: LoadedImage, choice: BitChoice) -> None:
    try:
        plane = image.plane(choice.plane)
    except KeyError as exc:
        raise ValueError(f"unknown plane: {choice.plane}") from exc
    if choice.bit >= plane.bit_depth:
        raise ValueError(f"bit {choice.bit} is outside 0..{plane.bit_depth - 1}")

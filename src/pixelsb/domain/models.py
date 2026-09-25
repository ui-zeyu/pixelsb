"""Immutable image samples and viewer state."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type SampleArray = NDArray[np.uint16]
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


class ExtractEncoding(StrEnum):
    """How the extract panel reads the byte stream as text."""

    ASCII = "ascii"
    UTF8 = "utf-8"
    UTF16_LE = "utf-16-le"
    UTF16_BE = "utf-16-be"


class BitOrder(StrEnum):
    """Which end of a byte the first bit of the stream lands in."""

    MSB = "msb"
    LSB = "lsb"


class ScanOrder(StrEnum):
    """Which pixel axis the stream runs along fastest."""

    XY = "xy"  # row by row, then the next row
    YZ = "yz"  # column by column, then the next column


@dataclass(frozen=True, slots=True)
class ViewAdjust:
    """Post-processing laid on the composed view: invert, grayscale, threshold."""

    invert: bool = False
    grayscale: bool = False
    threshold: bool = False
    level: int = 128

    def __post_init__(self) -> None:
        if not 0 <= self.level <= 255:
            raise ValueError("threshold level must be within 0..255")


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
    frame_count: int
    frame_index: int
    frame_delays: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.samples.dtype != np.uint16 or self.samples.ndim != 3:
            raise ValueError("samples must be a uint16 array of shape (height, width, planes)")
        samples = np.ascontiguousarray(self.samples)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        if self.width < 1 or self.height < 1:
            raise ValueError("image has no pixels")
        if not self.planes:
            raise ValueError("image has no planes")
        if samples.shape != (self.height, self.width, len(self.planes)):
            raise ValueError("sample shape does not match the image size and planes")
        if len({plane.name for plane in self.planes}) != len(self.planes):
            raise ValueError("plane names must be unique")
        if any(plane.index != index for index, plane in enumerate(self.planes)):
            raise ValueError("plane index must match its position")
        if self.frame_count < 1 or not 0 <= self.frame_index < self.frame_count:
            raise ValueError("frame index is outside the image")
        if self.frame_delays and len(self.frame_delays) != self.frame_count:
            raise ValueError("frame delays must cover every frame, or none of them")

    def plane(self, name: str) -> SamplePlane:
        return plane_named(self.planes, name)

    @property
    def any_converted(self) -> bool:
        return any(plane.origin is SampleOrigin.CONVERTED for plane in self.planes)


def plane_named(planes: tuple[SamplePlane, ...], name: str) -> SamplePlane:
    """The plane of that name; ``KeyError`` when the set has none."""
    try:
        return next(plane for plane in planes if plane.name == name)
    except StopIteration:
        raise KeyError(name) from None


def ensure_inside(image: LoadedImage, coord: PixelCoord) -> None:
    if not 0 <= coord.x < image.width or not 0 <= coord.y < image.height:
        raise ValueError(f"pixel ({coord.x}, {coord.y}) is outside {image.width}x{image.height}")


@dataclass(frozen=True, slots=True)
class ExtractOrder:
    """How the selected bits are laid into the extracted byte stream.

    ``planes`` is a preference over plane names rather than a permutation of one
    image's planes: the planes carrying bits are ranked by their position in it,
    and names it does not mention keep the image's own order, after those.
    """

    planes: tuple[str, ...] = ()
    bit_order: BitOrder = BitOrder.MSB
    scan: ScanOrder = ScanOrder.XY

    def __post_init__(self) -> None:
        if len(set(self.planes)) != len(self.planes):
            raise ValueError("channel order repeats a plane name")


@dataclass(frozen=True, slots=True, eq=False)
class ViewerState:
    """``selection is None`` means every bit of every plane (the original image)."""

    image: LoadedImage | None = None
    cursor: PixelCoord | None = None
    selection: frozenset[BitChoice] | None = None
    focus: BitChoice | None = None
    filter_expr: str = ""
    value_format: DisplayFormat = DisplayFormat.HEX
    extract_encoding: ExtractEncoding = ExtractEncoding.ASCII
    extract_order: ExtractOrder = ExtractOrder()
    adjust: ViewAdjust = ViewAdjust()
    zoom: float = 1.0
    only_matched: bool = False

    def __post_init__(self) -> None:
        if not MIN_ZOOM <= self.zoom <= MAX_ZOOM:
            raise ValueError(f"zoom {self.zoom} is outside {MIN_ZOOM}..{MAX_ZOOM}")
        if self.image is None:
            if self.cursor is not None or self.focus is not None or self.selection is not None:
                raise ValueError("cursor, focus, and selection require an image")
            return
        if self.selection is not None:
            for choice in self.selection:
                _require_choice(self.image, choice)
        if self.focus is not None:
            _require_choice(self.image, self.focus)
        if self.cursor is not None:
            ensure_inside(self.image, self.cursor)


def _require_choice(image: LoadedImage, choice: BitChoice) -> None:
    try:
        plane = image.plane(choice.plane)
    except KeyError as exc:
        raise ValueError(f"unknown plane: {choice.plane}") from exc
    if choice.bit >= plane.bit_depth:
        raise ValueError(f"bit {choice.bit} is outside 0..{plane.bit_depth - 1}")

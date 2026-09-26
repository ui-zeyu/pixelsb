"""Immutable image samples, the mask stack over them, and viewer state."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type SampleArray = NDArray[np.uint16]
type RgbArray = NDArray[np.uint8]
type RgbaArray = NDArray[np.uint8]
type ChannelArray = NDArray[np.uint8]
type IndexArray = NDArray[np.intp]

MIN_ZOOM = 1.0
MAX_ZOOM = 128.0
# The widest level a value mask can name, whatever the image's planes are; the
# useful ceiling for one image is the widest of its own planes (level_ceiling).
MAX_LEVEL = 0xFFFF
# The cat map's parameter bounds: a brute force finds large a and b, but past
# int32 they stop fitting the arithmetic any canvas would run them in.
ARNOLD_TIMES_MAX = 4096
ARNOLD_PARAM_LIMIT = 0x7FFF_FFFF
# The color trio a picture is composed from, in the order the channels carry it.
COLOR_SLOTS = ("R", "G", "B")


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

    @property
    def maximum(self) -> int:
        """The largest sample the plane holds."""
        return (1 << self.bit_depth) - 1


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


def level_ceiling(planes: tuple[SamplePlane, ...]) -> int:
    """The highest level a value mask can mean here: the widest plane's own maximum."""
    return max((plane.maximum for plane in planes), default=0xFF)


def plane_or_none(planes: tuple[SamplePlane, ...], name: str) -> SamplePlane | None:
    """The plane of that name as spelled, or ``None`` when the set has none."""
    return next((plane for plane in planes if plane.name == name), None)


def plane_named(planes: tuple[SamplePlane, ...], name: str) -> SamplePlane:
    """The plane of that name; ``KeyError`` when the set has none."""
    plane = plane_or_none(planes, name)
    if plane is None:
        raise KeyError(name)
    return plane


def ensure_inside(image: LoadedImage, coord: PixelCoord) -> None:
    if not 0 <= coord.x < image.width or not 0 <= coord.y < image.height:
        raise ValueError(f"pixel ({coord.x}, {coord.y}) is outside {image.width}x{image.height}")


# --- masks -----------------------------------------------------------------
#
# A mask is one transformation of the pixels. They are plain frozen data: what a
# mask means for a raster is decided in ``domain.stack``, so a mask can sit in
# the viewer state, be compared, and be a cache key without carrying any array.


@dataclass(frozen=True, slots=True)
class BitsMask:
    """The bits of each channel the canvas paints and the stream packs."""

    selection: frozenset[BitChoice]


@dataclass(frozen=True, slots=True)
class RegionMask:
    """A display-filter expression: the pixels it passes are the ones that survive."""

    expression: str


@dataclass(frozen=True, slots=True)
class InvertMask:
    """Every channel read upside down, at the plane's own maximum."""


@dataclass(frozen=True, slots=True)
class GrayscaleMask:
    """The color channels replaced by the pixel's brightness.

    A picture with no R, G and B to combine is gray already, so this leaves it
    alone.
    """


@dataclass(frozen=True, slots=True)
class ThresholdMask:
    """Every channel pushed to 0 or its maximum, split at ``level``.

    The level is compared against the channel as stored, so it reaches as high as
    the widest plane does: 255 for an 8-bit picture, 65535 for a 16-bit one.
    """

    level: int = 128

    def __post_init__(self) -> None:
        if not 0 <= self.level <= MAX_LEVEL:
            raise ValueError(f"threshold level must be within 0..{MAX_LEVEL}")


@dataclass(frozen=True, slots=True)
class XorMask:
    """Every channel exclusive-ored with ``value``, which flips the bits it sets."""

    value: int = 0xFF

    def __post_init__(self) -> None:
        if not 0 <= self.value <= MAX_LEVEL:
            raise ValueError(f"xor value must be within 0..{MAX_LEVEL}")


@dataclass(frozen=True, slots=True)
class CropMask:
    """The canvas folded onto the pixels that survived: only they are drawn."""


@dataclass(frozen=True, slots=True)
class FftMask:
    """Every channel the selection carries, replaced by its log-magnitude spectrum.

    A view of the frequency domain rather than a reversible edit: the bright
    points away from the center are the periodic patterns a frequency-domain
    watermark leaves, and the value masks read them as ordinary pixels. The
    bits mask below this one decides which channels take part: uncheck a
    channel there and its samples keep their stored values.
    """


@dataclass(frozen=True, slots=True)
class ArnoldMask:
    """The pixels rearranged by the inverse cat map, ``times`` times over.

    This is the recovery direction: a picture scrambled by the forward map with
    parameters ``(a, b)`` is read again by the same parameters here, because the
    inverse of ``[[1, b], [a, ab+1]]`` is exactly ``[[ab+1, -b], [-a, 1]]``. It
    needs the full square canvas: after the mix a cell's pixels come from one
    row *and* one column at once, so the raster's own coordinate bookkeeping
    cannot follow them, and coordinates start over from the transformed picture.
    """

    times: int = 1
    a: int = 1
    b: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.times <= ARNOLD_TIMES_MAX:
            raise ValueError(f"cat map times must be within 1..{ARNOLD_TIMES_MAX}")
        if abs(self.a) > ARNOLD_PARAM_LIMIT or abs(self.b) > ARNOLD_PARAM_LIMIT:
            raise ValueError(f"cat map parameters must be within ±{ARNOLD_PARAM_LIMIT}")


type Mask = (
    BitsMask
    | RegionMask
    | InvertMask
    | GrayscaleMask
    | ThresholdMask
    | XorMask
    | CropMask
    | FftMask
    | ArnoldMask
)


@dataclass(frozen=True, slots=True)
class Layer:
    """One mask in the stack, with the switch that turns it on and off."""

    mask: Mask
    enabled: bool = True

    def __post_init__(self) -> None:
        # The alias's own union is what isinstance takes; the alias itself is not.
        if not isinstance(self.mask, Mask.__value__):
            raise ValueError(f"a layer holds a mask, not {type(self.mask).__name__}")


@dataclass(frozen=True, slots=True)
class MaskFailure:
    """A mask the stack could not apply, and what went wrong."""

    index: int
    message: str


# --- the layer stack's product ---------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Raster:
    """The pixels every consumer reads: the canvas, the readout, the extractor.

    ``samples`` are the values the enabled masks left behind, ``selection`` the
    bits of them the canvas paints and the stream packs, ``live`` the pixels the
    region masks kept (``None`` = every pixel), and ``rows``/``columns`` the
    source coordinates of the raster's cells. ``None`` there means the cells are
    the image's own, in order, which is what a crop mask takes away from.
    """

    samples: SampleArray
    planes: tuple[SamplePlane, ...]
    selection: frozenset[BitChoice]
    live: NDArray[np.bool_] | None = None
    rows: IndexArray | None = None
    columns: IndexArray | None = None
    failures: tuple[MaskFailure, ...] = ()

    def __post_init__(self) -> None:
        if self.samples.dtype != np.uint16 or self.samples.ndim != 3:
            raise ValueError("samples must be a uint16 array of shape (height, width, planes)")
        samples = np.ascontiguousarray(self.samples)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        if samples.shape[2] != len(self.planes):
            raise ValueError("sample channels do not match the planes")
        if self.live is not None:
            if self.live.shape != samples.shape[:2]:
                raise ValueError("the live mask must cover the raster, one flag per cell")
            self.live.setflags(write=False)
        if self.rows is not None and self.rows.shape != (self.height,):
            raise ValueError("rows must hold one source row per raster row")
        if self.columns is not None and self.columns.shape != (self.width,):
            raise ValueError("columns must hold one source column per raster column")

    @property
    def height(self) -> int:
        return self.samples.shape[0]

    @property
    def width(self) -> int:
        return self.samples.shape[1]

    @property
    def cropped(self) -> bool:
        """Whether the cells were gathered from a slice of the image."""
        return self.rows is not None

    @property
    def live_count(self) -> int:
        """How many source pixels the region masks left standing."""
        return self.width * self.height if self.live is None else int(self.live.sum())

    def source_at(self, dx: int, dy: int) -> PixelCoord:
        """The source pixel drawn at raster cell ``(dx, dy)``."""
        x = dx if self.columns is None else int(self.columns[dx])
        y = dy if self.rows is None else int(self.rows[dy])
        return PixelCoord(x, y)

    def cell_of(self, coord: PixelCoord) -> tuple[int, int] | None:
        """Where a source pixel is drawn, or ``None`` when the raster has no cell for it.

        A pixel a region mask dropped is faded rather than gone, so it keeps its
        cell and its numbers can be read; ``live`` is what says whether it takes
        part in the extraction. Only a crop takes cells away.
        """
        column = _index_of(self.columns, coord.x, self.width)
        row = _index_of(self.rows, coord.y, self.height)
        if column is None or row is None:
            return None
        return column, row

    def live_in(self, x0: int, y0: int, x1: int, y1: int) -> int:
        """How many live pixels sit inside a source rectangle, edges included."""
        rows = np.arange(self.height, dtype=np.intp) if self.rows is None else self.rows
        columns = np.arange(self.width, dtype=np.intp) if self.columns is None else self.columns
        band = (rows >= y0) & (rows <= y1)
        span = (columns >= x0) & (columns <= x1)
        if self.live is None:
            return int(band.sum() * span.sum())
        return int(self.live[np.ix_(band, span)].sum())


def _index_of(index: IndexArray | None, value: int, extent: int) -> int | None:
    """Where a source coordinate sits along one axis, or ``None`` when it is off."""
    if index is None:
        return value if 0 <= value < extent else None
    position = int(np.searchsorted(index, value))
    return position if position < extent and index[position] == value else None


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


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """One animated frame's own rectangle on the canvas, and its delay.

    A GIF places each frame at an offset and at its own size, which is where a
    flag can hide; a full-canvas frame is the ordinary case.
    """

    index: int
    width: int
    height: int
    x: int
    y: int
    delay: int


@dataclass(frozen=True, slots=True, eq=False)
class ViewerState:
    """What the viewer is showing: an image, the mask stack over it, and the view.

    ``layers`` runs bottom to top, so the first mask sees the image itself and
    the last one sees everything below it. Neither ``cursor`` nor ``focus``
    belongs to a mask: they are where the keyboard is, not what the pixels are.
    """

    image: LoadedImage | None = None
    cursor: PixelCoord | None = None
    focus: BitChoice | None = None
    layers: tuple[Layer, ...] = ()
    value_format: DisplayFormat = DisplayFormat.HEX
    extract_encoding: ExtractEncoding = ExtractEncoding.ASCII
    extract_order: ExtractOrder = ExtractOrder()
    zoom: float = 1.0

    def __post_init__(self) -> None:
        if not MIN_ZOOM <= self.zoom <= MAX_ZOOM:
            raise ValueError(f"zoom {self.zoom} is outside {MIN_ZOOM}..{MAX_ZOOM}")
        if self.image is None:
            if self.cursor is not None or self.focus is not None or self.layers:
                raise ValueError("cursor, focus, and layers require an image")
            return
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

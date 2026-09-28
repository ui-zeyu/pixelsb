"""The layer stack: what the masks make of an image.

Every consumer reads one :class:`Raster` — the canvas, the cursor readout, the
extract stream — so a mask changes the picture and the bytes at once. Masks run
bottom to top, and each one sees what the masks under it left behind: a region
mask over a threshold compares the thresholded values, a spectrum over a
projection is the spectrum of the bit plane, and the masks read nothing else.
There is no view outside the stack: which bits are on show is itself a mask,
and its output is the picture every reader gets.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    COLOR_SLOTS,
    GRAY_SLOTS,
    ArnoldMask,
    BitChoice,
    BitsMask,
    CombineMask,
    CombineOp,
    CropMask,
    FftMask,
    GrayscaleMask,
    IndexArray,
    InvertMask,
    Layer,
    LoadedImage,
    Mask,
    MaskFailure,
    Raster,
    RegionMask,
    SampleArray,
    SamplePlane,
    ThresholdMask,
    XorMask,
    plane_or_none,
)
from pixelsb.domain.predicate import PredicateError, compile_filter
from pixelsb.domain.selection import bits_for
from pixelsb.domain.spectrum import stretched

# Rec. 709 luma weights, scaled into integer arithmetic.
_LUMA = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_SCALE = 10_000

# One reading per combine operation, in the widened plane the two pictures
# meet at. The bitwise and extremal ops ignore the maximum; the arithmetic
# two follow StegSolve: ADD halves the sum so the result stays in range, and
# SUB halves the difference and lifts it to the plane's midpoint, so two
# equal planes read flat gray and a difference stands out around it.
type Combine = Callable[[SampleArray, SampleArray, int], SampleArray]


def _subtracted(ours: SampleArray, theirs: SampleArray, maximum: int) -> SampleArray:
    """Half the difference toward the other picture, lifted by the midpoint.

    The division truncates toward zero rather than flooring, because the
    readings this matches are the ones StegSolve's Java ints produced.
    """
    delta = ours.astype(np.int32) - theirs
    half = np.sign(delta) * (np.abs(delta) // 2)
    return (half + (maximum + 1) // 2).astype(np.uint16)


_COMBINE: dict[CombineOp, Combine] = {
    CombineOp.XOR: lambda ours, theirs, _maximum: np.bitwise_xor(ours, theirs),
    CombineOp.AND: lambda ours, theirs, _maximum: np.bitwise_and(ours, theirs),
    CombineOp.OR: lambda ours, theirs, _maximum: np.bitwise_or(ours, theirs),
    CombineOp.MIN: lambda ours, theirs, _maximum: np.minimum(ours, theirs),
    CombineOp.MAX: lambda ours, theirs, _maximum: np.maximum(ours, theirs),
    CombineOp.ADD: lambda ours, theirs, _maximum: ((ours.astype(np.uint32) + theirs) // 2).astype(
        np.uint16
    ),
    CombineOp.SUB: _subtracted,
}


def resolve(image: LoadedImage, layers: tuple[Layer, ...]) -> Raster:
    """Apply the enabled masks, bottom to top: each one reads the one below it.

    A mask that cannot apply here — a region expression that will not compile,
    a crop with no region under it, the cat map wanting a square, a projection
    naming a channel the picture lacks — keeps everything and reports itself
    against its layer.
    """
    raster = Raster(samples=image.samples, planes=image.planes)
    failures: list[MaskFailure] = []
    for index, layer in enumerate(layers):
        if not layer.enabled:
            continue
        try:
            raster = apply_mask(raster, layer.mask)
        except (PredicateError, ValueError) as exc:
            failures.append(MaskFailure(index, str(exc)))
    return replace(raster, failures=tuple(failures))


def apply_mask(raster: Raster, mask: Mask) -> Raster:
    """One mask's effect on a raster."""
    match mask:
        case BitsMask(selection=selection):
            return _projected(raster, selection)
        case RegionMask(expression=expression):
            keep = compile_filter(expression, raster.planes).evaluate(raster)
            return replace(raster, live=keep if raster.live is None else raster.live & keep)
        case InvertMask():
            return replace(raster, samples=_maxima(raster.planes) - raster.samples)
        case GrayscaleMask():
            return replace(raster, samples=_gray(raster.samples, raster.planes))
        case ThresholdMask(level=level):
            lit = raster.samples > np.uint16(level)
            return replace(raster, samples=np.where(lit, _maxima(raster.planes), np.uint16(0)))
        case XorMask(value=value):
            return replace(raster, samples=raster.samples ^ np.uint16(value))
        case CropMask():
            return _cropped(raster)
        case FftMask(planes=planes):
            return replace(raster, samples=_spectrum(raster, planes))
        case CombineMask(op=op, other=other):
            return _combined(raster, op, other)
        case ArnoldMask(times=times, a=a, b=b):
            return _arnold(raster, times, a, b)
        case _ as unknown:
            assert_never(unknown)


def arnold_indices(size: int, times: int, a: int, b: int) -> tuple[IndexArray, IndexArray]:
    """The cat map's destination grids after ``times`` steps: ``(rows, columns)`` of
    the source cell each destination reads.

    The matrix acts on ``(row, column)`` coordinates, the order the common
    arnold_encode/arnold_decode scripts write their loops in — ``a`` multiplies
    the row on the encode's second line, ``b`` the column on its first — so the
    parameters a challenge's script names go here unchanged. The map is linear,
    so ``times`` steps are one power of its 2 by 2 matrix: the power is raised by
    repeated squaring, and the grids are then built in a single pass. The
    coefficients are reduced mod ``size`` before they meet an axis, so even the
    large parameters a brute force finds stay inside int64 arithmetic.
    """
    m00, m01, m10, m11 = _matrix_power((a * b + 1, -b, -a, 1), times, size)
    rows = np.arange(size, dtype=np.int64)[:, None]
    columns = np.arange(size, dtype=np.int64)[None, :]
    new_row = (m00 * rows + m01 * columns) % size
    new_column = (m10 * rows + m11 * columns) % size
    return new_row.astype(np.intp), new_column.astype(np.intp)


def _matrix_power(
    base: tuple[int, int, int, int],
    times: int,
    size: int,
) -> tuple[int, int, int, int]:
    """``base`` raised to ``times``, every entry reduced mod ``size``."""
    result = (1, 0, 0, 1)

    def multiply(
        left: tuple[int, int, int, int], right: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        return (
            (left[0] * right[0] + left[1] * right[2]) % size,
            (left[0] * right[1] + left[1] * right[3]) % size,
            (left[2] * right[0] + left[3] * right[2]) % size,
            (left[2] * right[1] + left[3] * right[3]) % size,
        )

    while times:
        if times & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        times >>= 1
    return result


def arnold_image(samples: SampleArray, times: int, a: int, b: int) -> SampleArray:
    """The samples after ``times`` cat-map steps: the picture a viewer would paint."""
    if samples.shape[0] != samples.shape[1]:
        raise ValueError("cat map needs a square picture")
    source_y, source_x = arnold_indices(samples.shape[0], times, a, b)
    return _scattered(samples, source_y, source_x)


def _arnold(raster: Raster, times: int, a: int, b: int) -> Raster:
    """The cat map over the samples as a whole; the live mask travels with the pixels.

    Any square block of samples will do — a crop that left a square still holds
    all of its cells. Coordinates restart from the transformed picture: after
    the mix a cell's source row and column stop being separable, and what the
    user wants from this mask is the restored picture's own pixels and numbers.
    """
    if raster.height != raster.width:
        raise ValueError(f"猫脸变换要方图：这张是 {raster.width}×{raster.height}")
    source_y, source_x = arnold_indices(raster.height, times, a, b)
    return replace(
        raster,
        samples=_scattered(raster.samples, source_y, source_x),
        live=None if raster.live is None else _scattered(raster.live, source_y, source_x),
        rows=None,
        columns=None,
    )


def _scattered(array: NDArray, source_y: IndexArray, source_x: IndexArray) -> NDArray:
    """The array with every cell moved to where the map sends it."""
    out = np.empty_like(array)
    out[source_y, source_x] = array
    return out


def _projected(raster: Raster, selection: frozenset[BitChoice]) -> Raster:
    """The projected picture: named channels rebuilt from their chosen bits.

    Each chosen channel's bits pack onto the low end in ascending order and its
    depth shrinks to how many were chosen, so the value every later reader sees
    is the one the canvas used to paint over the lens. Channels carrying none
    of the selection are out of the picture — the projection names the picture
    it makes, and a plane with no bits in it has nothing to show there.
    """
    missing = sorted(
        {choice.plane for choice in selection} - {plane.name for plane in raster.planes}
    )
    if missing:
        raise ValueError(f"位选择的通道这张图没有：{'、'.join(missing)}")
    if not selection:
        raise ValueError("位选择是空的：至少要选一位")
    channels: list[NDArray] = []
    planes: list[SamplePlane] = []
    for index, plane in enumerate(raster.planes):
        bits = bits_for(selection, plane.name)
        if not bits:
            continue
        channels.append(_packed(raster.samples[:, :, index], bits))
        planes.append(replace(plane, index=len(planes), bit_depth=len(bits)))
    return replace(raster, samples=np.stack(channels, axis=-1), planes=tuple(planes))


def _packed(channel: NDArray, bits: tuple[int, ...]) -> NDArray:
    """The channel's chosen bits, packed onto the low end in ascending order."""
    top = (1 << len(bits)) - 1
    low, high = bits[0], bits[-1]
    if high - low + 1 == len(bits):  # a contiguous run: one shift, one mask
        return (channel >> low) & top
    packed = np.zeros(channel.shape, dtype=channel.dtype)
    for offset, bit in enumerate(bits):
        packed |= ((channel >> bit) & 1) << offset
    return packed


def _spectrum(raster: Raster, planes: tuple[str, ...] | None) -> SampleArray:
    """The named channels, each as its log-magnitude spectrum.

    Each channel stretches to its own maximum, so a faint watermark shows as
    well in blue as in red. ``None`` names the color trio and gray, whatever
    of them the image has — alpha never takes part by default, since its
    spectrum is a near-dark picture. A channel the mask does not name keeps
    its samples.
    """
    if planes is None:
        names = (*COLOR_SLOTS, *GRAY_SLOTS)
        wanted = {plane.name for plane in raster.planes if plane.name in names}
    else:
        wanted = set(planes)
        missing = wanted - {plane.name for plane in raster.planes}
        if missing:
            raise ValueError(f"频谱的通道这张图没有：{'、'.join(sorted(missing))}")
    if not wanted:
        return raster.samples  # no channel takes part, so no sample changes
    out = np.array(raster.samples)
    for index, plane in enumerate(raster.planes):
        if plane.name not in wanted:
            continue
        band = stretched(raster.samples[:, :, index], plane.maximum)
        out[:, :, index] = band if band is not None else 0
    return out


def ensure_combinable(base: Raster | LoadedImage, other: LoadedImage) -> None:
    """Whether two pictures can combine at all: one size, and planes to share.

    The command judges the fit where both pictures are known, before the layer
    lands; the stack judges it again at fold time, because an open picture
    carries the stack to the next image, where the same two answers are the
    layer's own failure.
    """
    if base.width != other.width or base.height != other.height:
        raise ValueError(
            f"合成要同尺寸的图：这张是 {base.width}×{base.height}，"
            f"另一张是 {other.width}×{other.height}"
        )
    ours = {plane.name for plane in base.planes}
    theirs = {plane.name for plane in other.planes}
    if not ours & theirs:
        ours_names = "、".join(plane.name for plane in base.planes)
        theirs_names = "、".join(plane.name for plane in other.planes)
        raise ValueError(f"两张图没有同名的通道：这张有 {ours_names}，另一张有 {theirs_names}")


def _combined(raster: Raster, op: CombineOp, other: LoadedImage) -> Raster:
    """The raster combined with the other picture, plane by shared plane.

    A plane the other picture lacks keeps its samples, and a plane only the
    other picture has is dropped — this raster's picture stays the frame of
    reference. Shared planes meet at the wider of their two depths, so an
    8-bit and a 16-bit channel of one name combine in 16 bits. The fit the
    command judged before the layer landed is judged again here, because the
    stack travels to whatever picture opens next.
    """
    ensure_combinable(raster, other)
    channels: list[NDArray] = []
    planes: list[SamplePlane] = []
    for index, plane in enumerate(raster.planes):
        mate = plane_or_none(other.planes, plane.name)
        if mate is None:
            channels.append(raster.samples[:, :, index])
            planes.append(plane)
            continue
        depth = max(plane.bit_depth, mate.bit_depth)
        channels.append(
            _COMBINE[op](
                raster.samples[:, :, index], other.samples[:, :, mate.index], (1 << depth) - 1
            )
        )
        planes.append(replace(plane, index=len(planes), bit_depth=depth))
    return replace(raster, samples=np.stack(channels, axis=-1), planes=tuple(planes))


def match_span(live: NDArray[np.bool_]) -> tuple[IndexArray, IndexArray] | None:
    """The rows and columns a live mask touches; ``None`` when it touches none."""
    columns = np.flatnonzero(live.any(axis=0))
    rows = np.flatnonzero(live.any(axis=1))
    if columns.size == 0 or rows.size == 0:
        return None
    return rows, columns


def _maxima(planes: tuple[SamplePlane, ...]) -> NDArray[np.uint16]:
    """One maximum per plane, to broadcast against a whole raster."""
    return np.array([plane.maximum for plane in planes], dtype=np.uint16)


def _gray(samples: SampleArray, planes: tuple[SamplePlane, ...]) -> SampleArray:
    """The color channels replaced by their brightness; anything else is left alone."""
    indexes = _color_indexes(planes)
    if indexes is None:
        return samples
    luma = (samples[:, :, list(indexes)].astype(np.uint32) * _LUMA).sum(axis=-1) // _LUMA_SCALE
    converted = samples.copy()
    converted[..., list(indexes)] = luma[..., None]  # one broadcast write, all three slots
    return converted


def _color_indexes(planes: tuple[SamplePlane, ...]) -> tuple[int, int, int] | None:
    """Where the color trio sits in the planes, or ``None`` for a gray picture."""
    red, green, blue = (plane_or_none(planes, name) for name in COLOR_SLOTS)
    if red is None or green is None or blue is None:
        return None
    return red.index, green.index, blue.index


def _cropped(raster: Raster) -> Raster:
    """Crop the raster to the rows and columns its live pixels occupy."""
    if raster.live is None:
        raise ValueError("裁剪要有一条先落下的区域操作：没有通过的像素可收")
    span = match_span(raster.live)
    if span is None:
        return _nothing_left(raster)
    rows, columns = span
    return replace(
        raster,
        samples=raster.samples[np.ix_(rows, columns)],
        live=raster.live[np.ix_(rows, columns)],
        rows=rows if raster.rows is None else raster.rows[rows],
        columns=columns if raster.columns is None else raster.columns[columns],
    )


def _nothing_left(raster: Raster) -> Raster:
    """The raster a stack with nothing left to show has: no cells at all."""
    return replace(
        raster,
        samples=raster.samples[:0, :0],
        live=np.zeros((0, 0), dtype=bool),
        rows=np.empty(0, dtype=np.intp) if raster.rows is None else raster.rows[:0],
        columns=np.empty(0, dtype=np.intp) if raster.columns is None else raster.columns[:0],
    )

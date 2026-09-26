"""The layer stack: what the masks make of an image.

Every consumer reads one :class:`Raster` — the canvas, the cursor readout, the
extract stream — so a mask changes the picture and the bytes at once. Masks run
bottom to top, and each one sees what the masks under it left behind: a region
mask over a threshold compares the thresholded values, not the stored ones.
"""

from dataclasses import replace
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import (
    COLOR_SLOTS,
    ArnoldMask,
    BitsMask,
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
from pixelsb.domain.selection import all_bits

# Rec. 709 luma weights, scaled into integer arithmetic.
_LUMA = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_SCALE = 10_000


def resolve(image: LoadedImage, layers: tuple[Layer, ...]) -> Raster:
    """Apply the enabled masks, bottom to top, onto the image's own pixels."""
    raster = Raster(samples=image.samples, planes=image.planes, selection=all_bits(image.planes))
    failures: list[MaskFailure] = []
    for index, layer in enumerate(layers):
        if not layer.enabled:
            continue
        try:
            raster = apply_mask(raster, layer.mask)
        except (PredicateError, ValueError) as exc:
            # A region expression that will not compile or run, and a transform
            # that refuses this raster (the cat map wants the whole square),
            # keep everything; the panel reports it against the layer.
            failures.append(MaskFailure(index, str(exc)))
    return replace(raster, failures=tuple(failures))


def apply_mask(raster: Raster, mask: Mask) -> Raster:
    """One mask's effect on a raster."""
    match mask:
        case BitsMask(selection=selection):
            return replace(raster, selection=selection & all_bits(raster.planes))
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
        case FftMask():
            return replace(raster, samples=_spectrum(raster))
        case ArnoldMask(times=times, a=a, b=b):
            return _arnold(raster, times, a, b)
        case _ as unknown:
            assert_never(unknown)


def arnold_indices(size: int, times: int, a: int, b: int) -> tuple[IndexArray, IndexArray]:
    """The cat map's destination grids after ``times`` steps: ``out[ny, nx] = in``.

    The map is linear, so ``times`` steps are one power of its 2 by 2 matrix: the
    power is raised by repeated squaring — twelve multiplies for 4096 steps,
    however large ``times`` grows — and the grids are then built in a single
    pass. The coefficients are reduced mod ``size`` before they meet an axis,
    so even the large parameters a brute force finds stay inside int64
    arithmetic.
    """
    m00, m01, m10, m11 = _matrix_power((a * b + 1, -b, -a, 1), times, size)
    columns = np.arange(size, dtype=np.int64)[None, :]
    rows = np.arange(size, dtype=np.int64)[:, None]
    new_x = (m00 * columns + m01 * rows) % size
    new_y = (m10 * columns + m11 * rows) % size
    return new_y.astype(np.intp), new_x.astype(np.intp)


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
    out = np.empty_like(samples)
    out[source_y, source_x] = samples
    return out


def _arnold(raster: Raster, times: int, a: int, b: int) -> Raster:
    """The cat map over the whole canvas; the live mask travels with the pixels.

    Coordinates restart from the transformed picture — that is the honest
    reading, since after the mix a cell's source row and column stop being
    separable, and what the user wants from this mask is the restored picture's
    own pixels and numbers.
    """
    if raster.rows is not None:
        raise ValueError("猫脸变换要作用在完整画幅上：被裁剪过的图没有完整的方形坐标")
    if raster.height != raster.width:
        raise ValueError(f"猫脸变换要方图：这张是 {raster.width}×{raster.height}")
    source_y, source_x = arnold_indices(raster.height, times, a, b)
    return replace(
        raster,
        samples=_scattered(raster.samples, source_y, source_x),
        live=None if raster.live is None else _scattered(raster.live, source_y, source_x),
    )


def _scattered(array: NDArray, source_y: IndexArray, source_x: IndexArray) -> NDArray:
    """The array with every cell moved to where the map sends it."""
    out = np.empty_like(array)
    out[source_y, source_x] = array
    return out


def _spectrum(raster: Raster) -> SampleArray:
    """The channels the selection carries, each as its log-magnitude spectrum.

    The logarithm goes twice: once because the spectrum's dynamic range is far
    past what 16 bits would show linearly, and again because the DC peak — the
    whole picture's average brightness — would otherwise eat the range and
    leave everything but the center dot in the dark. Each channel stretches to
    its own maximum, so a faint watermark shows as well in blue as in red.

    A channel with no bit selected keeps its samples: the canvas paints none of
    it, so a spectrum there would be work nobody sees. That is the dial for the
    planes a spectrum does not help — alpha and palette indexes — and it is the
    bits mask below this one that works it, like any other stacked mask.
    """
    painted = {choice.plane for choice in raster.selection}
    if not any(plane.name in painted for plane in raster.planes):
        return raster.samples  # no channel takes part, so no sample changes
    out = np.array(raster.samples)
    for index, plane in enumerate(raster.planes):
        if plane.name not in painted:
            continue
        magnitude = np.abs(np.fft.fft2(raster.samples[:, :, index].astype(np.float64)))
        view = np.log1p(np.log1p(np.fft.fftshift(magnitude)))
        peak = float(view.max())
        out[:, :, index] = (
            np.rint(view * (plane.maximum / peak)).astype(np.uint16) if peak > 0.0 else 0
        )
    return out


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
    gray = luma.astype(np.uint16)
    converted = samples.copy()
    for index in indexes:
        converted[:, :, index] = gray
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
        return raster
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

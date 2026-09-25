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
    BitsMask,
    CropMask,
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
        except (OverflowError, PredicateError) as exc:
            # A region expression that will not compile or run keeps everything;
            # the panel reports it against the layer that carries it.
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
        case _ as unknown:
            assert_never(unknown)


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

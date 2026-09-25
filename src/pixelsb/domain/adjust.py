"""View post-processing laid on the composed pixels, after the bit math."""

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import RgbArray, ViewAdjust

# Rec. 709 luma weights, scaled into uint32 arithmetic.
_LUMA = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_SCALE = 10_000


def apply_adjust(rgb: RgbArray, adjust: ViewAdjust) -> RgbArray:
    """Return the pixels with the adjustments laid on, unchanged when all are off.

    Grayscale comes first so the threshold reads the brightness, and the invert
    lands last, so a threshold flip still reads as the negative it is.
    """
    out: NDArray[np.uint8] = rgb
    if adjust.grayscale:
        luma = (out.astype(np.uint32) * _LUMA).sum(axis=-1) // _LUMA_SCALE
        out = np.stack([luma, luma, luma], axis=-1).astype(np.uint8)
    if adjust.threshold:
        lit = out > np.uint8(adjust.level)
        out = np.where(lit, np.uint8(255), np.uint8(0)).astype(np.uint8)
    if adjust.invert:
        out = np.uint8(255) - out
    return out

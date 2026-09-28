"""The log-magnitude spectrum: one definition for the fft mask and the page."""

import numpy as np
from numpy.typing import NDArray


def log_magnitude(samples: NDArray[np.uint16]) -> NDArray[np.float64]:
    """One plane's spectrum: |fft2|, DC centered, squashed through two logs.

    The logarithm goes twice: the spectrum's dynamic range runs far past what
    16 bits would show linearly, and the DC peak — the picture's average
    brightness — would otherwise eat the range and leave everything but the
    center dot in the dark.
    """
    magnitude = np.abs(np.fft.fft2(samples.astype(np.float64)))
    return np.log1p(np.log1p(np.fft.fftshift(magnitude)))


def stretched(samples: NDArray[np.uint16], ceiling: int) -> NDArray[np.uint16] | None:
    """The log magnitude stretched so its own peak reaches ``ceiling``.

    ``None`` means the spectrum never leaves zero, so there is no peak to
    stretch to; the reader decides what that means — the fft mask keeps the
    channel dark, the spectrum page skips the slot.
    """
    view = log_magnitude(samples)
    peak = float(view.max())
    if peak <= 0.0:
        return None
    return np.rint(view * (ceiling / peak)).astype(np.uint16)

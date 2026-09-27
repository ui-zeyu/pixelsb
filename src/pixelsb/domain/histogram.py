"""Channel histograms, and the chi-square read of a bit plane's value pairs.

The histogram is the picture's value census: how many cells hold each value the
raster carries — the samples as the stack leaves them, a projection's packing
included, a crop already taken away, and only the pixels the region masks keep.
The chi-square read judges one bit plane over those same values: values that
differ only in that bit form a pair, and writing the bit trades pixels inside
each pair — so a plane something was written into has pair counts sitting at
their average, and the statistic collapses toward its degrees of freedom.
"""

import math

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import Raster, SamplePlane

_SERIES_TERMS = 200_000
_FRACTION_TERMS = 200_000
_TAIL_SIGMAS = 80.0  # past this many standard deviations the double has no digits left
_MAX_SHOWN = 8  # the chi-square columns, the plane's low bits


def shown_bits(plane: SamplePlane) -> tuple[int, ...]:
    """The bits of one plane the page reads: its low bits, to at most eight."""
    return tuple(range(min(plane.bit_depth, _MAX_SHOWN)))


def plane_counts(raster: Raster, plane: SamplePlane) -> tuple[NDArray[np.int64], int]:
    """The census of one plane as the raster holds it, over the pixels the region kept.

    The values are the samples the stack left: a projection's packing included,
    a crop already taken away, and only the pixels the region masks left
    standing — the population about to be extracted.
    """
    channel = raster.samples[:, :, plane.index]
    return (
        np.bincount(_kept(channel, raster.live), minlength=plane.maximum + 1).astype(np.int64),
        plane.maximum,
    )


def _kept(values: NDArray, live: NDArray[np.bool_] | None) -> NDArray:
    """The values of the cells the region masks left standing, straightened out."""
    return values[live].ravel() if live is not None else values.ravel()


def pair_chi_square(counts: NDArray[np.int64], bit: int) -> tuple[float, int] | None:
    """The chi-square statistic over the value pairs ``bit`` joins, and its degrees.

    The counts must divide evenly into the pairs the bit asks for — a whole
    plane's census does. Pairs no pixel reached are left out, and ``None`` means
    fewer than two of them remained: the statistic needs a degree to lose.
    """
    step = 1 << bit
    if counts.size < 2 * step or counts.size % (2 * step):
        raise ValueError("counts must divide into whole pairs for the bit asked")
    grouped = counts.reshape(-1, 2 * step)
    low = grouped[:, :step].astype(np.float64)
    high = grouped[:, step:].astype(np.float64)
    expected = (low + high) / 2.0
    reached = expected > 0
    pairs = int(reached.sum())
    if pairs < 2:
        return None
    deviation = (low[reached] - expected[reached]) ** 2 / expected[reached]
    return float(deviation.sum()), pairs - 1


def chi2_tail(statistic: float, degrees: int) -> float:
    """The chi-square tail: P(χ² ≥ statistic) at ``degrees`` of freedom.

    The regularized upper incomplete gamma Q(degrees/2, statistic/2): the
    power series below the integrand's shoulder, Lentz's continued fraction
    past it, either way to a relative 1e-15. A statistic so far out in the tail
    that a double has no digits for it reads as 0 — past eighty standard
    deviations the exact value would not change a reading of the picture.
    """
    if degrees < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if statistic <= 0.0:
        return 1.0
    a = degrees / 2.0
    x = statistic / 2.0
    if x > a + _TAIL_SIGMAS * math.sqrt(a):
        return 0.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_fraction(a, x)


def _log_prefactor(a: float, x: float) -> float:
    """``ln`` of x^(a-1) e^-x / Γ(a), the factor both expansions multiply."""
    return -x + a * math.log(x) - math.lgamma(a)


def _gamma_p_series(a: float, x: float) -> float:
    """The lower incomplete gamma P(a, x) as its power series."""
    term = 1.0 / a
    total = term
    for n in range(1, _SERIES_TERMS):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(_log_prefactor(a, x))


def _gamma_q_fraction(a: float, x: float) -> float:
    """The upper incomplete gamma Q(a, x) as a continued fraction (modified Lentz)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _FRACTION_TERMS):
        numerator = -i * (i - a)
        b += 2.0
        d = numerator * d + b
        d = tiny if abs(d) < tiny else d
        c = b + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(_log_prefactor(a, x)) * h

"""The value census and the pair test: what they count, and when they agree."""

import math
from collections.abc import Sequence

import numpy as np
import pytest

from pixelsb.domain import histogram
from pixelsb.domain.models import BitChoice, Raster
from tests.support import planes_rgb


def counts_of(values: np.ndarray | Sequence[int]) -> np.ndarray:
    """A plane's value histogram from stored values."""
    return np.bincount(np.asarray(values, dtype=np.uint16), minlength=256).astype(np.int64)


def test_the_census_counts_every_cell_of_its_plane() -> None:
    samples = np.zeros((2, 3, 3), dtype=np.uint16)
    samples[:, :, 1] = 7
    raster_built = Raster(samples=samples, planes=planes_rgb(), selection=frozenset())
    counts, top = histogram.plane_counts(raster_built, raster_built.planes[1])
    assert counts.size == 256
    assert counts.sum() == 6
    assert counts[7] == 6
    assert top == 255


def test_the_census_keeps_only_the_bits_the_selection_carries() -> None:
    """A restricted plane counts masked values: bit 0 alone, the rest read as zero."""
    samples = np.zeros((2, 3, 3), dtype=np.uint16)
    samples[:, :, 1] = np.array([1, 2, 3, 0, 1, 3]).reshape(2, 3)
    raster_built = Raster(
        samples=samples,
        planes=planes_rgb(),
        selection=frozenset({BitChoice("G", 0)}),
    )
    counts, top = histogram.plane_counts(raster_built, raster_built.planes[1])
    assert top == 1
    assert counts.tolist() == [2, 4]
    assert histogram.shown_bits(raster_built, raster_built.planes[1]) == (0,)
    assert histogram.shown_bits(raster_built, raster_built.planes[0]) == tuple(range(8))


def test_the_census_spans_the_mask_and_the_pair_test_stays_on_stored_values() -> None:
    """One carried bit at position 4: the chart counts 0 and 16, the test reads 0..255."""
    samples = np.zeros((2, 3, 3), dtype=np.uint16)
    samples[:, :, 0] = np.array([0, 16, 1, 17, 0, 16]).reshape(2, 3)
    raster_built = Raster(
        samples=samples,
        planes=planes_rgb(),
        selection=frozenset({BitChoice("R", 4)}),
    )
    counts, top = histogram.plane_counts(raster_built, raster_built.planes[0])
    assert top == 16
    assert counts.tolist() == [3] + [0] * 15 + [3]
    stored = histogram.stored_counts(raster_built, raster_built.planes[0])
    assert stored.size == 256
    verdict = histogram.pair_chi_square(stored, 4)
    assert verdict == (0.0, 1)  # pairs (0,16) and (1,17) both in balance


def test_the_pair_test_joins_each_value_with_its_bit_partner() -> None:
    # pair (0, 1): 4 against 0 -> (4-2)²/2 = 2; pair (2, 3): 4 against 0 -> 2.
    verdict = histogram.pair_chi_square(counts_of([0, 0, 0, 0, 2, 2, 2, 2]), 0)
    assert verdict is not None
    statistic, degrees = verdict
    assert statistic == pytest.approx(4.0)
    assert degrees == 1
    # one degree of freedom: the tail is erfc(sqrt(statistic/2)) exactly
    assert histogram.chi2_tail(statistic, degrees) == pytest.approx(math.erfc(math.sqrt(2.0)))


def test_a_bit_written_at_random_flattens_the_pairs() -> None:
    rng = np.random.default_rng(11)
    base = np.tile(np.arange(0, 256, 2), 256)  # every even value, no odd one: lopsided pairs
    written = (base & ~1) | rng.integers(0, 2, base.size)  # the LSB write halves each even value
    plain = histogram.pair_chi_square(counts_of(base), 0)
    embedded = histogram.pair_chi_square(counts_of(written), 0)
    assert plain is not None
    assert embedded is not None
    assert histogram.chi2_tail(*plain) < 0.05
    assert histogram.chi2_tail(*embedded) > 0.95


def test_pairs_no_pixel_reached_are_left_out() -> None:
    verdict = histogram.pair_chi_square(np.array([5, 5, 0, 0], dtype=np.int64), 0)
    assert verdict is None  # one pair at most: nothing to test against
    assert histogram.pair_chi_square(np.zeros(256, dtype=np.int64), 0) is None


def test_counts_that_do_not_align_with_the_bit_are_refused() -> None:
    # a bit-3 pair row spans eight values: 100 does not divide into them
    with pytest.raises(ValueError, match="whole pairs"):
        histogram.pair_chi_square(np.zeros(100, dtype=np.int64), 3)
    with pytest.raises(ValueError, match="whole pairs"):
        histogram.pair_chi_square(np.zeros(7, dtype=np.int64), 0)


def test_the_tail_matches_its_closed_forms() -> None:
    for value in (0.5, 3.841458820694124, 10.0, 55.0):
        assert histogram.chi2_tail(value, 1) == pytest.approx(math.erfc(math.sqrt(value / 2)))
        if value < 40:  # exp(-20) still sits inside pytest.approx's rel range
            assert histogram.chi2_tail(value, 2) == pytest.approx(math.exp(-value / 2))


def test_the_tail_is_one_at_zero_and_dies_in_the_far_tail() -> None:
    assert histogram.chi2_tail(0.0, 127) == 1.0
    assert histogram.chi2_tail(0.0, 1) == 1.0
    assert histogram.chi2_tail(1e12, 127) == 0.0
    with pytest.raises(ValueError, match="degrees"):
        histogram.chi2_tail(1.0, 0)


def test_the_series_and_the_fraction_agree_where_they_overlap() -> None:
    """Two expansions of the same integral must agree wherever both run."""
    a, x = 4.0, 6.0  # past the shoulder: the fraction's home range, the series still converges
    by_fraction = histogram._gamma_q_fraction(a, x)
    by_series = 1.0 - histogram._gamma_p_series(a, x)
    assert by_fraction == pytest.approx(by_series, rel=1e-10)


def test_the_tail_falls_monotonically_across_a_wide_range() -> None:
    tails = [histogram.chi2_tail(value, 32) for value in (1.0, 8.0, 31.0, 64.0, 500.0)]
    assert tails == sorted(tails, reverse=True)

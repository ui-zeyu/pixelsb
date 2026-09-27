"""The log-magnitude spectrum: the one definition both readers share."""

import numpy as np

from pixelsb.domain.spectrum import log_magnitude


def test_the_view_is_the_double_log_of_the_centered_magnitude() -> None:
    rng = np.random.default_rng(7)
    samples = rng.integers(0, 256, (8, 8)).astype(np.uint16)
    shifted = np.fft.fftshift(np.abs(np.fft.fft2(samples.astype(np.float64))))
    assert np.array_equal(log_magnitude(samples), np.log1p(np.log1p(shifted)))


def test_a_constant_plane_lights_only_its_dc() -> None:
    view = log_magnitude(np.full((4, 6), 9, np.uint16))
    lit = np.argwhere(view > 0.0)
    assert lit.tolist() == [[2, 3]]  # the DC sits at the center of the shift

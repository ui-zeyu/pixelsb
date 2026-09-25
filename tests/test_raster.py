"""The raster: where its cells come from, and which source pixels it shows."""

import numpy as np
import pytest

from pixelsb.domain.models import CropMask, LoadedImage, PixelCoord, Raster, RegionMask
from tests.support import layers, make_image, planes_rgb, raster


def _image(height: int = 3, width: int = 4) -> LoadedImage:
    samples = np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3)
    return make_image(samples, planes_rgb())


def test_an_uncropped_raster_maps_a_pixel_to_the_cell_it_is() -> None:
    base = raster(_image())
    assert base.source_at(2, 1) == PixelCoord(2, 1)
    assert base.cell_of(PixelCoord(2, 1)) == (2, 1)
    assert base.cell_of(PixelCoord(4, 0)) is None
    assert base.cell_of(PixelCoord(0, 3)) is None


def test_a_cropped_raster_maps_between_both_spaces() -> None:
    cropped = raster(_image(), RegionMask("left >= 1 and left <= 2"), CropMask())
    assert cropped.source_at(0, 0) == PixelCoord(1, 0)
    assert cropped.cell_of(PixelCoord(2, 1)) == (1, 1)
    assert cropped.cell_of(PixelCoord(0, 0)) is None  # cropped away


def test_a_pixel_a_region_mask_dropped_is_still_drawn_but_not_live() -> None:
    """Fading is not removal: the pixel keeps its cell, and `live` says it is out."""
    marked = raster(_image(), RegionMask("R > 0"))
    live = marked.live
    assert live is not None
    assert marked.cell_of(PixelCoord(0, 0)) == (0, 0)
    assert not live[0, 0]
    assert live[0, 1]


def test_the_live_count_and_the_rectangle_count_agree_with_the_mask() -> None:
    marked = raster(_image(), RegionMask("top == 0"))
    assert marked.live_count == 4
    assert marked.live_in(0, 0, 3, 2) == 4
    assert marked.live_in(0, 1, 3, 2) == 0
    assert marked.live_in(1, 0, 2, 0) == 2
    assert raster(_image()).live_in(0, 0, 1, 1) == 4  # no region mask: everything counts


def test_live_in_counts_against_the_source_rectangle_after_a_crop() -> None:
    cropped = raster(_image(), RegionMask("left >= 2"), CropMask())
    assert cropped.live_in(2, 0, 3, 2) == 6
    assert cropped.live_in(0, 0, 1, 2) == 0


def test_the_raster_refuses_samples_and_planes_that_do_not_match() -> None:
    image = _image()
    with pytest.raises(ValueError, match="channels do not match"):
        Raster(samples=image.samples, planes=image.planes[:2], selection=frozenset())


def test_the_raster_refuses_a_live_mask_of_the_wrong_shape() -> None:
    image = _image()
    with pytest.raises(ValueError, match="live mask"):
        Raster(
            samples=image.samples,
            planes=image.planes,
            selection=frozenset(),
            live=np.ones((1, 1), dtype=bool),
        )


def test_the_raster_refuses_row_and_column_indices_of_the_wrong_length() -> None:
    image = _image()
    with pytest.raises(ValueError, match="rows must hold one source row"):
        Raster(
            samples=image.samples,
            planes=image.planes,
            selection=frozenset(),
            rows=np.arange(2, dtype=np.intp),
        )
    with pytest.raises(ValueError, match="columns must hold one source column"):
        Raster(
            samples=image.samples,
            planes=image.planes,
            selection=frozenset(),
            columns=np.arange(2, dtype=np.intp),
        )


def test_a_layer_holds_its_mask_and_its_switch() -> None:
    mask = RegionMask("R > 0")
    (layer,) = layers(mask)
    assert layer.mask == mask
    assert layer.enabled
    assert not layers(mask, enabled=False)[0].enabled

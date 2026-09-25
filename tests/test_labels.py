"""The numbers drawn inside the pixels, and the font that has to fit in one."""

import numpy as np

from pixelsb.domain.labels import (
    font_pixel_size,
    label_fits,
    region_texts,
    widest_text,
    zoom_required,
)
from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    DisplayFormat,
    Mask,
    Raster,
    RegionMask,
)
from tests.support import make_image, planes_rgb, raster

_SAMPLES = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.uint16)
_LSBS = frozenset({BitChoice("R", 0), BitChoice("G", 0)})


def _board(*masks: Mask) -> Raster:
    return raster(make_image(_SAMPLES, planes_rgb()), *masks)


def test_labels_use_the_cell_bigness_instead_of_tiny_fonts() -> None:
    assert font_pixel_size(20, "FF") == 13
    assert font_pixel_size(128, "1") == 92
    assert font_pixel_size(63, "R FF\nG FF\nB FF") == 16
    assert zoom_required("1") == 9
    assert zoom_required("FF") == 9
    assert label_fits(9, "1")
    assert not label_fits(8, "1")
    assert label_fits(9, "FF")
    assert not label_fits(8, "FF")
    # One line per active channel: the stack needs a taller cell, not a smaller font.
    assert zoom_required("R FF\nG FF\nB FF") == 21
    assert label_fits(21, "R FF\nG FF\nB FF")
    assert not label_fits(20, "R FF\nG FF\nB FF")
    assert zoom_required("R1\nG0") == 15
    assert label_fits(15, "R1\nG0")


def test_region_texts_reads_a_block_row_major() -> None:
    board = _board(BitsMask(_LSBS))
    assert region_texts(board, DisplayFormat.DECIMAL, 0, 0, 2, 1) == ["R:1\nG:0", "R:0\nG:1"]
    assert region_texts(board, DisplayFormat.DECIMAL, 0, 0, 1, 1) == ["R:1\nG:0"]
    assert region_texts(board, DisplayFormat.DECIMAL, 5, 5, 9, 9) == []
    assert region_texts(board, DisplayFormat.DECIMAL, 2, 0, 4, 1) == []  # past the raster


def test_one_channel_carries_the_label_without_naming_it() -> None:
    board = _board(BitsMask(frozenset({BitChoice("R", 0)})))
    assert region_texts(board, DisplayFormat.DECIMAL, 0, 0, 2, 1) == ["1", "0"]


def test_nothing_selected_labels_nothing() -> None:
    board = _board(BitsMask(frozenset()))
    assert region_texts(board, DisplayFormat.DECIMAL, 0, 0, 2, 1) == ["", ""]
    assert widest_text(board, DisplayFormat.DECIMAL) == ""


def test_labels_read_what_a_cropped_raster_holds() -> None:
    board = _board(BitsMask(_LSBS), RegionMask("left >= 1"), CropMask())
    assert region_texts(board, DisplayFormat.DECIMAL, 0, 0, 1, 1) == ["R:0\nG:1"]


def test_the_font_is_fitted_to_the_widest_label_the_selection_can_make() -> None:
    board = _board(BitsMask(_LSBS))
    assert widest_text(board, DisplayFormat.DECIMAL) == "R:1\nG:1"
    assert widest_text(board, DisplayFormat.BINARY) == "R:1\nG:1"
    wide = _board(BitsMask(frozenset({BitChoice("R", bit) for bit in range(8)})))
    assert widest_text(wide, DisplayFormat.HEX) == "FF"

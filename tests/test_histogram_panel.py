"""The histogram page: what it counts, and when a bit reads as written."""

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import BitChoice, BitsMask, LoadedImage, RegionMask, XorMask
from pixelsb.io.loading import image_from_pixels
from pixelsb.ui.histogram_panel import HistogramPanel
from tests.support import raster


def _image(tmp_path: Path, samples: np.ndarray) -> LoadedImage:
    return image_from_pixels(tmp_path / "memory.png", samples)


def test_the_page_counts_the_canvas_channels(qtbot: QtBot, tmp_path: Path) -> None:
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    image = _image(tmp_path, np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(image, raster(image))
    assert [data.name for data in panel._chart._planes] == ["R", "G", "B"]
    assert panel._cells["R", 7] is not None
    assert panel._empty.isHidden()
    assert not panel._grid_host.isHidden()


def test_a_written_bit_reads_hot_and_a_plain_one_stays_cold(qtbot: QtBot, tmp_path: Path) -> None:
    """A random LSB flattens the value pairs: that bit's cell takes the warning tint.

    The picture holds only values below 128, so bit 7's pairs are lopsided —
    one member full, the other empty (there is no 128) — and its p stays cold.
    """
    rng = np.random.default_rng(5)
    values = np.tile(np.arange(0, 128, 2, dtype=np.uint8), (64, 1))
    bits = rng.integers(0, 2, values.shape).astype(np.uint8)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[..., 0] = (values & np.uint8(0xFE)) | bits
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    image = _image(tmp_path, pixels)
    panel.set_source(image, raster(image))
    hot = panel._cells["R", 0]
    assert hot.property("hot") is True
    assert hot.text() != "—"
    cold = panel._cells["R", 7]
    assert cold.property("hot") is not True


def test_the_grid_follows_the_planes_the_image_has(qtbot: QtBot, tmp_path: Path) -> None:
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    image = _image(tmp_path, np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(image, raster(image))
    assert ("A", 0) not in panel._cells
    alpha = _image(tmp_path, np.zeros((4, 4, 4), dtype=np.uint8))
    panel.set_source(alpha, raster(alpha))
    assert ("A", 0) in panel._cells


def test_a_recipe_change_recounts_the_canvas(qtbot: QtBot, tmp_path: Path) -> None:
    """The page reads the stack's result, so a mask moves the histogram."""
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 10
    image = _image(tmp_path, pixels)
    panel.set_source(image, raster(image))
    before = panel._chart._planes[0].shape[10]
    panel.set_source(image, raster(image, XorMask(0xFF)))
    after = panel._chart._planes[0].shape[10]
    assert before > 0.0
    assert after == 0.0  # 10 ^ 0xFF moved every pixel away from 10


def test_a_projection_moves_the_census_and_the_columns(qtbot: QtBot, tmp_path: Path) -> None:
    """The page reads the raster's planes: a one-bit projection, one value column.

    The projection is in the stack, so the raster carries R alone and the grid
    shrinks to the bits it has; the balanced LSB still lands hot in that column.
    """
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[..., 0] = [0, 1, 2, 3, 0, 1, 2, 3]
    image = _image(tmp_path, pixels)
    panel.set_source(image, raster(image))
    panel.set_source(image, raster(image, BitsMask(frozenset({BitChoice("R", 0)}))))
    view = panel._chart._planes[0]
    assert view.top == 1  # the census spans the projected values 0 and 1
    assert ("R", 7) not in panel._cells
    assert ("R", 0) in panel._cells
    assert ("G", 0) not in panel._cells  # planes the projection drops leave the grid
    assert panel._cells["R", 0].text() == "—"  # one value pair: nothing to test against


def test_a_region_change_recounts_the_pixels_it_kept(qtbot: QtBot, tmp_path: Path) -> None:
    """A region decides what counts: circle a payload and the read follows it."""
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:2, :2, 0] = 200  # four bright pixels in the corner the region keeps
    image = _image(tmp_path, pixels)
    panel.set_source(image, raster(image))
    before = panel._chart._planes[0].shape
    panel.set_source(image, raster(image, RegionMask("rect(0, 0, 2, 2)")))
    after = panel._chart._planes[0].shape
    assert before is not after
    assert before[0] > 0.0  # the whole canvas counts, zeros included
    assert after[0] == 0.0  # the dropped pixels no longer count
    assert after[200] > 0.0  # only the four kept pixels are left


def test_no_image_shows_the_note(qtbot: QtBot) -> None:
    panel = HistogramPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_source(None, None)
    assert panel._empty.isVisible()
    assert panel._chart._planes == ()

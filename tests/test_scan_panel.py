"""The scan page: which candidates an image gets, and how a hit lands."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import Raster, RegionMask
from pixelsb.domain.scan import ScanHit, scan_candidates
from pixelsb.io.loading import image_from_pixels
from pixelsb.ui.scan_panel import ScanPanel
from tests.support import planes_rgb, raster

_PLANES = planes_rgb()


def test_each_channel_set_sweeps_every_bit_in_every_order() -> None:
    candidates = scan_candidates(_PLANES)
    # R, G, B alone plus rgb and bgr: five sets, eight bits, four read orders.
    assert len(candidates) == 5 * 8 * 4
    first = candidates[0]
    assert first.command == "r.0"
    assert first.order.planes == ("R",)
    assert {choice.plane for choice in first.selection} == {"R"}
    bgr = next(candidate for candidate in candidates if candidate.order.planes == ("B", "G", "R"))
    assert bgr.command == "b.0 or g.0 or r.0"


def test_a_channel_set_the_image_lacks_drops_out_whole() -> None:
    from pixelsb.domain.models import SampleOrigin, SamplePlane

    gray = (SamplePlane("L", 0, 8, SampleOrigin.RAW),)
    candidates = scan_candidates(gray)
    assert len(candidates) == 1 * 8 * 4
    assert {candidate.command for candidate in candidates} == {f"l.{bit}" for bit in range(8)}


def test_the_run_button_names_the_action_a_click_takes(qtbot: QtBot, tmp_path: Path) -> None:
    """One button: it starts the sweep, and clicking it again stops the run."""
    from pixelsb.ui import text

    panel = ScanPanel()
    qtbot.addWidget(panel)
    image = image_from_pixels(tmp_path / "memory.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(image, raster(image))
    assert panel._run.text() == text.SWEEP_START
    panel._toggle_run()
    assert panel._worker is not None
    assert panel._run.text() == text.SWEEP_STOP
    panel._toggle_run()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    assert panel._run.text() == text.SWEEP_START


def test_a_hit_takes_its_place_the_moment_it_lands(qtbot: QtBot) -> None:
    """Rows go up while the sweep runs, and a flag jumps above the guesses already up."""
    panel = ScanPanel()
    qtbot.addWidget(panel)
    candidates = scan_candidates(_PLANES)
    earlier = ScanHit(candidates[0], "txt")
    flagged = ScanHit(candidates[1], "", ("flag{here}",))
    panel._on_scored(earlier)
    assert panel._tree.topLevelItemCount() == 1  # up before the run is over
    panel._on_scored(flagged)
    top, second = panel._tree.topLevelItem(0), panel._tree.topLevelItem(1)
    assert top is not None
    assert second is not None
    assert top.data(0, Qt.ItemDataRole.UserRole) is flagged
    assert second.data(0, Qt.ItemDataRole.UserRole) is earlier


def test_a_new_image_stops_a_run_and_clears_the_list(qtbot: QtBot, tmp_path: Path) -> None:
    panel = ScanPanel()
    qtbot.addWidget(panel)
    one = image_from_pixels(tmp_path / "one.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(one, raster(one))
    panel._on_start()
    two = image_from_pixels(tmp_path / "two.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(two, raster(two))
    assert panel._worker is None
    assert panel._tree.topLevelItemCount() == 0


def test_a_new_recipe_keeps_the_hit_list(qtbot: QtBot, tmp_path: Path) -> None:
    """Clicking a hit rewrites the stack, and that must not wipe what the user reads."""
    panel = ScanPanel()
    qtbot.addWidget(panel)
    image = image_from_pixels(tmp_path / "memory.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(image, raster(image))
    panel._on_scored(ScanHit(scan_candidates(image.planes)[0], "txt"))
    assert panel._tree.topLevelItemCount() == 1
    panel.set_source(image, raster(image))  # a fresh canvas object, same picture
    assert panel._tree.topLevelItemCount() == 1
    next_image = image_from_pixels(tmp_path / "next.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(next_image, raster(next_image))
    assert panel._tree.topLevelItemCount() == 0


def test_the_sweep_reads_the_canvas_not_the_file(qtbot: QtBot, tmp_path: Path) -> None:
    """The scan shares the other pages' source: the stack's result, not the file."""
    image = image_from_pixels(tmp_path / "white.png", np.full((4, 4, 3), 255, dtype=np.uint8))
    canvas = Raster(samples=np.zeros_like(image.samples), planes=image.planes)
    panel = ScanPanel()
    qtbot.addWidget(panel)
    panel.set_source(image, canvas)
    panel._on_start()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    rows = [panel._tree.topLevelItem(at) for at in range(panel._tree.topLevelItemCount())]
    row = next(item for item in rows if item is not None and item.text(0) == "r.7")
    assert row is not None
    # the file is all 0xFF: a ÿ in the preview would mean it swept the file
    assert "ÿ" not in row.text(2)


def test_the_sweep_reads_only_the_live_pixels(qtbot: QtBot, tmp_path: Path) -> None:
    """A region mask filters the sweep the way it filters the extract panel.

    Row 0's LSBs spell one ASCII byte and row 1's are all ones: a sweep that
    ignored the region would pack both rows into ``A.``; the live-only sweep
    reads ``A`` alone.
    """
    pixels = np.zeros((2, 8, 3), dtype=np.uint8)
    pixels[0, :, 0] = [0, 0x01, 0, 0, 0, 0, 0, 0x01]  # 0b01000001 = "A"
    pixels[1, :, 0] = 0xFF
    image = image_from_pixels(tmp_path / "region.png", pixels)
    panel = ScanPanel()
    qtbot.addWidget(panel)
    panel.set_source(image, raster(image, RegionMask("top < 1")))
    panel._on_start()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    rows = [panel._tree.topLevelItem(at) for at in range(panel._tree.topLevelItemCount())]
    previews = [item.text(2) for item in rows if item is not None and item.text(0) == "r.0"]
    assert "A" in previews
    assert "A." not in previews


def test_the_result_line_names_the_type_and_at_most_two_flags() -> None:
    from pixelsb.ui import text

    hit = ScanHit(scan_candidates(_PLANES)[0], "txt", ("flag{a}", "flag{a}", "flag{b}", "flag{c}"))
    assert text.sweep_result(hit) == "txt · flag{a} · flag{b} · 另有 1 个"
    assert text.sweep_result(ScanHit(scan_candidates(_PLANES)[0])) == text.SWEEP_NOTHING


def test_each_row_carries_a_printable_preview(qtbot: QtBot, tmp_path: Path) -> None:
    panel = ScanPanel()
    qtbot.addWidget(panel)
    image = image_from_pixels(tmp_path / "memory.png", np.zeros((4, 4, 3), dtype=np.uint8))
    panel.set_source(image, raster(image))
    panel._on_start()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    row = panel._tree.topLevelItem(0)
    assert row is not None
    assert row.text(2) != "" or row.text(3) == "—"  # a taste of the stream, or nothing to show
    assert panel._tree.columnCount() == 4

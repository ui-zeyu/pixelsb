"""The scan page: which candidates an image gets, and how a hit lands."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from pixelsb.domain.scan import ScanHit, scan_candidates
from pixelsb.io.loading import image_from_pixels
from pixelsb.ui.scan_panel import ScanPanel
from tests.support import planes_rgb

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


def test_the_sweep_lists_every_candidate_and_applies_the_picked_one(
    qtbot: QtBot, tmp_path: Path
) -> None:
    panel = ScanPanel()
    qtbot.addWidget(panel)
    panel.set_image(image_from_pixels(tmp_path / "memory.png", np.zeros((4, 4, 3), dtype=np.uint8)))
    panel._on_start()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    assert panel._tree.topLevelItemCount() == panel._total
    picked: list[object] = []
    panel.apply_requested.connect(picked.append)
    row = panel._tree.topLevelItem(3)
    assert row is not None
    panel._on_clicked(row, 0)
    assert picked == [row.data(0, Qt.ItemDataRole.UserRole)]


def test_a_new_image_stops_a_run_and_clears_the_list(qtbot: QtBot, tmp_path: Path) -> None:
    panel = ScanPanel()
    qtbot.addWidget(panel)
    panel.set_image(image_from_pixels(tmp_path / "one.png", np.zeros((4, 4, 3), dtype=np.uint8)))
    panel._on_start()
    panel.set_image(image_from_pixels(tmp_path / "two.png", np.zeros((4, 4, 3), dtype=np.uint8)))
    assert panel._worker is None
    assert panel._tree.topLevelItemCount() == 0


def test_the_result_line_names_the_type_and_at_most_two_flags() -> None:
    from pixelsb.ui import text

    hit = ScanHit(scan_candidates(_PLANES)[0], "txt", ("flag{a}", "flag{a}", "flag{b}", "flag{c}"))
    assert text.sweep_result(hit) == "txt · flag{a} · flag{b} · 另有 1 个"
    assert text.sweep_result(ScanHit(scan_candidates(_PLANES)[0])) == text.SWEEP_NOTHING


def test_each_row_carries_a_printable_preview(qtbot: QtBot, tmp_path: Path) -> None:
    panel = ScanPanel()
    qtbot.addWidget(panel)
    panel.set_image(image_from_pixels(tmp_path / "memory.png", np.zeros((4, 4, 3), dtype=np.uint8)))
    panel._on_start()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    row = panel._tree.topLevelItem(0)
    assert row is not None
    assert row.text(2) != "" or row.text(3) == "—"  # a taste of the stream, or nothing to show
    assert panel._tree.columnCount() == 4

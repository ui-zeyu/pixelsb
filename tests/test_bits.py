"""The bit grid: what each box, row header, and column header shows."""

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import BitChoice, SampleOrigin, SamplePlane
from pixelsb.ui.bits import BitMatrix
from tests.support import planes_rgb


def _matrix(qtbot: QtBot) -> BitMatrix:
    matrix = BitMatrix()
    qtbot.addWidget(matrix)
    return matrix


def _states(boxes: Iterable[QCheckBox]) -> set[Qt.CheckState]:
    return {box.checkState() for box in boxes}


def test_the_original_reading_checks_every_box(qtbot: QtBot) -> None:
    matrix = _matrix(qtbot)
    matrix.set_layer(planes_rgb(), None)
    assert _states(matrix._boxes.values()) == {Qt.CheckState.Checked}
    assert _states(matrix._row_boxes.values()) == {Qt.CheckState.Checked}
    assert _states(matrix._col_boxes.values()) == {Qt.CheckState.Checked}


def test_a_partly_selected_group_shows_a_partial_state(qtbot: QtBot) -> None:
    matrix = _matrix(qtbot)
    matrix.set_layer(planes_rgb(), frozenset({BitChoice("R", 0)}))
    assert matrix._boxes[BitChoice("R", 0)].checkState() is Qt.CheckState.Checked
    assert matrix._boxes[BitChoice("R", 1)].checkState() is Qt.CheckState.Unchecked
    # One bit of the row, and of the column, is what "part of it" looks like.
    assert matrix._row_boxes["R"].checkState() is Qt.CheckState.PartiallyChecked
    assert matrix._row_boxes["G"].checkState() is Qt.CheckState.Unchecked
    assert matrix._col_boxes[0].checkState() is Qt.CheckState.PartiallyChecked
    assert matrix._col_boxes[7].checkState() is Qt.CheckState.Unchecked


def test_a_narrow_plane_only_draws_the_bits_it_has(qtbot: QtBot) -> None:
    matrix = _matrix(qtbot)
    planes = (*planes_rgb(), SamplePlane("L", 3, 4, SampleOrigin.RAW))
    matrix.set_layer(planes, None)
    assert BitChoice("L", 3) in matrix._boxes
    assert BitChoice("L", 4) not in matrix._boxes
    # The columns past a 4-bit plane still exist for the channels that have them.
    assert sorted(matrix._col_boxes) == list(range(8))

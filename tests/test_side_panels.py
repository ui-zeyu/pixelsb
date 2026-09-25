"""The rail and the pages it switches."""

from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from pixelsb.ui.side_panels import Panel, SidePanels


def _panels(qtbot: QtBot) -> tuple[SidePanels, QLabel, QLabel]:
    panels = SidePanels()
    qtbot.addWidget(panels)
    bits, info = QLabel("bits"), QLabel("info")
    panels.add(Panel.EXTRACT, "▦", "", bits)
    panels.add(Panel.INFO, "ⓘ", "", info)
    return panels, bits, info


def test_the_first_page_starts_on_top(qtbot: QtBot) -> None:
    panels, bits, _info = _panels(qtbot)
    assert panels.current is Panel.EXTRACT
    assert panels._stack.currentWidget() is bits
    assert panels._buttons[Panel.EXTRACT].isChecked()
    assert not panels._buttons[Panel.INFO].isChecked()


def test_the_rail_and_the_call_both_switch_pages(qtbot: QtBot) -> None:
    panels, bits, info = _panels(qtbot)
    panels._buttons[Panel.INFO].click()
    assert panels.current is Panel.INFO
    assert panels._stack.currentWidget() is info
    panels.set_current(Panel.EXTRACT)
    assert panels.current is Panel.EXTRACT
    assert panels._stack.currentWidget() is bits
    assert panels._buttons[Panel.EXTRACT].isChecked()
    assert not panels._buttons[Panel.INFO].isChecked()
    # Raising the page that is already up is a no-op, not a jump back.
    panels.set_current(Panel.INFO)
    assert panels._stack.currentWidget() is info
    assert panels._buttons[Panel.INFO].isChecked()

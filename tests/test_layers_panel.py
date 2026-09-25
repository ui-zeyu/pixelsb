"""The layer panel: the stack as a list, and the editor of the selected mask."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QCheckBox
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    GrayscaleMask,
    InvertMask,
    Layer,
    Mask,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ThresholdMask,
    ViewerState,
    XorMask,
)
from pixelsb.domain.selection import channel_members
from pixelsb.ui import text
from pixelsb.ui.layers import LayerPanel
from tests.support import board_of, make_image, planes_rgb
from tests.support import layers as layers_of


def _state(*masks: Mask) -> ViewerState:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("a.png"))
    return ViewerState(image=image, layers=layers_of(*masks))


def _panel(qtbot: QtBot, *masks: Mask) -> LayerPanel:
    panel = LayerPanel()
    qtbot.addWidget(panel)
    panel.resize(360, 700)
    panel.show()
    _show(panel, *masks)
    return panel


def _show(panel: LayerPanel, *masks: Mask) -> None:
    state = _state(*masks)
    panel.set_state(state, board_of(state))


def _rows(panel: LayerPanel) -> list[tuple[int | None, str, str]]:
    """The list top down: each row's layer index, name, and the parameters it shows."""
    return [
        (index, panel._rows[index]._name.text(), panel._rows[index]._detail.text())
        for index in reversed(range(len(panel._state.layers)))
    ] + [(None, panel._rows[None]._name.text(), panel._rows[None]._detail.text())]


def test_the_list_reads_top_down_with_the_image_itself_at_the_bottom(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask(), RegionMask("R > 0"))
    assert _rows(panel) == [
        (1, "区域", "R > 0"),
        (0, "反相", "inv"),
        (None, text.LAYER_BASE, "2×2 · 模式 RGB · 单帧"),
    ]


def test_the_base_row_reads_the_file_that_is_open(qtbot: QtBot) -> None:
    """The image belongs in the list's cache key, or the row keeps an older file's facts."""
    panel = LayerPanel()
    qtbot.addWidget(panel)
    panel.set_state(ViewerState(), None)
    assert _rows(panel)[-1][2] == ""
    first = _state(InvertMask())
    panel.set_state(first, board_of(first))
    assert _rows(panel)[-1][2] == "2×2 · 模式 RGB · 单帧"
    other = ViewerState(
        image=make_image(
            np.zeros((7, 5, 1), dtype=np.uint16),
            (SamplePlane("L", 0, 8, SampleOrigin.RAW),),
        ),
        layers=layers_of(InvertMask()),
    )
    panel.set_state(other, board_of(other))
    assert _rows(panel)[-1][2] == "5×7 · 模式 RGB · 单帧"


def test_a_shadowed_bits_mask_says_so(qtbot: QtBot) -> None:
    lower = BitsMask(frozenset({BitChoice("R", 0)}))
    panel = _panel(qtbot, lower, BitsMask(frozenset({BitChoice("G", 0)})))
    panel._select(0)
    assert panel._note.text() == text.MASK_BITS_SHADOWED
    panel._select(1)
    assert panel._note.text() == text.LAYER_BITS_NOTE


def test_a_switched_off_bits_mask_shadows_nothing(qtbot: QtBot) -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    state = ViewerState(
        image=image,
        layers=(
            Layer(BitsMask(frozenset({BitChoice("R", 0)}))),
            Layer(BitsMask(frozenset({BitChoice("G", 0)})), enabled=False),
        ),
    )
    panel = LayerPanel()
    qtbot.addWidget(panel)
    panel.set_state(state, board_of(state))
    panel._select(0)
    assert panel._note.text() == text.LAYER_BITS_NOTE


def test_the_rows_name_what_each_mask_holds(qtbot: QtBot) -> None:
    panel = _panel(
        qtbot,
        BitsMask(
            frozenset(
                {
                    BitChoice("R", 0),
                    BitChoice("R", 1),
                    BitChoice("R", 2),
                    BitChoice("G", 5),
                }
            )
        ),
        ThresholdMask(200),
        XorMask(0x0F),
        CropMask(),
        GrayscaleMask(),
    )
    assert [name for _index, name, _detail in _rows(panel)] == [
        "灰度",
        "裁剪",
        "异或",
        "阈值",
        "位选择",
        text.LAYER_BASE,
    ]
    details = {name: detail for _index, name, detail in _rows(panel)}
    assert details["位选择"] == "r.0 or r.1 or r.2 or g.5"  # every grid state reads as its command
    assert details["阈值"] == "thr 200"
    assert details["异或"] == "xor 0x0F"


def test_a_single_bit_selection_reads_as_the_same_command_the_box_writes(
    qtbot: QtBot,
) -> None:
    """The grid click and the command agree: one channel's one bit is ``b.0``."""
    panel = _panel(qtbot, BitsMask(frozenset({BitChoice("B", 0)})))
    assert _rows(panel)[0][2] == "b.0"
    _show(panel, BitsMask(channel_members(planes_rgb(), "B")))
    assert _rows(panel)[0][2] == "b"


def test_a_new_mask_takes_the_selection(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask())
    assert panel.selected == 0
    asked: list[Mask] = []
    panel.add_requested.connect(asked.append)
    panel.add_requested.emit(ThresholdMask(150))
    assert asked == [ThresholdMask(150)]
    _show(panel, InvertMask(), ThresholdMask(150))  # what the window does with it next
    assert panel.selected == 1
    assert panel._note.text() == text.LAYER_COMMAND_NOTE


def test_a_bits_layer_points_at_the_extract_panel(qtbot: QtBot) -> None:
    panel = _panel(qtbot, BitsMask(frozenset({BitChoice("R", 0)})))
    assert panel._note.text() == text.LAYER_BITS_NOTE
    panel._select(None)
    assert panel._note.text() == text.LAYER_BASE_NOTE


def test_a_region_mask_points_at_the_filter_box(qtbot: QtBot) -> None:
    panel = _panel(qtbot, RegionMask("R > 0"))
    assert panel._note.text() == text.LAYER_REGION_NOTE


def test_the_base_row_explains_itself_and_cannot_be_removed(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask())
    panel._select(None)
    assert panel._rows[None]._check is None  # the image has no switch to flick
    assert panel._note.text() == text.LAYER_BASE_NOTE
    asked: list[object] = []
    panel.remove_requested.connect(asked.append)
    panel.move_requested.connect(lambda layer, step: asked.append((layer, step)))
    assert not panel._remove.isEnabled()
    assert not panel._up.isEnabled()
    assert not panel._down.isEnabled()
    panel._remove_selected()
    panel._move(1)
    assert asked == []


def test_the_row_switch_reports_the_layer_it_belongs_to(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask(), CropMask())
    asked: list[tuple[int, bool]] = []
    panel.enabled_requested.connect(lambda layer, on: asked.append((layer, on)))
    panel._rows[0].switch.setChecked(False)
    panel._rows[1].switch.setChecked(False)
    assert asked == [(0, False), (1, False)]


def test_a_switched_off_layer_still_shows_where_it_sits(qtbot: QtBot) -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    state = ViewerState(image=image, layers=(Layer(InvertMask(), enabled=False),))
    panel = LayerPanel()
    qtbot.addWidget(panel)
    panel.set_state(state, board_of(state))
    assert not panel._rows[0].switch.isChecked()
    assert panel._rows[0].property("off") is True
    assert _rows(panel)[0][1] == "反相"


def test_the_move_buttons_carry_the_selection_along(qtbot: QtBot) -> None:
    """Otherwise the next press of the same button would move the mask back."""
    panel = _panel(qtbot, InvertMask(), CropMask(), GrayscaleMask())
    asked: list[tuple[object, ...]] = []
    panel.move_requested.connect(lambda layer, step: asked.append(("move", layer, step)))
    panel.remove_requested.connect(lambda layer: asked.append(("remove", layer)))
    panel._select(0)
    panel._up.click()
    panel._up.click()
    panel._down.click()
    panel._remove.click()
    assert asked == [("move", 0, 1), ("move", 1, 1), ("move", 2, -1), ("remove", 1)]


def test_a_move_at_the_end_of_the_list_does_nothing(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask())
    asked: list[tuple[int, int]] = []
    panel.move_requested.connect(lambda layer, step: asked.append((layer, step)))
    panel._select(0)
    panel._up.click()
    panel._down.click()
    assert asked == []
    assert panel.selected == 0


def test_a_broken_region_mask_is_flagged_on_its_own_row(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask(), RegionMask("R >= "))
    row = panel._rows[1]
    assert row._detail.text().startswith(text.WARNING_MARK)
    assert panel._note.text().startswith("语法错误")
    assert panel._note.property("warn") is True
    assert panel._rows[0]._detail.text() == "inv"  # the healthy row keeps its own line


def test_the_add_menu_offers_only_the_parameterless_masks(qtbot: QtBot) -> None:
    """A mask with a parameter of its own is written in the filter box instead."""
    panel = _panel(qtbot)
    menu = panel._add_button.menu()
    panel._fill_menu()
    labels = [action.text() for action in menu.actions()]
    assert labels == [text.mask_info(mask).label for mask in text.mask_menu()]
    assert set(labels) == {"反相", "灰度", "裁剪"}
    added: list[Mask] = []
    panel.add_requested.connect(added.append)
    menu.actions()[0].trigger()
    assert added == [InvertMask()]
    menu.actions()[2].trigger()
    assert added == [InvertMask(), CropMask()]


def test_a_panel_without_an_image_lists_the_base_row_only(qtbot: QtBot) -> None:
    panel = LayerPanel()
    qtbot.addWidget(panel)
    panel.set_state(ViewerState(), None)
    assert list(panel._rows) == [None]
    assert panel.selected is None
    assert not panel._remove.isEnabled()
    menu = panel._add_button.menu()
    panel._fill_menu()
    assert menu.actions() == []


def test_the_panel_keeps_its_selection_on_a_layer_that_survives(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask())
    panel._select(0)
    panel.set_state(_state(ThresholdMask()), board_of(_state(ThresholdMask())))
    assert panel.selected == 0
    panel.set_state(ViewerState(), None)
    assert panel.selected is None


def test_the_image_row_stays_selected_until_a_mask_is_added(qtbot: QtBot) -> None:
    panel = _panel(qtbot, InvertMask())
    panel._select(None)
    _show(panel, InvertMask())  # a state change with the same stack
    assert panel.selected is None
    _show(panel, InvertMask(), CropMask())
    assert panel.selected == 1  # the new mask takes the selection


def test_clicking_the_blank_space_deselects(qtbot: QtBot) -> None:
    """With nothing selected the command input is empty and typing adds on top."""
    panel = _panel(qtbot, InvertMask())
    panel._select(0)
    qtbot.mouseClick(
        panel, Qt.MouseButton.LeftButton, pos=QPoint(panel.width() // 2, panel.height() - 24)
    )
    assert panel.selected is None


def test_clicking_a_row_picks_it(qtbot: QtBot) -> None:
    """The press used to travel on to the card and undo the pick the row had just made."""
    panel = _panel(qtbot, InvertMask(), ThresholdMask(128))
    panel._select(None)
    for index in (1, 0):
        row = panel._rows[index]
        qtbot.mouseClick(row, Qt.MouseButton.LeftButton, pos=QPoint(4, row.height() // 2))
        assert panel.selected == index
    card = panel._card_frame
    qtbot.mouseClick(card, Qt.MouseButton.LeftButton, pos=QPoint(3, card.height() - 3))
    assert panel.selected is None


def test_a_rows_own_parts_pick_it_too(qtbot: QtBot) -> None:
    """The switch and the labels are the row, so a click on either picks the row."""
    panel = _panel(qtbot, InvertMask(), ThresholdMask(128))
    panel._select(0)
    row = panel._rows[1]
    qtbot.mouseClick(row._detail, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert panel.selected == 1
    base = panel._rows[None]
    qtbot.mouseClick(base._name, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert panel.selected is None  # the image's own row is a selection like any other
    qtbot.mouseClick(row.switch, Qt.MouseButton.LeftButton)
    assert panel.selected == 1
    assert not row.switch.isChecked()  # the switch still does its own job


@pytest.mark.parametrize("mask", [InvertMask(), GrayscaleMask(), CropMask()])
def test_the_parameterless_masks_show_their_own_line(qtbot: QtBot, mask: Mask) -> None:
    panel = _panel(qtbot, mask)
    assert panel._note.text() == text.mask_info(mask).tip
    assert panel._note.property("warn") is False
    assert isinstance(panel._rows[0].switch, QCheckBox)

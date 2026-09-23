from pathlib import Path

import numpy as np
import pytest

from pixelsb.domain.models import (
    MAX_ZOOM,
    BitChoice,
    DisplayFormat,
    PixelCoord,
    ValueMode,
    ViewerState,
)
from pixelsb.domain.transitions import (
    clear_anchor,
    cycle_format,
    move_cursor,
    open_image,
    select_lsb,
    select_only,
    set_anchor,
    set_zoom,
    step_focus_bit,
    step_zoom,
    toggle_bit,
)
from tests.support import make_image, planes_rgb


def test_open_image_clears_navigation_and_keeps_view_settings() -> None:
    image = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("a.png"))
    other = make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb(), path=Path("b.png"))
    state = open_image(
        ViewerState(
            value_format=DisplayFormat.BINARY,
            value_mode=ValueMode.OFFSET,
            zoom=4,
        ),
        image,
    )
    state = set_anchor(state, PixelCoord(0, 0))
    state = select_only(state, "R", 3)
    opened = open_image(state, other)
    assert opened.anchor is None
    assert opened.cursor is None
    assert opened.selection is None
    assert opened.focus == BitChoice("R", 0)
    assert opened.value_format is DisplayFormat.BINARY
    assert opened.value_mode is ValueMode.OFFSET
    assert opened.zoom == 4
    assert opened.image is other


def test_stepping_the_focus_stays_inside_the_plane() -> None:
    state = select_only(
        open_image(ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())),
        "R",
        0,
    )
    assert step_focus_bit(state, -1).focus == BitChoice("R", 0)
    top = step_focus_bit(select_only(state, "R", 7), 1)
    assert top.focus == BitChoice("R", 7)
    assert top.selection == frozenset({BitChoice("R", 7)})


def test_move_cursor_starts_at_the_origin_and_clamps() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    )
    state = move_cursor(state, 10, 10)
    assert state.cursor == PixelCoord(0, 0)
    state = move_cursor(state, 10, 10)
    assert state.cursor == PixelCoord(1, 1)
    state = move_cursor(state, -5, -5)
    assert state.cursor == PixelCoord(0, 0)


def test_missing_plane_shortcut_leaves_the_state_alone() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    )
    assert select_lsb(state, "L") is state


def test_readout_layer_defaults_to_the_original_and_toggles() -> None:
    from pixelsb.domain.selection import all_bits
    from pixelsb.domain.transitions import (
        effective_readout,
        select_only_readout,
        set_readout_channel,
        toggle_readout_bit,
    )

    state = select_only(
        open_image(ViewerState(), make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())),
        "R",
        0,
    )
    assert state.readout is None
    with pytest.raises(ValueError):
        toggle_readout_bit(state, "R", 8)
    excluded = toggle_readout_bit(state, "R", 0)
    image = excluded.image
    assert image is not None
    assert excluded.readout == all_bits(image) - {BitChoice("R", 0)}
    restored = toggle_readout_bit(excluded, "R", 0)
    assert restored.readout is None
    channel = set_readout_channel(state, "G", on=False)
    assert channel.readout == all_bits(image) - {BitChoice("G", bit) for bit in range(8)}
    narrowed = select_only_readout(state, "G", 0)
    assert narrowed.readout == frozenset({BitChoice("G", 0)})
    assert len(effective_readout(narrowed)) == 1


def test_channel_and_column_group_switches() -> None:
    from pixelsb.domain.selection import all_bits
    from pixelsb.domain.transitions import (
        set_channel,
        set_column,
        set_detached,
        set_readout_channel,
        set_readout_column,
    )

    image = make_image(np.zeros((1, 1, 3), dtype=np.uint16), planes_rgb())
    state = open_image(ViewerState(), image)
    off = set_channel(state, "R", on=False)
    assert off.selection == all_bits(image) - {BitChoice("R", bit) for bit in range(8)}
    on = set_channel(off, "R", on=True)
    assert on.selection is None
    column = set_column(state, 0, on=False)
    assert column.selection == all_bits(image) - {
        BitChoice("R", 0),
        BitChoice("G", 0),
        BitChoice("B", 0),
    }
    readout_column = set_readout_column(state, 1, on=False)
    assert readout_column.readout == all_bits(image) - {
        BitChoice("R", 1),
        BitChoice("G", 1),
        BitChoice("B", 1),
    }
    readout_channel = set_readout_channel(readout_column, "R", on=False)
    assert readout_channel.readout == all_bits(image) - {
        BitChoice("R", bit) for bit in range(8)
    } - {
        BitChoice("G", 1),
        BitChoice("B", 1),
    }
    detached = set_detached(readout_channel, True)
    assert detached.detached
    assert set_detached(detached, False).detached is False


def test_clear_anchor_zoom_and_format_cycle() -> None:
    state = open_image(
        ViewerState(), make_image(np.zeros((2, 2, 3), dtype=np.uint16), planes_rgb())
    )
    state = set_anchor(state, PixelCoord(1, 1))
    assert clear_anchor(state).anchor is None
    assert cycle_format(state).value_format is DisplayFormat.BINARY
    assert cycle_format(cycle_format(state)).value_format is DisplayFormat.DECIMAL
    assert step_focus_bit(select_only(state, "R", 0), -1).focus == BitChoice("R", 0)
    added = toggle_bit(select_only(state, "R", 0), "G", 0)
    assert added.selection == frozenset({BitChoice("R", 0), BitChoice("G", 0)})
    assert step_zoom(set_zoom(state, MAX_ZOOM), 1).zoom == MAX_ZOOM
    with pytest.raises(ValueError):
        set_zoom(state, 0)

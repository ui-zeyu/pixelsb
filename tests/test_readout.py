from pathlib import Path

import numpy as np

from pixelsb.domain.labels import label_fits, widest_text, zoom_required
from pixelsb.domain.models import (
    DisplayFormat,
    PixelCoord,
    SampleOrigin,
    SamplePlane,
    ValueMode,
    ViewerState,
)
from pixelsb.domain.readout import build_readout, cursor_label
from pixelsb.domain.transitions import (
    open_image,
    select_only,
    set_anchor,
    set_cursor,
    set_format,
    set_value_mode,
)
from pixelsb.ui.text import readout_text, status_text
from tests.support import make_image, planes_rgb

_SAMPLES = np.array(
    [
        [[255, 0, 0], [0, 255, 1]],
        [[0, 0, 255], [16, 32, 64]],
    ],
    dtype=np.uint16,
)


def _state() -> ViewerState:
    return open_image(ViewerState(), make_image(_SAMPLES, planes_rgb(), path=Path("view.png")))


def test_readout_text_shows_absolute_and_signed_offset() -> None:
    state = set_cursor(set_anchor(_state(), PixelCoord(0, 0)), PixelCoord(1, 0))
    assert readout_text(state) == (
        "光标 (1, 0)\n锚点 (0, 0)\ndx +1\ndy +0\nR  00  -FF\nG  FF  +FF\nB  01  +01\n"
        "画面 原图 · 数字 原始值"
    )


def test_readout_without_an_anchor_omits_the_delta_column() -> None:
    state = set_cursor(_state(), PixelCoord(1, 0))
    assert readout_text(state) == (
        "光标 (1, 0)\n锚点 —\nR  00\nG  FF\nB  01\n画面 原图 · 数字 原始值"
    )


def test_empty_states_have_hints() -> None:
    assert readout_text(ViewerState()) == "未打开图像"
    assert readout_text(_state()) == "移动鼠标或方向键查看像素"


def test_plane_selection_keeps_original_numbers_until_the_readout_changes() -> None:
    state = set_format(set_cursor(_state(), PixelCoord(0, 0)), DisplayFormat.BINARY)
    state = select_only(state, "R", 0)
    assert cursor_label(state) == "R:11111111\nG:00000000\nB:00000000"
    assert readout_text(state).endswith("画面 R0 · 数字 原始值")
    template = widest_text(state)
    assert template == "R:11111111\nG:11111111\nB:11111111"
    assert zoom_required(template) == 36
    assert not label_fits(35, template)
    assert label_fits(36, template)
    from pixelsb.domain.transitions import select_only_readout, toggle_readout_bit

    numbers = select_only_readout(state, "R", 0)
    assert cursor_label(numbers) == "1"
    assert readout_text(numbers).endswith("画面 R0 · 数字 R0")
    excluded = toggle_readout_bit(state, "R", 0)
    assert cursor_label(excluded) == "R:11111110\nG:00000000\nB:00000000"
    summary = readout_text(excluded)
    assert "画面 R0 · 数字 R1 R2 R3 R4 R5 R6 R7" in summary
    assert summary.endswith("G6 G7 B0 B1 B2 B3 B4 B5 B6 B7")
    offset = set_value_mode(set_anchor(numbers, PixelCoord(1, 0)), ValueMode.OFFSET)
    assert build_readout(offset) is not None
    assert "dx" in readout_text(offset)


def test_status_mentions_converted_samples() -> None:
    raw = _state()
    assert "原始" in status_text(raw)
    assert "位平面来自转换后的数据" not in status_text(raw)
    converted = open_image(
        ViewerState(),
        make_image(
            np.zeros((1, 1, 4), dtype=np.uint16),
            (
                SamplePlane("R", 0, 8, SampleOrigin.CONVERTED),
                SamplePlane("G", 1, 8, SampleOrigin.CONVERTED),
                SamplePlane("B", 2, 8, SampleOrigin.CONVERTED),
                SamplePlane("A", 3, 8, SampleOrigin.CONVERTED),
            ),
            source_mode="CMYK",
            path=Path("cmyk.tif"),
        ),
    )
    assert "位平面来自转换后的数据" in status_text(converted)

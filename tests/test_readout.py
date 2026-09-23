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
    toggle_bit,
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
        "光标 (1, 0)\n锚点 (0, 0)\ndx +1\ndy +0\nR  00  -FF\nG  FF  +FF\nB  01  +01\n原图"
    )


def test_readout_without_an_anchor_omits_the_delta_column() -> None:
    state = set_cursor(_state(), PixelCoord(1, 0))
    assert readout_text(state) == "光标 (1, 0)\n锚点 —\nR  00\nG  FF\nB  01\n原图"


def test_empty_states_have_hints() -> None:
    assert readout_text(ViewerState()) == "未打开图像"
    assert readout_text(_state()) == "移动鼠标或方向键查看像素"


def test_one_bit_label_is_a_digit_and_needs_zoom_before_it_is_drawn() -> None:
    state = set_format(set_cursor(_state(), PixelCoord(0, 0)), DisplayFormat.BINARY)
    state = select_only(state, "R", 0)
    assert cursor_label(state) == "1"
    assert readout_text(state).endswith("选择 R0")
    template = widest_text(state)
    assert template == "1"
    assert zoom_required(template) == 9
    assert not label_fits(8, template)
    assert label_fits(9, template)
    both = toggle_bit(state, "G", 0)
    assert cursor_label(both) == "R1\nG0"
    offset = set_value_mode(set_anchor(both, PixelCoord(1, 0)), ValueMode.OFFSET)
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

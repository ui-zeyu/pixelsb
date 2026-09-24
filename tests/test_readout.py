from pathlib import Path

import numpy as np

from pixelsb.domain.labels import pixel_text, widest_text
from pixelsb.domain.models import (
    DisplayFormat,
    PixelCoord,
    SampleOrigin,
    SamplePlane,
    ViewerState,
)
from pixelsb.domain.transitions import (
    open_image,
    select_only,
    set_cursor,
    set_format,
)
from pixelsb.ui.text import readout_text, status_info, zoom_label
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


def test_readout_text_lists_the_channels_of_the_cursor_pixel() -> None:
    state = set_cursor(_state(), PixelCoord(1, 0))
    assert readout_text(state) == ("光标 (1, 0)\nR  00\nG  FF\nB  01\n画面 原图")


def test_empty_states_have_hints() -> None:
    assert readout_text(ViewerState()) == "未打开图像"
    assert readout_text(_state()) == "移动鼠标或方向键查看像素"


def test_numbers_follow_the_canvas_selection() -> None:
    state = set_format(set_cursor(_state(), PixelCoord(0, 0)), DisplayFormat.BINARY)
    state = select_only(state, "R", 0)
    assert pixel_text(state, PixelCoord(0, 0)) == "1"
    assert readout_text(state).endswith("画面 R0")
    assert widest_text(state) == "1"


def test_zoom_label_keeps_two_decimals_at_most() -> None:
    assert zoom_label(4.833333333333333) == "4.83×"
    assert zoom_label(23.4567) == "23.46×"
    assert zoom_label(1.125) == "1.13×"
    assert zoom_label(127.999) == "128×"
    assert zoom_label(10.5) == "10.5×"
    assert zoom_label(2.0) == "2×"


def test_status_mentions_converted_samples() -> None:
    raw = _state()
    assert "原始" in status_info(raw)
    assert "位平面来自转换后的数据" not in status_info(raw)
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
    assert "位平面来自转换后的数据" in status_info(converted)

from pathlib import Path

import numpy as np

from pixelsb.domain.labels import widest_text
from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    DisplayFormat,
    InvertMask,
    Mask,
    PixelCoord,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ViewerState,
)
from pixelsb.domain.readout import build_readout, label_zoom
from pixelsb.domain.transitions import set_cursor, set_format
from pixelsb.ui.text import readout_text, status_info, zoom_label
from tests.support import board_of, make_image, planes_rgb
from tests.support import layers as layers_of

_SAMPLES = np.array(
    [
        [[255, 0, 0], [0, 255, 1]],
        [[0, 0, 255], [16, 32, 64]],
    ],
    dtype=np.uint16,
)


def _state(*masks: Mask) -> ViewerState:
    image = make_image(_SAMPLES, planes_rgb(), path=Path("view.png"))
    return ViewerState(image=image, layers=layers_of(*masks))


def test_readout_text_lists_the_channels_of_the_cursor_pixel() -> None:
    state = set_cursor(_state(), PixelCoord(1, 0))
    assert readout_text(state, board_of(state)) == ("光标 (1, 0)\nR  00\nG  FF\nB  01\n画面 全部位")


def test_empty_states_have_hints() -> None:
    assert readout_text(ViewerState(), None) == "未打开图像"
    assert readout_text(_state(), None) == "移动鼠标或方向键查看像素"


def test_numbers_follow_the_canvas_selection() -> None:
    state = _state(BitsMask(frozenset({BitChoice("R", 0)})))
    state = set_format(set_cursor(state, PixelCoord(0, 0)), DisplayFormat.BINARY)
    board = board_of(state)
    readout = build_readout(state, board)
    assert readout is not None
    assert [(channel.name, channel.text) for channel in readout.channels] == [("R", "1")]
    assert readout.bits == (BitChoice("R", 0),)
    assert readout_text(state, board).endswith("画面 R0")
    assert widest_text(board, DisplayFormat.BINARY) == "1"
    assert label_zoom(state, board) == 9


def test_the_numbers_are_the_ones_the_stack_left_behind() -> None:
    state = set_cursor(_state(InvertMask()), PixelCoord(0, 0))
    assert readout_text(state, board_of(state)).splitlines()[1:-1] == [
        "R  00",
        "G  FF",
        "B  FF",
    ]


def test_a_pixel_a_crop_took_out_has_no_numbers() -> None:
    state = set_cursor(_state(RegionMask("left >= 1"), CropMask()), PixelCoord(0, 0))
    board = board_of(state)
    assert build_readout(state, board) is None
    assert readout_text(state, board) == "该像素不在当前视图中"


def test_a_faded_pixel_still_reads_its_numbers() -> None:
    """A region mask dims a pixel rather than removing it: it is there to read."""
    state = set_cursor(_state(RegionMask("left >= 1")), PixelCoord(0, 0))
    board = board_of(state)
    live = board.live
    assert live is not None
    assert not live[0, 0]  # dimmed: in the picture, out of the stream
    assert readout_text(state, board).splitlines()[1:-1] == ["R  FF", "G  00", "B  00"]


def test_a_cropped_raster_still_reads_the_pixel_under_the_cursor() -> None:
    state = _state(RegionMask("left >= 1"), CropMask())
    state = set_cursor(state, PixelCoord(1, 0))
    assert readout_text(state, board_of(state)).splitlines()[1:-1] == [
        "R  00",
        "G  FF",
        "B  01",
    ]


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
    converted = ViewerState(
        image=make_image(
            np.zeros((1, 1, 4), dtype=np.uint16),
            (
                SamplePlane("R", 0, 8, SampleOrigin.CONVERTED),
                SamplePlane("G", 1, 8, SampleOrigin.CONVERTED),
                SamplePlane("B", 2, 8, SampleOrigin.CONVERTED),
                SamplePlane("A", 3, 8, SampleOrigin.CONVERTED),
            ),
            source_mode="CMYK",
            path=Path("cmyk.tif"),
        )
    )
    assert "位平面来自转换后的数据" in status_info(converted)

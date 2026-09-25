from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QScrollArea
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    DisplayFormat,
    LoadedImage,
    Mask,
    PixelCoord,
    Raster,
    RegionMask,
    ViewerState,
)
from pixelsb.domain.samples import render_raster
from pixelsb.domain.transitions import set_cursor, set_format
from pixelsb.ui.canvas import CanvasMode, ImageCanvas
from pixelsb.ui.painting import cell_of
from tests.support import board_of, layers, make_image, planes_rgb


def _canvas(qtbot: QtBot) -> ImageCanvas:
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    canvas = ImageCanvas(scroll)
    scroll.setWidget(canvas)
    qtbot.addWidget(canvas)
    return canvas


def _image(width: int, height: int, name: str = "a.png") -> LoadedImage:
    samples = np.zeros((height, width, 3), dtype=np.uint16)
    samples[..., 0] = np.arange(width, dtype=np.uint16)[None, :] * 8
    return make_image(samples, planes_rgb(), path=Path(name))


def _state_for(image: LoadedImage, *masks: Mask, zoom: float = 1.0) -> ViewerState:
    return ViewerState(image=image, layers=layers(*masks), zoom=zoom)


def _board(state: ViewerState) -> Raster:
    return board_of(state)


def _labelled_state(image: LoadedImage, zoom: float = 12) -> ViewerState:
    return _state_for(image, BitsMask(frozenset({BitChoice("R", 0)})), zoom=zoom)


def test_cell_of_is_row_major() -> None:
    assert cell_of(0, 40) == (0, 0)
    assert cell_of(39, 40) == (0, 39)
    assert cell_of(40, 40) == (1, 0)
    assert cell_of(799, 40) == (19, 39)


def test_a_state_that_only_moves_the_cursor_reuses_the_pixels(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    rgb = canvas._frame.rgb
    assert rgb is not None
    canvas.set_state(set_cursor(state, PixelCoord(1, 1)), _board(state))
    assert canvas._frame.rgb is rgb
    canvas.set_state(set_format(state, DisplayFormat.BINARY), _board(state))
    assert canvas._frame.rgb is rgb
    changed = _labelled_state(_image(8, 4), zoom=4)  # another stack: render again
    canvas.set_state(changed, _board(changed))
    assert canvas._frame.rgb is not rgb
    bigger = _state_for(_image(9, 7, "big.png"))
    canvas.set_state(bigger, _board(bigger))
    assert canvas._frame.rgb.shape[:2] == (7, 9)  # the buffer follows the image


def test_moving_the_cursor_repaints_only_the_two_cells(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    state = set_cursor(state, PixelCoord(1, 1))
    canvas.set_state(state, _board(state))
    painted: list[PixelCoord | None] = []

    def record(coord: PixelCoord | None, _zoom: float) -> None:
        painted.append(coord)

    monkeypatch.setattr(canvas, "_repaint_pixel", record)
    canvas.set_state(set_cursor(state, PixelCoord(2, 1)), _board(state))
    # The cell left behind and the one arrived at; anything else is a full repaint.
    assert painted == [PixelCoord(1, 1), PixelCoord(2, 1)]


def test_the_empty_state_keeps_the_placeholder_size(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    assert canvas.size().width() == 32
    canvas.set_state(ViewerState(), None)
    assert (canvas.size().width(), canvas.size().height()) == (320, 240)
    assert canvas.displayed_size() is None


def test_painting_labels_does_not_crash(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _labelled_state(_image(40, 20))
    canvas.set_state(state, _board(state))
    canvas.grab()


def test_painting_labels_with_a_region_mask(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(40, 20)
    marked = _state_for(
        image, BitsMask(frozenset({BitChoice("R", 0)})), RegionMask("R >= 8"), zoom=12
    )
    canvas.set_state(marked, _board(marked))
    canvas.grab()


def test_empty_state_and_cursor_marker_paint(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(ViewerState(), None)
    canvas.grab()
    state = set_cursor(_state_for(_image(6, 6)), PixelCoord(1, 1))
    canvas.set_state(state, _board(state))
    canvas.grab()


def test_filtered_pixels_fade_toward_the_canvas(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    marked = _state_for(image, RegionMask("left < 4"))
    canvas.set_state(marked, _board(marked))
    rgb = canvas._frame.rgb
    assert rgb is not None
    plain = render_raster(_board(_state_for(image)))
    faded = rgb[0, 5]
    assert (faded >= plain[0, 5]).all()
    assert int(faded.min()) > 200
    assert rgb[0, 1].tolist() == plain[0, 1].tolist()


def test_cropping_crops_the_canvas_to_the_lattice(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = _state_for(image, RegionMask("grid(0, 0, 4, 2)"), CropMask(), zoom=12)
    canvas.set_state(state, _board(state))
    assert canvas._frame.raster is not None
    assert (canvas._frame.raster.width, canvas._frame.raster.height) == (2, 2)
    assert canvas.displayed_size() == (2, 2)
    rgb = canvas._frame.rgb
    assert rgb is not None
    plain = render_raster(_board(_state_for(image)))
    assert rgb.shape[:2] == (2, 2)
    assert rgb[0, 1].tolist() == plain[0, 4].tolist()
    assert rgb[1, 1].tolist() == plain[2, 4].tolist()
    canvas.grab()  # paints the cropped raster, labels included


def test_a_cropped_canvas_maps_coordinates_between_both_spaces(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = _state_for(image, RegionMask("grid(0, 0, 4, 2)"), CropMask(), zoom=6)
    canvas.set_state(state, _board(state))
    assert canvas.displayed_at(PixelCoord(4, 2)) == (1, 1)
    assert canvas.displayed_at(PixelCoord(1, 0)) is None
    # Source (4, 2) is displayed at cell (1, 1); a click there reads back the source.
    assert canvas._coord(QPoint(1 * 6 + 1, 1 * 6 + 1)) == PixelCoord(4, 2)
    for coord in (PixelCoord(4, 2), PixelCoord(1, 0)):  # marker on a shown and a hidden pixel
        moved = set_cursor(state, coord)
        canvas.set_state(moved, _board(moved))
        canvas.grab()


def test_a_crop_that_kept_nothing_shows_the_empty_note(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), RegionMask("left > 100"), CropMask())
    canvas.set_state(state, _board(state))
    assert canvas._frame.qimage is None
    assert canvas._frame.no_match
    canvas.grab()


def test_taking_the_crop_mask_out_restores_the_full_raster(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    region = RegionMask("grid(0, 0, 4, 2)")
    cropped = _state_for(image, region, CropMask())
    canvas.set_state(cropped, _board(cropped))
    whole = _state_for(image, region)
    canvas.set_state(whole, _board(whole))
    assert canvas._frame.raster is not None
    assert not canvas._frame.raster.cropped
    assert not canvas._frame.no_match
    assert canvas._frame.rgb is not None
    assert canvas._frame.rgb.shape[:2] == (4, 8)


def test_select_mode_keeps_the_region_until_enter_or_escape(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    canvas.set_mode(CanvasMode.SELECT)
    selected: list[tuple[int, int, int, int]] = []
    committed: list[tuple[int, int, int, int]] = []
    canceled: list[bool] = []
    canvas.region_selected.connect(lambda *rect: selected.append(rect))
    canvas.region_committed.connect(lambda *rect: committed.append(rect))
    canvas.region_canceled.connect(lambda: canceled.append(True))
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    assert selected == [(0, 0, 2, 1)]
    assert canvas._marquee == (0, 0, 2, 1)  # the drag settles into an active selection
    qtbot.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._marquee is None
    assert canceled == [True]
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    qtbot.keyClick(canvas, Qt.Key.Key_Return)
    assert committed == [(0, 0, 2, 1)]
    assert canvas._marquee is None


def test_select_mode_ignores_a_plain_click(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    canvas.set_mode(CanvasMode.SELECT)
    selected: list[tuple[int, int, int, int]] = []
    canceled: list[bool] = []
    canvas.region_selected.connect(lambda *rect: selected.append(rect))
    canvas.region_canceled.connect(lambda: canceled.append(True))
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert selected == []
    assert canceled == [True]
    assert canvas._marquee is None


def test_pan_mode_drag_does_not_select(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    assert canvas._mode is CanvasMode.PAN
    selected: list[tuple[int, int, int, int]] = []
    canvas.region_selected.connect(lambda *rect: selected.append(rect))
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    assert selected == []
    assert canvas._marquee is None


def test_switching_images_drops_a_stale_region(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    canvas.set_mode(CanvasMode.SELECT)
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    assert canvas._marquee is not None
    other = _state_for(_image(4, 2, "other.png"), zoom=4)
    canvas.set_state(other, _board(other))
    assert canvas._marquee is None


def test_space_temporarily_pans_in_select_mode(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    state = _state_for(_image(8, 4), zoom=4)
    canvas.set_state(state, _board(state))
    canvas.set_mode(CanvasMode.SELECT)
    selected: list[tuple[int, int, int, int]] = []
    canvas.region_selected.connect(lambda *rect: selected.append(rect))
    qtbot.keyPress(canvas, Qt.Key.Key_Space)
    assert canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    assert selected == []  # the drag panned instead of selecting
    qtbot.keyRelease(canvas, Qt.Key.Key_Space)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor

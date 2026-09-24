from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QScrollArea
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import PixelCoord, ViewerState
from pixelsb.domain.predicate import compile_filter
from pixelsb.domain.samples import render_rgb
from pixelsb.domain.transitions import (
    open_image,
    select_only,
    set_cursor,
    set_only_matched,
    set_zoom,
)
from pixelsb.ui.canvas import CanvasMode, ImageCanvas, cell_of
from tests.support import make_image, planes_rgb


def _canvas(qtbot: QtBot) -> ImageCanvas:
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    canvas = ImageCanvas(scroll)
    scroll.setWidget(canvas)
    qtbot.addWidget(canvas)
    return canvas


def _image(width: int, height: int, name: str = "a.png"):
    samples = np.zeros((height, width, 3), dtype=np.uint16)
    samples[..., 0] = np.arange(width, dtype=np.uint16)[None, :] * 8
    return make_image(samples, planes_rgb(), path=Path(name))


def _labelled_state(image, zoom: int = 12) -> ViewerState:
    state = select_only(open_image(ViewerState(), image), "R", 0)
    return set_zoom(state, zoom)


def test_cell_of_is_row_major() -> None:
    assert cell_of(0, 40) == (0, 0)
    assert cell_of(39, 40) == (0, 39)
    assert cell_of(40, 40) == (1, 0)
    assert cell_of(799, 40) == (19, 39)


def test_canvas_keeps_the_buffer_in_sync_with_the_image(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(_state_for(_image(4, 3, "small.png")))
    assert canvas._rgb is not None
    assert canvas._rgb.shape[:2] == (3, 4)
    canvas.set_state(_state_for(_image(9, 7, "big.png")))
    assert canvas._rgb is not None
    assert canvas._rgb.shape[:2] == (7, 9)


def test_painting_labels_does_not_crash(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(_labelled_state(_image(40, 20)))
    canvas.grab()


def test_painting_labels_with_a_filter(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(40, 20)
    state = _labelled_state(image)
    match = compile_filter("R >= 8", planes_rgb()).evaluate(image, state.selection)
    canvas.set_state(state, match)
    canvas.grab()


def test_painting_survives_a_lagging_buffer(qtbot: QtBot) -> None:
    """A cached color buffer smaller than the image must not index out of bounds."""
    canvas = _canvas(qtbot)
    image = _image(40, 20)
    canvas.set_state(_labelled_state(image))
    assert canvas._rgb is not None
    canvas._rgb = np.ascontiguousarray(canvas._rgb[:5, :7])
    canvas.grab()


def test_empty_state_and_cursor_marker_paint(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(ViewerState())
    canvas.grab()
    canvas.set_state(set_cursor(_state_for(_image(6, 6)), PixelCoord(1, 1)))
    canvas.grab()


def test_filtered_pixels_fade_toward_the_canvas(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = _state_for(image)
    match = compile_filter("left < 4", planes_rgb()).evaluate(image, state.selection)
    canvas.set_state(state, match)
    rgb = canvas._rgb
    assert rgb is not None
    plain = render_rgb(image, state.selection)
    faded = rgb[0, 5]
    assert (faded >= plain[0, 5]).all()
    assert int(faded.min()) > 200
    assert rgb[0, 1].tolist() == plain[0, 1].tolist()


def test_only_matched_compacts_the_canvas_to_the_lattice(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = set_zoom(set_only_matched(_state_for(image), True), 12)
    match = compile_filter("grid(0, 0, 4, 2)", planes_rgb()).evaluate(image, state.selection)
    canvas.set_state(state, match)
    assert canvas._view is not None
    assert (canvas._view.width, canvas._view.height) == (2, 2)
    assert canvas.displayed_size() == (2, 2)
    rgb = canvas._rgb
    assert rgb is not None
    plain = render_rgb(image, state.selection)
    assert rgb.shape[:2] == (2, 2)
    assert rgb[0, 1].tolist() == plain[0, 4].tolist()
    assert rgb[1, 1].tolist() == plain[2, 4].tolist()
    canvas.grab()  # paints the compacted raster with labels from region_texts_at


def test_only_matched_maps_coordinates_between_both_spaces(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = set_zoom(set_only_matched(_state_for(image), True), 6)
    match = compile_filter("grid(0, 0, 4, 2)", planes_rgb()).evaluate(image, state.selection)
    canvas.set_state(state, match)
    assert canvas.displayed_at(PixelCoord(4, 2)) == (1, 1)
    assert canvas.displayed_at(PixelCoord(1, 0)) is None
    # Source (4, 2) is displayed at cell (1, 1); a click there reads back the source.
    assert canvas._coord(QPoint(1 * 6 + 1, 1 * 6 + 1)) == PixelCoord(4, 2)
    for coord in (PixelCoord(4, 2), PixelCoord(1, 0)):  # marker on a shown and a hidden pixel
        canvas.set_state(set_cursor(state, coord), match)
        canvas.grab()


def test_only_matched_with_no_hits_shows_the_empty_note(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    state = set_only_matched(_state_for(image), True)
    match = compile_filter("left > 100", planes_rgb()).evaluate(image, state.selection)
    canvas.set_state(state, match)
    assert canvas._image is None
    assert canvas._no_match
    canvas.grab()


def test_toggling_only_matched_off_restores_the_full_raster(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    image = _image(8, 4)
    match = compile_filter("grid(0, 0, 4, 2)", planes_rgb()).evaluate(
        image, _state_for(image).selection
    )
    canvas.set_state(set_only_matched(_state_for(image), True), match)
    canvas.set_state(_state_for(image), match)
    assert canvas._view is None
    assert not canvas._no_match
    assert canvas._rgb is not None
    assert canvas._rgb.shape[:2] == (4, 8)


def test_select_mode_keeps_the_region_until_enter_or_escape(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(set_zoom(_state_for(_image(8, 4)), 4))
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
    canvas.set_state(set_zoom(_state_for(_image(8, 4)), 4))
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
    canvas.set_state(set_zoom(_state_for(_image(8, 4)), 4))
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
    canvas.set_state(set_zoom(_state_for(_image(8, 4)), 4))
    canvas.set_mode(CanvasMode.SELECT)
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(canvas, QPoint(9, 5))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(9, 5))
    assert canvas._marquee is not None
    canvas.set_state(_state_for(_image(4, 2, "other.png")))
    assert canvas._marquee is None


def test_space_temporarily_pans_in_select_mode(qtbot: QtBot) -> None:
    canvas = _canvas(qtbot)
    canvas.set_state(set_zoom(_state_for(_image(8, 4)), 4))
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


def _state_for(image) -> ViewerState:
    return open_image(ViewerState(), image)

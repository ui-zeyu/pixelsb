from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QScrollArea
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import ViewerState
from pixelsb.domain.predicate import compile_filter
from pixelsb.domain.transitions import open_image, select_only, set_zoom
from pixelsb.ui.canvas import ImageCanvas, cell_of
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


def _state_for(image) -> ViewerState:
    return open_image(ViewerState(), image)

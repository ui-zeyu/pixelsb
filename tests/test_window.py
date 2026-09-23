from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import BitChoice, DisplayFormat, PixelCoord
from pixelsb.domain.transitions import select_only, set_anchor, set_cursor, set_format
from pixelsb.ui.main_window import MainWindow


def test_open_anchor_bit_plane_and_detail_text(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(lambda state: set_anchor(state, PixelCoord(0, 0)))
    window.apply(lambda state: set_cursor(state, PixelCoord(1, 0)))
    window.apply(lambda state: select_only(state, "R", 0))
    window.apply(lambda state: set_format(state, DisplayFormat.BINARY))
    state = window.store.state
    assert state.anchor == PixelCoord(0, 0)
    assert state.cursor == PixelCoord(1, 0)
    assert state.selection == frozenset({BitChoice("R", 0)})
    assert window.canvas.displayed_size() == (2, 2)
    detail = window.inspector.detail_text()
    assert "光标 (1, 0)" in detail
    assert "锚点 (0, 0)" in detail
    assert "dx +1" in detail


def test_click_sets_the_anchor(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    window.open_path(rgb_png)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    assert window.store.state.anchor == PixelCoord(0, 0)
    zoom = int(window.store.state.zoom)
    qtbot.mouseMove(window.canvas, QPoint(zoom + 1, 1))
    assert window.store.state.cursor == PixelCoord(1, 0)


def test_failed_open_keeps_the_current_image(qtbot: QtBot, rgb_png: Path, tmp_path: Path) -> None:
    messages: list[str] = []
    window = MainWindow(reporter=messages.append)
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.open_path(tmp_path / "missing.png")
    assert window.store.state.image is not None
    assert window.store.state.image.path == rgb_png
    assert messages


def test_arrow_key_moves_the_cursor(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(0, 0)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(1, 0)

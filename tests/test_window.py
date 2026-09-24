from itertools import pairwise
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import BitChoice, DisplayFormat, PixelCoord
from pixelsb.domain.transitions import select_only, set_cursor, set_format, set_zoom
from pixelsb.ui import main_window, text, theme
from pixelsb.ui.main_window import MainWindow


def test_open_bit_plane_and_detail_text(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(lambda state: set_cursor(state, PixelCoord(1, 0)))
    window.apply(lambda state: select_only(state, "R", 0))
    window.apply(lambda state: set_format(state, DisplayFormat.BINARY))
    state = window.store.state
    assert state.cursor == PixelCoord(1, 0)
    assert state.selection == frozenset({BitChoice("R", 0)})
    assert window.canvas.displayed_size() == (2, 2)
    detail = window.inspector.detail_text()
    assert "光标 (1, 0)" in detail
    assert "画面 R0" in detail


def test_left_click_keeps_the_cursor_where_hover_left_it(
    qtbot: QtBot,
    rgb_png: Path,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    window.open_path(rgb_png)
    zoom = int(window.store.state.zoom)
    qtbot.mouseMove(window.canvas, QPoint(zoom + 1, 1))
    assert window.store.state.cursor == PixelCoord(1, 0)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
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


def test_filter_match_follows_the_selection(qtbot: QtBot, rgb_png: Path) -> None:
    from pixelsb.domain.transitions import select_only, set_filter_expr

    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(lambda state: set_filter_expr(state, "R == 255"))
    first = window._match
    assert first is not None
    assert first[0, 0] and not first[1, 0]
    window.apply(lambda state: select_only(state, "R", 7))
    second = window._match
    assert second is not None
    assert not second.any()


def test_bad_filter_shows_an_error_and_keeps_the_image(
    qtbot: QtBot,
    rgb_png: Path,
) -> None:
    from pixelsb.domain.transitions import set_filter_expr

    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.apply(lambda state: set_filter_expr(state, "A > 0"))
    assert window._match is None
    assert window._match_error is not None
    assert "未知字段" in window._match_error
    assert window.store.state.filter_expr == "A > 0"


def test_filter_box_typing_and_focus_flow(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    edit = window._filter_edit
    edit.setFocus()
    qtbot.keyClicks(edit, "B >= R")
    assert edit.text() == "B >= R"
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert window.store.state.filter_expr == "B >= R"
    assert window.focusWidget() is edit
    qtbot.keyClicks(edit, " and 0 <= top")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert window.store.state.filter_expr == "B >= R and 0 <= top"
    qtbot.keyClick(edit, Qt.Key.Key_Escape)
    assert window.store.state.filter_expr == ""
    assert window.focusWidget() is window.canvas


def test_typing_in_the_filter_box_keeps_its_keys(qtbot: QtBot, rgb_png: Path) -> None:
    from pixelsb.domain.transitions import set_zoom

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_zoom(state, 4))
    window._filter_edit.setFocus()
    qtbot.keyClicks(window._filter_edit, "+B")
    assert window._filter_edit.text() == "+B"
    assert window.store.state.zoom == 4
    window.canvas.setFocus()
    qtbot.keyClick(window.canvas, Qt.Key.Key_Plus)
    assert window.store.state.zoom == 5


def test_arrow_key_moves_the_cursor(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(0, 0)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert window.store.state.cursor == PixelCoord(1, 0)


def test_the_zoom_slider_is_geometric(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_zoom(state, 8.0))
    assert window._zoom_slider.value() == main_window._slider_position(8.0)
    window._zoom_slider.setValue(main_window._slider_position(16.0))
    assert window.store.state.zoom == 16.0
    assert window._zoom_label.text() == text.zoom_label(16.0)


def test_the_zoom_slider_spends_equal_travel_per_doubling() -> None:
    positions = [main_window._slider_position(zoom) for zoom in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert positions[0] == 0
    assert positions[-1] == main_window._SLIDER_STEPS
    gaps = [later - earlier for earlier, later in pairwise(positions)]
    assert max(gaps) - min(gaps) <= 1
    for zoom in (1, 2, 3, 5, 17, 100, 128):
        assert main_window._slider_zoom(main_window._slider_position(zoom)) == zoom


def test_the_zoom_label_fits_the_widest_value(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._zoom_label.setText(text.zoom_label(123.4567))
    assert window._zoom_label.sizeHint().width() <= window._zoom_label.width()


def test_the_filter_box_gets_a_light_clear_icon(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    button = window._filter_edit.findChild(QToolButton)
    assert button is not None
    expected = main_window._clear_icon().pixmap(16, 16).toImage()
    assert button.icon().pixmap(16, 16).toImage() == expected
    window._filter_edit.setText("B >= R")
    assert button.isVisible() is True


def test_the_clear_icon_is_a_light_cross() -> None:
    image = main_window._clear_icon().pixmap(32, 32).toImage()
    opaque = [
        (color.red() + color.green() + color.blue())
        for y in range(image.height())
        for x in range(image.width())
        if (color := image.pixelColor(x, y)).alpha() > 200
    ]
    # A muted gray cross; the style's own clear icon is a near-black disc.
    assert opaque
    assert min(opaque) > 250
    assert image.pixelColor(0, 0).alpha() == 0


def test_the_toolbar_is_a_single_row_of_shared_height(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    row = window._filter_edit.parentWidget()
    assert row is not None
    assert row.objectName() == "toolbarRow"
    for widget in (
        window._filter_count,
        window._zoom_out,
        window._zoom_slider,
        window._zoom_in,
        window._zoom_label,
        window._zoom_fit,
        window._zoom_reset,
        window._format_combo,
    ):
        assert widget.parentWidget() is row
    heights = {
        window._filter_edit.height(),
        window._zoom_fit.height(),
        window._format_combo.height(),
        window._zoom_slider.height(),
        window.inspector._extract_search.height(),
    }
    assert heights == {theme.CONTROL_HEIGHT}
    assert row.height() == theme.CONTROL_HEIGHT + 20


def test_panel_and_status_widgets_track_the_state(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    assert window._zoom_label.text() == text.zoom_label(window.store.state.zoom)
    assert "rgb.png" in window._status_info.text()
    assert window._status_view.text().startswith("光标")

    assert window.inspector._number_matrix.isVisible() is False
    window.inspector._detach.setChecked(True)
    assert window.store.state.detached is True
    assert window.inspector._number_matrix.isVisible() is True
    window.inspector._detach.setChecked(False)
    assert window.store.state.detached is False

    window._format_combo.setCurrentIndex(window._format_combo.findData(DisplayFormat.BINARY.value))
    assert window.store.state.value_format is DisplayFormat.BINARY

    window._filter_edit.setText("B >= R")
    window._commit_filter()
    assert window._filter_count.text() == "通过 3/4"
    assert window._status_view.text().startswith("光标")
    window.grab()

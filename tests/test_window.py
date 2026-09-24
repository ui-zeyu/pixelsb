from itertools import pairwise
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QPointingDevice
from PySide6.QtWidgets import QPushButton, QToolButton, QWidget
from pytestqt.qtbot import QtBot

from pixelsb.domain import geometry
from pixelsb.domain.extract import format_extract
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    DisplayFormat,
    ExtractEncoding,
    PixelCoord,
    ScanOrder,
)
from pixelsb.domain.transitions import select_only, set_cursor, set_format, set_zoom
from pixelsb.ui import painting, text, theme
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
    first = window._match.mask
    assert first is not None
    assert first[0, 0]
    assert not first[1, 0]
    window.apply(lambda state: select_only(state, "R", 7))
    second = window._match.mask
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
    assert window._match.mask is None
    assert window._match.error is not None
    assert "未知字段" in window._match.error
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


def test_only_matched_toggle_needs_a_filter_and_drives_the_canvas(
    qtbot: QtBot,
    rgb_png: Path,
) -> None:
    from pixelsb.domain.transitions import set_filter_expr, set_only_matched

    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    checkbox = window._only_matched
    assert not checkbox.isEnabled()
    window.apply(lambda state: set_filter_expr(state, "grid(0, 0, 2, 2)"))
    assert checkbox.isEnabled()
    checkbox.setChecked(True)
    assert window.store.state.only_matched
    assert window.canvas.displayed_size() == (1, 1)
    edit = window._filter_edit
    edit.clear()
    window._apply_filter_text()
    assert not window.store.state.only_matched
    assert not checkbox.isChecked()
    assert not checkbox.isEnabled()
    # The display toggle never touches the extracted byte stream.
    window.apply(lambda state: set_filter_expr(state, "grid(0, 0, 2, 2)"))
    before = window.inspector._extract_rows
    window.apply(lambda state: set_only_matched(state, on=True))
    assert window.inspector._extract_rows == before


def test_switching_tools_hands_focus_back_to_the_canvas(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window._mode_select.click()
    assert window.focusWidget() is window.canvas  # space must reach the canvas, not the button
    qtbot.keyPress(window.canvas, Qt.Key.Key_Space)
    assert window._mode_select.isChecked()  # space pans, it does not toggle the tool
    assert window.canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor
    qtbot.keyRelease(window.canvas, Qt.Key.Key_Space)
    assert window.canvas.cursor().shape() == Qt.CursorShape.CrossCursor


def test_select_mode_previews_then_appends_on_enter(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    zoom = int(window.store.state.zoom)
    assert window._mode_move.isChecked()
    assert not window._mode_select.isChecked()
    window._mode_select.click()
    assert window._mode_select.isChecked()
    assert not window._mode_move.isChecked()
    qtbot.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    qtbot.mouseMove(window.canvas, QPoint(zoom + 1, zoom + 1))
    assert "选区" in window._status_view.text()
    qtbot.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(zoom + 1, zoom + 1))
    # Still a preview: nothing is selected until Enter appends the rect.
    assert "回车追加到过滤器" in window._status_view.text()
    assert "通过 4/4" in window._status_view.text()
    assert window.store.state.filter_expr == ""
    assert window._match.mask is None
    assert window._filter_count.text() == ""
    qtbot.keyClick(window.canvas, Qt.Key.Key_Return)
    assert window._filter_edit.text() == "rect(0, 0, 2, 2)"
    assert window.store.state.filter_expr == "rect(0, 0, 2, 2)"
    assert window._status_view.text().startswith("光标")
    qtbot.keyClick(window.canvas, Qt.Key.Key_Escape)
    assert window.store.state.filter_expr == "rect(0, 0, 2, 2)"  # Esc only clears a preview


def test_committing_a_region_appends_to_the_expression(qtbot: QtBot, rgb_png: Path) -> None:
    from pixelsb.domain.transitions import set_filter_expr

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_filter_expr(state, "B >= 200"))
    assert window._match.mask is not None
    assert int(window._match.mask.sum()) == 1  # only the blue pixel at (0, 1)
    extract = window.inspector._extract_rows
    window._on_region_selected(1, 0, 1, 1)  # the right column holds no match
    assert window.store.state.filter_expr == "B >= 200"  # preview only
    assert "通过 0/4" in window._status_view.text()
    window._on_region_committed(1, 0, 1, 1)
    assert window.store.state.filter_expr == "(B >= 200) and rect(1, 0, 2, 2)"
    assert window._match.mask is not None
    assert int(window._match.mask.sum()) == 0
    assert window.inspector._extract_rows != extract


def test_escape_drops_the_preview_without_touching_the_filter(qtbot: QtBot, rgb_png: Path) -> None:
    from pixelsb.domain.transitions import set_filter_expr

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_filter_expr(state, "B >= 200"))
    window._on_region_selected(1, 0, 1, 1)
    window._on_region_canceled()
    assert window.store.state.filter_expr == "B >= 200"
    assert window._match.mask is not None
    assert int(window._match.mask.sum()) == 1
    assert window._status_view.text().startswith("光标")


def test_the_extract_tabs_and_the_state_agree(qtbot: QtBot, extract_png: Path) -> None:
    from pixelsb.domain.transitions import set_extract_encoding

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(extract_png)
    view = window.inspector._extract_view
    rows = format_extract("中".encode("utf-16-le"))
    view.set_rows(rows, ExtractEncoding.UTF16_LE)
    assert view.text_pane.toPlainText() == "中"
    view.set_rows(rows, ExtractEncoding.ASCII)
    assert view.text_pane.toPlainText() == "-N"  # the same bytes, read as ASCII
    assert view.text_pane._header.label() == "ASCII"
    # The header asks for an encoding; the state answers and the header follows.
    view.choose_encoding(ExtractEncoding.UTF16_BE)
    assert window.store.state.extract_encoding is ExtractEncoding.UTF16_BE
    window.apply(lambda state: set_extract_encoding(state, ExtractEncoding.UTF8))
    assert view.text_pane._header.label() == "UTF-8"


def test_the_plane_steppers_walk_the_display_order(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.inspector._plane_next.click()
    assert window.store.state.selection == frozenset({BitChoice("R", 7)})  # the ladder head
    window.inspector._plane_next.click()
    assert window.store.state.selection == frozenset({BitChoice("R", 6)})
    window.inspector._plane_prev.click()
    window.inspector._plane_prev.click()
    assert window.store.state.selection == frozenset({BitChoice("B", 0)})  # backwards, wrapping


def test_the_channel_steppers_walk_whole_channels(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_path(rgb_png)
    window.inspector._channel_next.click()
    for name in ("R", "G", "B"):
        assert window.store.state.selection == frozenset({BitChoice(name, bit) for bit in range(8)})
        window.inspector._channel_next.click()
    assert window.store.state.selection == frozenset({BitChoice("R", bit) for bit in range(8)})


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
    assert window._zoom_slider.value() == geometry.slider_position(8.0)
    window._zoom_slider.setValue(geometry.slider_position(16.0))
    assert window.store.state.zoom == 16.0
    assert window._zoom_label.text() == text.zoom_label(16.0)


def test_the_zoom_slider_spends_equal_travel_per_doubling() -> None:
    positions = [geometry.slider_position(zoom) for zoom in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert positions[0] == 0
    assert positions[-1] == geometry.SLIDER_STEPS
    gaps = [later - earlier for earlier, later in pairwise(positions)]
    assert max(gaps) - min(gaps) <= 1
    for zoom in (1, 2, 3, 5, 17, 100, 128):
        assert geometry.slider_zoom(geometry.slider_position(zoom)) == zoom


def test_the_zoom_label_fits_the_widest_value(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._zoom_label.setText(text.zoom_label(123.4567))
    assert window._zoom_label.sizeHint().width() <= window._zoom_label.width()


def test_a_trackpad_pinch_scales_the_zoom_in_place(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_zoom(state, 4.0))
    window._zoom_scale(1.25, 100, 50)
    assert window.store.state.zoom == pytest.approx(5.0)
    window._zoom_scale(0.5, 100, 50)
    assert window.store.state.zoom == pytest.approx(2.5)


def test_a_native_pinch_gesture_requests_a_scale(qtbot: QtBot, rgb_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    window.open_path(rgb_png)
    window.apply(lambda state: set_zoom(state, 4.0))
    before = window.store.state.zoom
    event = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture,
        QPointingDevice.primaryPointingDevice(),
        2,
        QPointF(30, 20),
        QPointF(30, 20),
        QPointF(31, 21),
        0.04,
        QPointF(0, 0),
    )
    # Real delivery invokes the widget's event() with the native gesture.
    assert window.canvas.event(event) is True
    assert window.store.state.zoom == pytest.approx(before * 1.04)
    # The same gesture over the letterbox reaches the canvas through the filter.
    assert window.canvas.eventFilter(window._scroll.viewport(), event) is True
    assert window.store.state.zoom == pytest.approx(before * 1.04 * 1.04)


def test_zooming_keeps_the_pointer_pixel_stationary(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(700, 400)
    window.show()
    window.open_path(extract_png)
    window.apply(lambda state: set_zoom(state, 64.0))
    horizontal = window._scroll.horizontalScrollBar()
    vertical = window._scroll.verticalScrollBar()
    horizontal.setValue(120)
    vertical.setValue(60)
    pointer_x, pointer_y = 200, 100
    before = (
        (horizontal.value() + pointer_x) / 64.0,
        (vertical.value() + pointer_y) / 64.0,
    )
    window._zoom_scale(2.0, pointer_x, pointer_y)
    after = (
        (horizontal.value() + pointer_x) / 128.0,
        (vertical.value() + pointer_y) / 128.0,
    )
    assert after[0] == pytest.approx(before[0], abs=0.5)
    assert after[1] == pytest.approx(before[1], abs=0.5)


def test_the_filter_box_gets_a_light_clear_icon(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    button = window._filter_edit.findChild(QToolButton)
    assert button is not None
    expected = painting.clear_icon().pixmap(16, 16).toImage()
    assert button.icon().pixmap(16, 16).toImage() == expected
    window._filter_edit.setText("B >= R")
    assert button.isVisible() is True


def test_the_clear_icon_is_a_light_cross() -> None:
    image = painting.clear_icon().pixmap(32, 32).toImage()
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

    window._format_combo.setCurrentIndex(window._format_combo.findData(DisplayFormat.BINARY.value))
    assert window.store.state.value_format is DisplayFormat.BINARY

    window._filter_edit.setText("B >= R")
    window._commit_filter()
    assert window._filter_count.text() == "通过 3/4"
    assert window._status_view.text().startswith("光标")
    window.grab()


def test_the_extract_order_controls_drive_the_dump(qtbot: QtBot, extract_png: Path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_path(extract_png)
    inspector = window.inspector
    pane = inspector._extract_view.hex_pane
    assert pane.toPlainText().startswith("41 42 43")

    inspector._channel_combo.setCurrentIndex(inspector._channel_combo.count() - 1)
    assert window.store.state.extract_order.planes == ("B", "G", "R")
    assert pane.toPlainText().startswith("43 42 41")

    _order_button(inspector._scan, "YZ").click()
    assert window.store.state.extract_order.scan is ScanOrder.YZ
    _order_button(inspector._bit_order, "LSB").click()
    assert window.store.state.extract_order.bit_order is BitOrder.LSB
    # Low first hands back each channel's stored byte, in the order just picked.
    assert pane.toPlainText().startswith("c2 42 82")

    # The three settings survive opening another image.
    window.open_path(extract_png)
    assert window.store.state.extract_order.scan is ScanOrder.YZ


def _order_button(pair: QWidget, label: str) -> QPushButton:
    return next(button for button in pair.findChildren(QPushButton) if button.text() == label)

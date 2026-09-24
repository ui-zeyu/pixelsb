"""The theme is light: the token table, the palette, and the indicators it applies."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox

from pixelsb.ui import theme


def _rgb(color: QColor) -> tuple[int, int, int]:
    return (color.red(), color.green(), color.blue())


_ACCENT = _rgb(QColor(theme.ACCENT))
_WHITE = (255, 255, 255)


def _relative_luminance(color: QColor) -> float:
    return (0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()) / 255


def _rule_block(selector: str) -> str:
    return theme.STYLESHEET.split(selector, 1)[1].split("}", 1)[0]


def test_tokens_are_light_with_a_blue_accent() -> None:
    for token in (theme.SURFACE, theme.WINDOW, theme.CANVAS, theme.FIELD, theme.HOVER):
        assert _relative_luminance(QColor(token)) > 0.85, token
    assert _relative_luminance(QColor(theme.TEXT)) < 0.2
    accent = QColor(theme.ACCENT)
    assert accent.blue() > accent.red() and accent.blue() > accent.green()


def test_apply_theme_sets_a_light_palette(qapp: QApplication) -> None:
    theme.apply_theme(qapp)
    palette = qapp.palette()
    assert _relative_luminance(palette.color(QPalette.ColorRole.Window)) > 0.85
    assert palette.color(QPalette.ColorRole.WindowText) == QColor(theme.TEXT)
    assert palette.color(QPalette.ColorRole.Highlight) == QColor(theme.ACCENT)
    assert theme.ACCENT in qapp.styleSheet()


def test_stylesheet_drops_the_dark_values() -> None:
    for stale in ("#121212", "#1e1e1e", "#2c2c2c", "#3f3f3f"):
        assert stale not in theme.STYLESHEET
    assert theme.ACCENT in _rule_block("QLineEdit:focus")
    assert theme.HAIRLINE in _rule_block("QFrame#hairline")


def test_the_stylesheet_leaves_the_indicator_to_the_style() -> None:
    assert "QCheckBox::indicator" not in theme.STYLESHEET


def test_ghost_buttons_are_frameless_and_tinted_on_hover() -> None:
    assert "border: none" in _rule_block("QPushButton#ghost")
    assert theme.HOVER in _rule_block("QPushButton#ghost:hover")
    assert theme.PRESSED in _rule_block("QPushButton#ghost:pressed")


def _indicator_pixels(state: Qt.CheckState) -> list[tuple[int, int, int]]:
    box = QCheckBox()
    box.setCheckState(state)
    box.setFixedSize(20, 20)
    image = box.grab().toImage()
    return [
        _rgb(image.pixelColor(x, y)) for y in range(image.height()) for x in range(image.width())
    ]


def test_a_checked_box_is_accent_with_a_light_tick(qapp: QApplication) -> None:
    theme.apply_theme(qapp)
    pixels = _indicator_pixels(Qt.CheckState.Checked)
    assert pixels.count(_ACCENT) > 100
    # Only the tick is pure white; the widget's own background is not.
    assert pixels.count(_WHITE) > 5


def test_a_partial_box_keeps_its_middle_light(qapp: QApplication) -> None:
    theme.apply_theme(qapp)
    pixels = _indicator_pixels(Qt.CheckState.PartiallyChecked)
    assert 0 < pixels.count(_ACCENT) < 100
    assert pixels.count(_WHITE) > 50


def test_a_clear_box_draws_no_accent(qapp: QApplication) -> None:
    theme.apply_theme(qapp)
    pixels = _indicator_pixels(Qt.CheckState.Unchecked)
    assert _ACCENT not in pixels
    assert pixels.count(_WHITE) > 50

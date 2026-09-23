"""The theme is light: the token table and the palette it applies."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from pixelsb.ui import theme


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
    assert theme.ACCENT in _rule_block("QCheckBox::indicator:checked")
    assert theme.HAIRLINE in _rule_block("QFrame#hairline")

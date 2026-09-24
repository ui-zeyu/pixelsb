"""Light theme: the color and size tokens, the stylesheet built from them, and
``CheckStyle``, which paints checkbox indicators by hand."""

from typing import override

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle, QStyleOption, QWidget

SURFACE = "#ffffff"
WINDOW = "#f5f6f8"
CANVAS = "#ebedf0"
FIELD = "#fbfbfc"
INPUT = "#f2f4f7"
HOVER = "#f2f4f7"
PRESSED = "#e8ecf1"
HAIRLINE = "#e3e6ea"
HAIRLINE_STRONG = "#d3d8df"
TEXT = "#1d1f23"
TEXT_MUTED = "#6b7280"
TEXT_DISABLED = "#a8adb5"
ACCENT = "#2f6feb"
DANGER = "#d9534f"

# One height for boxes and inputs, so a row of controls reads as one band.
CONTROL_HEIGHT = 28

INDICATOR_SIZE = 15
_INDICATOR_RADIUS = 3.5
_TICK_WIDTH = 1.8
_DASH_HEIGHT = 2.0

STYLESHEET = f"""
QMainWindow, QDialog, QSplitter {{
    background: {WINDOW};
}}

QMenuBar {{
    background: {SURFACE};
    color: {TEXT};
    border-bottom: 1px solid {HAIRLINE};
}}
QMenuBar::item {{ padding: 4px 10px; border-radius: 5px; background: transparent; }}
QMenuBar::item:selected {{ background: {HOVER}; }}
QMenu {{
    background: {SURFACE};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{ padding: 5px 20px 5px 10px; border-radius: 5px; }}
QMenu::item:selected {{ background: {HOVER}; color: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: {HAIRLINE}; margin: 5px 8px; }}

QToolBar {{ background: {SURFACE}; border: none; padding: 0; spacing: 0; }}
QFrame#toolbarRow {{ background: {SURFACE}; border-bottom: 1px solid {HAIRLINE}; }}
QFrame#card {{ background: {FIELD}; border: none; border-radius: 8px; }}
QFrame#hairline {{ background: {HAIRLINE}; border: none; }}

QLabel {{ background: transparent; color: {TEXT}; }}
QLabel#muted, QLabel#note, QLabel#zoomLabel, QLabel#sectionTitle {{ color: {TEXT_MUTED}; }}
QLabel#sectionTitle {{ font-size: 11px; font-weight: 600; }}
QLabel#note {{ font-size: 11px; }}
QLabel#zoomLabel {{ font-size: 12px; }}
QLabel#error {{ color: {DANGER}; }}

QPushButton, QComboBox {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
    padding: 3px 10px;
    min-height: 18px;
}}
QPushButton:hover, QComboBox:hover {{ background: {HOVER}; border-color: {HAIRLINE_STRONG}; }}
QPushButton:pressed {{ background: {PRESSED}; }}
QPushButton:disabled, QComboBox:disabled, QLabel:disabled {{ color: {TEXT_DISABLED}; }}
QPushButton#segmentLeft {{ border-top-right-radius: 0; border-bottom-right-radius: 0; }}
QPushButton#segmentRight {{
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
    margin-left: -1px;
}}
QPushButton#segmentLeft:checked, QPushButton#segmentRight:checked {{
    background: {HOVER};
    color: {ACCENT};
}}
QPushButton#linkButton {{ border: none; background: transparent; color: {ACCENT}; padding: 0 4px; }}
QPushButton#linkButton:hover {{ background: {HOVER}; }}
QPushButton#ghost {{ border: none; background: transparent; color: {TEXT}; padding: 2px 8px; }}
QPushButton#ghost:hover {{ background: {HOVER}; }}
QPushButton#ghost:pressed {{ background: {PRESSED}; }}
QPushButton#stepper {{
    border: none;
    background: transparent;
    color: {TEXT};
    font-size: 16px;
    padding: 0;
}}
QPushButton#stepper:hover {{ background: {HOVER}; color: {ACCENT}; border-radius: 6px; }}
QPushButton#stepper:pressed {{ background: {PRESSED}; color: {ACCENT}; }}

QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: {HOVER};
    selection-color: {ACCENT};
}}

QLineEdit {{
    background: {INPUT};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 9px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {SURFACE};
}}
QLineEdit:hover {{ background: {PRESSED}; }}
QLineEdit:focus {{ border-color: {ACCENT}; background: {SURFACE}; }}
QLineEdit > QToolButton {{ background: transparent; border: none; }}
QLineEdit > QToolButton:hover {{ background: {PRESSED}; border-radius: 4px; }}

QPlainTextEdit {{
    background: {FIELD};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {SURFACE};
}}
QPlainTextEdit#textPane {{ color: {TEXT_MUTED}; }}

QCheckBox {{ spacing: 6px; background: transparent; color: {TEXT}; }}
QCheckBox#headerBox {{ color: {TEXT_MUTED}; font-size: 12px; spacing: 4px; }}

QSlider::groove:horizontal {{ height: 4px; background: {HAIRLINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px;
    margin: -6px 0;
    background: {SURFACE};
    border: 1px solid {HAIRLINE_STRONG};
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ border-color: {ACCENT}; }}

QScrollArea#canvasArea, QScrollArea#canvasArea > QWidget > QWidget {{
    background: {CANVAS};
    border: none;
}}
QScrollArea#inspectorArea, QScrollArea#inspectorArea > QWidget > QWidget {{
    background: {SURFACE};
    border: none;
    border-left: 1px solid {HAIRLINE};
}}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 9px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #cfd5dd; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:horizontal {{ background: #cfd5dd; border-radius: 4px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: #b5bcc6; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {SURFACE}; border-top: 1px solid {HAIRLINE}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {TEXT_MUTED}; font-size: 12px; padding: 0 6px; }}

QToolTip {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
    padding: 4px 6px;
}}
"""


class CheckStyle(QProxyStyle):
    """Fusion, with checkbox indicators drawn by hand.

    The stylesheet leaves ``QCheckBox::indicator`` alone, so Qt asks this style
    for the box: a rounded outline when clear, the accent with a white tick when
    checked, and an accent outline with a dash when only part of a row or column
    is on.
    """

    @override
    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        if element is not QStyle.PrimitiveElement.PE_IndicatorCheckBox:
            super().drawPrimitive(element, option, painter, widget)
            return
        _paint_indicator(painter, option)

    @override
    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        if metric in (
            QStyle.PixelMetric.PM_IndicatorWidth,
            QStyle.PixelMetric.PM_IndicatorHeight,
        ):
            return INDICATOR_SIZE
        return super().pixelMetric(metric, option, widget)


def _paint_indicator(painter: QPainter, option: QStyleOption) -> None:
    state = option.state
    enabled = bool(state & QStyle.StateFlag.State_Enabled)
    box = _centered_box(option.rect)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if state & QStyle.StateFlag.State_On:
        _paint_checked(painter, box, enabled=enabled)
    elif state & QStyle.StateFlag.State_NoChange:
        _paint_partial(painter, box, enabled=enabled)
    else:
        _paint_clear(
            painter,
            box,
            enabled=enabled,
            hovered=bool(state & QStyle.StateFlag.State_MouseOver),
        )
    painter.restore()


def _centered_box(rect: QRect) -> QRectF:
    side = min(float(INDICATOR_SIZE), float(rect.width()), float(rect.height()))
    center = QRectF(rect).center()
    return QRectF(center.x() - side / 2, center.y() - side / 2, side, side)


def _paint_clear(painter: QPainter, box: QRectF, *, enabled: bool, hovered: bool) -> None:
    border = ACCENT if hovered else HAIRLINE_STRONG
    painter.setPen(QPen(QColor(border), 1.0))
    painter.setBrush(QColor(SURFACE if enabled else HOVER))
    painter.drawRoundedRect(
        box.adjusted(0.5, 0.5, -0.5, -0.5), _INDICATOR_RADIUS, _INDICATOR_RADIUS
    )


def _paint_checked(painter: QPainter, box: QRectF, *, enabled: bool) -> None:
    fill = QColor(ACCENT if enabled else TEXT_DISABLED)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    painter.drawRoundedRect(box, _INDICATOR_RADIUS, _INDICATOR_RADIUS)
    pen = QPen(QColor(SURFACE))
    pen.setWidthF(_TICK_WIDTH)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolyline(
        QPolygonF([_inside(box, 0.28, 0.53), _inside(box, 0.44, 0.68), _inside(box, 0.73, 0.34)])
    )


def _paint_partial(painter: QPainter, box: QRectF, *, enabled: bool) -> None:
    color = QColor(ACCENT if enabled else TEXT_DISABLED)
    painter.setPen(QPen(color, 1.2))
    painter.setBrush(QColor(SURFACE))
    painter.drawRoundedRect(
        box.adjusted(0.7, 0.7, -0.7, -0.7), _INDICATOR_RADIUS, _INDICATOR_RADIUS
    )
    dash = QRectF(0.0, 0.0, box.width() * 0.5, _DASH_HEIGHT)
    dash.moveCenter(box.center())
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawRoundedRect(dash, _DASH_HEIGHT / 2, _DASH_HEIGHT / 2)


def _inside(box: QRectF, x: float, y: float) -> QPointF:
    """A point in ``box``, given as fractions of its width and height."""
    return QPointF(box.x() + box.width() * x, box.y() + box.height() * y)


def apply_theme(application: QApplication) -> None:
    application.setStyle(CheckStyle("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WINDOW))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(HOVER))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Mid, QColor(HAIRLINE))
    application.setPalette(palette)
    application.setStyleSheet(STYLESHEET)

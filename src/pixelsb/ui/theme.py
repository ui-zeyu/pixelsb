"""Light theme. Every color used by the interface is a token in this table."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

SURFACE = "#ffffff"
WINDOW = "#f5f6f8"
CANVAS = "#ebedf0"
FIELD = "#fbfbfc"
HOVER = "#f2f4f7"
PRESSED = "#e8ecf1"
HAIRLINE = "#e3e6ea"
HAIRLINE_STRONG = "#d3d8df"
TEXT = "#1d1f23"
TEXT_MUTED = "#6b7280"
TEXT_DISABLED = "#a8adb5"
ACCENT = "#2f6feb"
ACCENT_SOFT = "rgba(47, 111, 235, 0.45)"
DANGER = "#d9534f"

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
QFrame#filterRow {{ background: {SURFACE}; border-bottom: 1px solid {HAIRLINE}; }}
QFrame#actionRow {{ background: {SURFACE}; border-bottom: 1px solid {HAIRLINE}; }}
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
QPushButton#linkButton {{ border: none; background: transparent; color: {ACCENT}; padding: 0 4px; }}
QPushButton#linkButton:hover {{ background: {HOVER}; }}

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
    background: {FIELD};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
    padding: 4px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {SURFACE};
}}
QLineEdit:focus {{ border-color: {ACCENT}; background: {SURFACE}; }}

QPlainTextEdit {{
    background: {FIELD};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {SURFACE};
}}

QCheckBox {{ spacing: 6px; background: transparent; color: {TEXT}; }}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {HAIRLINE_STRONG};
    border-radius: 4px;
    background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:indeterminate {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QCheckBox[groupOn="true"] {{ color: {ACCENT}; }}
QCheckBox#headerBox {{ color: {TEXT_MUTED}; font-size: 12px; spacing: 4px; }}
QCheckBox#headerBox[groupOn="true"] {{ color: {ACCENT}; }}

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


def apply_theme(application: QApplication) -> None:
    application.setStyle("Fusion")
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

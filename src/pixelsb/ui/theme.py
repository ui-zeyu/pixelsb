"""Dark theme so bit planes stay readable during long sessions."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

STYLESHEET = """
QMainWindow, QMenuBar, QMenu, QStatusBar, QToolBar, QDialog {
    background: #1e1e1e;
    color: #e8e8e8;
}
QToolBar {
    spacing: 8px;
    padding: 4px;
    border: none;
}
QComboBox, QPushButton {
    background: #2c2c2c;
    color: #e8e8e8;
    border: 1px solid #3f3f3f;
    border-radius: 4px;
    padding: 3px 8px;
}
QCheckBox {
    color: #e8e8e8;
    spacing: 6px;
}
QComboBox:hover, QPushButton:hover {
    border-color: #6a6a6a;
}
QPushButton:disabled, QComboBox:disabled, QCheckBox:disabled {
    color: #777777;
}
QScrollArea, QSplitter {
    background: #121212;
    border: none;
}
QSplitter::handle {
    background: #2a2a2a;
}
QStatusBar {
    background: #181818;
}
QMenu::item:selected {
    background: #3a6ea5;
}
QToolTip {
    background: #2c2c2c;
    color: #e8e8e8;
    border: 1px solid #3f3f3f;
}
"""


def apply_theme(application: QApplication) -> None:
    application.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8e8e8"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2a2a2a"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e8e8e8"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2c2c2c"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e8e8e8"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#3a6ea5"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2c2c2c"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e8e8e8"))
    application.setPalette(palette)
    application.setStyleSheet(STYLESHEET)

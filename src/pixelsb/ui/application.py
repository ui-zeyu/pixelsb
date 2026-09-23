"""Application entry point."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pixelsb.ui.main_window import MainWindow
from pixelsb.ui.theme import apply_theme

_DESCRIPTION = "Pixel-level image viewer for channel values, bit planes, and bit combinations."


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pixelsb",
        description=_DESCRIPTION,
    )
    parser.add_argument("path", nargs="?", type=Path, help="image to open")
    args = parser.parse_args(argv)
    application = QApplication([sys.argv[0]])
    application.setApplicationName("pixelsb")
    apply_theme(application)
    window = MainWindow()
    window.showMaximized()
    if args.path is not None:
        window.open_path(args.path)
    return application.exec()

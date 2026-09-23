"""Pixel-level image viewer for channel values, bit planes, and bit combinations."""

from importlib.metadata import version

__version__ = version("pixelsb")


def main() -> None:
    from pixelsb.ui.application import run

    raise SystemExit(run())

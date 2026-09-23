import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from pixelsb.ui import theme

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def _themed(qapp: QApplication) -> None:
    """Every widget test runs under the real stylesheet and style."""
    theme.apply_theme(qapp)


@pytest.fixture
def rgb_png(tmp_path: Path) -> Path:
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 1))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (16, 32, 64))
    path = tmp_path / "rgb.png"
    image.save(path)
    return path


@pytest.fixture
def extract_png(tmp_path: Path) -> Path:
    """8x4 RGB: 96 bytes, six dump rows of 16.

    Bits pack with the first extracted bit in the MSB, so these sample values
    read back as the ASCII bytes ``41 42 43`` -- "ABC" -- over and over.
    """
    image = Image.new("RGB", (8, 4), (0x82, 0x42, 0xC2))
    path = tmp_path / "extract.png"
    image.save(path)
    return path

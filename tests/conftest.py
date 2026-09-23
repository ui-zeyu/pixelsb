import os
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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

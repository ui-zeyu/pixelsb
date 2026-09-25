"""Writing pixels back out as image files."""

from pathlib import Path

from PIL import Image

from pixelsb.domain.models import RgbArray


def save_rgb(rgb: RgbArray, path: Path) -> None:
    """Save an HxWx3 byte array; the file suffix picks the format."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 array")
    Image.fromarray(rgb).save(path)

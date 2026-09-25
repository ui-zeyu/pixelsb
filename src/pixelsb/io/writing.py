"""Writing pixels back out as image files."""

from pathlib import Path

from PIL import Image

from pixelsb.domain.models import RgbaArray, RgbArray


def save_image(pixels: RgbArray | RgbaArray, path: Path) -> None:
    """Save an HxWx3 or HxWx4 byte array; the file suffix picks the format."""
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("expected an HxWx3 or HxWx4 array")
    Image.fromarray(pixels).save(path)

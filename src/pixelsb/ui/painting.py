"""Conversions from sample arrays to widgets."""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage


def qimage_from_rgb(array: NDArray[np.uint8]) -> QImage:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HxWx3 array")
    height, width, _channels = array.shape
    buffer = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    image = QImage(buffer, width, height, width * 3, QImage.Format.Format_RGB888)
    copied = image.copy()
    if copied.isNull():
        raise ValueError("could not copy the image buffer")
    return copied

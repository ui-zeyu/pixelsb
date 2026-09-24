"""Conversions from sample arrays to widgets, and the icons drawn by hand."""

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLineEdit, QToolButton

from pixelsb.ui import theme


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


def lighten_clear_button(edit: QLineEdit) -> None:
    """Swap the style's dark disc clear icon for a light cross."""
    button = edit.findChild(QToolButton)
    if button is None:
        return
    button.setIcon(clear_icon())
    button.setIconSize(QSize(16, 16))


def clear_icon() -> QIcon:
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(theme.TEXT_MUTED))
    pen.setWidthF(3.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    inset = 10.0
    far = size - inset
    painter.drawLine(QPointF(inset, inset), QPointF(far, far))
    painter.drawLine(QPointF(far, inset), QPointF(inset, far))
    painter.end()
    return QIcon(pixmap)

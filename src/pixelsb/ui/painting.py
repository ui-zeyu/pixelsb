"""Numpy arrays drawn into widgets, and the shapes the canvas paints with."""

import math
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLineEdit, QToolButton

from pixelsb.domain.labels import (
    LABEL_PAD,
    MIN_FONT,
    font_pixel_size,
    label_fits,
)
from pixelsb.domain.match_view import MatchView
from pixelsb.domain.models import PixelCoord, RgbArray
from pixelsb.ui import text, theme

# Canvas ink: the cursor marker, its halo, the pixel grid, and the region fill.
_CURSOR = QColor("#f59f00")
_CURSOR_HALO = QColor(255, 255, 255, 200)
_GRID = QColor(17, 20, 24, 26)
_DIM_KEEP = 8
_DARK_TEXT = QColor("#111111")
_LIGHT_TEXT = QColor("#f5f5f5")
_LUMA_WEIGHTS = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_THRESHOLD = 1_500_000


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


@lru_cache(maxsize=64)
def label_font(template: str, zoom: float) -> QFont | None:
    if not label_fits(zoom, template):
        return None
    lines = template.split("\n")
    longest = max(lines, key=len)
    font = QFont()
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFamilies(["Menlo", "Consolas"])
    size = font_pixel_size(zoom, template)
    while size >= 1:
        font.setPixelSize(size)
        metrics = QFontMetrics(font)
        if (
            metrics.horizontalAdvance(longest) <= zoom - LABEL_PAD
            and metrics.lineSpacing() * len(lines) <= zoom - LABEL_PAD
        ):
            return font if size >= MIN_FONT else None
        size -= 1
    return None


def cell_of(index: int, columns: int) -> tuple[int, int]:
    """Row-major ``(row, column)`` of the index-th label in a ``columns``-wide block."""
    return divmod(index, columns)


def visible_rects(
    widget_rect: QRectF,
    zoom: float,
    width: int,
    height: int,
) -> tuple[QRectF, QRectF]:
    first_x = max(math.floor(widget_rect.left() / zoom), 0)
    first_y = max(math.floor(widget_rect.top() / zoom), 0)
    last_x = min(math.floor(widget_rect.right() / zoom), width - 1)
    last_y = min(math.floor(widget_rect.bottom() / zoom), height - 1)
    if last_x < first_x or last_y < first_y:
        return QRectF(), QRectF()
    source = QRectF(first_x, first_y, last_x - first_x + 1, last_y - first_y + 1)
    dest = QRectF(first_x * zoom, first_y * zoom, source.width() * zoom, source.height() * zoom)
    return source, dest


def draw_grid(painter: QPainter, source: QRectF, zoom: float) -> None:
    pen = QPen(_GRID)
    pen.setCosmetic(True)
    painter.setPen(pen)
    left = source.left()
    top = source.top()
    right = source.right() + zoom
    bottom = source.bottom() + zoom
    for column in range(int(source.left()), int(source.right()) + 2):
        x = column * zoom
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))
    for row in range(int(source.top()), int(source.bottom()) + 2):
        y = row * zoom
        painter.drawLine(QPointF(left, y), QPointF(right, y))


def bright_map(rgb: RgbArray) -> NDArray[np.bool_]:
    """Per-pixel "light enough for dark text", computed once per rendered image."""
    weighted = rgb.astype(np.uint32) * _LUMA_WEIGHTS
    return weighted.sum(axis=-1) > _LUMA_THRESHOLD


def fade_out(rgb: RgbArray, match: NDArray[np.bool_]) -> None:
    """Push the pixels that fail the filter toward the canvas color, in place."""
    rgb[~match] = 255 - (255 - rgb[~match]) // _DIM_KEEP


def draw_empty_state(
    painter: QPainter,
    rect: QRect,
    hint: str = text.CANVAS_HINT,
    shortcut: str = text.CANVAS_SHORTCUT,
) -> None:
    """Centered lines: what to do, and the shortcut that does it."""
    font = QFont(painter.font())
    font.setPixelSize(15)
    painter.setFont(font)
    painter.setPen(QColor(theme.TEXT))
    painter.drawText(_shifted(rect, -18), Qt.AlignmentFlag.AlignCenter, hint)
    if not shortcut:
        return
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(QColor(theme.TEXT_MUTED))
    painter.drawText(_shifted(rect, 18), Qt.AlignmentFlag.AlignCenter, shortcut)


def _shifted(rect: QRect, offset: int) -> QRect:
    return QRect(rect.x(), rect.y() + offset, rect.width(), rect.height())


def draw_marker(
    painter: QPainter,
    view: MatchView | None,
    coord: PixelCoord | None,
    zoom: float,
) -> None:
    if coord is None:
        return
    dx, dy = coord.x, coord.y
    if view is not None:
        cell = view.display_of(coord)
        if cell is None:
            return
        dx, dy = cell
    rect = QRectF(dx * zoom, dy * zoom, zoom, zoom)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    halo = QPen(_CURSOR_HALO)
    halo.setCosmetic(True)
    halo.setWidth(3)
    painter.setPen(halo)
    painter.drawRect(rect.adjusted(-1, -1, 0, 0))
    pen = QPen(_CURSOR)
    pen.setCosmetic(True)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.drawRect(rect.adjusted(0, 0, -1, -1))


def draw_marquee(
    painter: QPainter,
    rect: tuple[int, int, int, int] | None,
    zoom: float,
) -> None:
    """The region being dragged: translucent amber fill with a dashed outline."""
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    area = QRectF(x0 * zoom, y0 * zoom, (x1 - x0 + 1) * zoom, (y1 - y0 + 1) * zoom)
    fill = QColor(_CURSOR)
    fill.setAlpha(28)
    painter.fillRect(area, fill)
    pen = QPen(_CURSOR)
    pen.setCosmetic(True)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(area)


def label_color(bright: bool) -> QColor:
    """Text ink for one pixel: dark over a light pixel, light over a dark one."""
    return _DARK_TEXT if bright else _LIGHT_TEXT

"""Zoomable image canvas. One image pixel maps to ``zoom`` logical pixels."""

import math
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QScrollArea, QWidget

from pixelsb.domain.geometry import GRID_ZOOM, pixel_at
from pixelsb.domain.labels import (
    LABEL_PAD,
    MIN_FONT,
    font_pixel_size,
    label_fits,
    region_texts,
    widest_text,
)
from pixelsb.domain.models import PixelCoord, RgbArray, ViewerState
from pixelsb.domain.samples import render_rgb
from pixelsb.ui import text
from pixelsb.ui.painting import qimage_from_rgb

_BACKGROUND = QColor("#121212")
_CURSOR = QColor("#ffb000")
_ANCHOR = QColor("#3aa0ff")
_GRID = QColor(0, 0, 0, 130)
_DRAG_THRESHOLD = 4
_DIM_DIVISOR = 12
_DARK_TEXT = QColor("#111111")
_LIGHT_TEXT = QColor("#f5f5f5")
_LUMA_WEIGHTS = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_THRESHOLD = 1_500_000


class ImageCanvas(QWidget):
    hovered = Signal(int, int)
    anchored = Signal(int, int)
    zoom_requested = Signal(int, int, int)
    file_dropped = Signal(str)

    def __init__(self, scroll: QScrollArea) -> None:
        super().__init__()
        self._scroll = scroll
        self._state = ViewerState()
        self._image: QImage | None = None
        self._rgb: RgbArray | None = None
        self._match: NDArray[np.bool_] | None = None
        self._cache_key: tuple[object, ...] | None = None
        self._label_key: tuple[object, ...] | None = None
        self._target = QSize(320, 240)
        self._space_down = False
        self._panning = False
        self._pan_origin = QPoint()
        self._scroll_origin = (0, 0)
        self._left_press: QPoint | None = None
        self._left_dragging = False
        self._left_press_scroll = (0, 0)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

    @property
    def state(self) -> ViewerState:
        return self._state

    def displayed_size(self) -> tuple[int, int] | None:
        image = self._state.image
        if image is None or self._image is None:
            return None
        return image.width, image.height

    def set_state(self, state: ViewerState, match: NDArray[np.bool_] | None = None) -> None:
        previous = self._state
        image = state.image
        rebuilt = False
        if image is None:
            rebuilt = self._image is not None
            self._image = None
            self._rgb = None
            self._cache_key = None
            self._target = QSize(320, 240)
        else:
            # The image object itself is the identity: id() values get recycled
            # after the previous image is freed, which would hit a stale cache.
            key = (image, state.selection, state.filter_expr)
            if key != self._cache_key:
                rgb = render_rgb(image, state.selection)
                if match is not None and match.shape == rgb.shape[:2]:
                    rgb[~match] //= _DIM_DIVISOR
                self._rgb = rgb
                self._image = qimage_from_rgb(rgb)
                self._cache_key = key
                rebuilt = True
            self._target = QSize(
                math.ceil(image.width * state.zoom),
                math.ceil(image.height * state.zoom),
            )
        label_key = (
            state.value_format,
            state.value_mode,
            state.anchor,
            state.selection,
            state.readout,
            state.detached,
            state.filter_expr,
            match is not None,
        )
        self._match = match
        labels_changed = label_key != self._label_key
        self._label_key = label_key
        self._state = state
        if self.size() != self._target:
            self.resize(self._target)
            self.updateGeometry()
            rebuilt = True
        if rebuilt or previous.zoom != state.zoom or labels_changed:
            self.update()
            return
        for coord in (previous.cursor, previous.anchor, state.cursor, state.anchor):
            self._repaint_pixel(coord, state.zoom)

    def _repaint_pixel(self, coord: PixelCoord | None, zoom: float) -> None:
        if coord is None:
            return
        pad = 2
        x = math.floor(coord.x * zoom) - pad
        y = math.floor(coord.y * zoom) - pad
        size = math.ceil(zoom) + pad * 2
        self.update(QRect(x, y, size, size))

    def _move_pan(self, global_pos: QPoint) -> None:
        delta = global_pos - self._pan_origin
        self._scroll.horizontalScrollBar().setValue(self._scroll_origin[0] - delta.x())
        self._scroll.verticalScrollBar().setValue(self._scroll_origin[1] - delta.y())

    def _start_pan(self, global_pos: QPoint) -> None:
        self._panning = True
        self._pan_origin = global_pos
        self._scroll_origin = (
            self._scroll.horizontalScrollBar().value(),
            self._scroll.verticalScrollBar().value(),
        )
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _drag_distance(self, event: QMouseEvent, press: QPoint) -> float:
        return (event.globalPosition().toPoint() - press).manhattanLength()

    def sizeHint(self) -> QSize:
        return self._target

    def minimumSizeHint(self) -> QSize:
        return self._target

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter, event)
        finally:
            # Never leave an active painter behind, even if drawing raises.
            painter.end()

    def _paint(self, painter: QPainter, event: QPaintEvent) -> None:
        painter.fillRect(event.rect(), _BACKGROUND)
        qimage = self._image
        image = self._state.image
        if qimage is None or image is None:
            painter.setPen(QColor("#9a9a9a"))
            painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, text.CANVAS_HINT)
            return
        zoom = self._state.zoom
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        source, dest = _visible_rects(QRectF(event.rect()), zoom, qimage.width(), qimage.height())
        if not dest.isEmpty():
            painter.drawImage(dest, qimage, source)
        if zoom >= GRID_ZOOM:
            _draw_grid(painter, source, zoom)
        if self._state.anchor == self._state.cursor:
            _draw_marker(painter, self._state.anchor, _ANCHOR, zoom, inset=0)
            _draw_marker(painter, self._state.cursor, _CURSOR, zoom, inset=2)
        else:
            _draw_marker(painter, self._state.anchor, _ANCHOR, zoom, inset=0)
            _draw_marker(painter, self._state.cursor, _CURSOR, zoom, inset=0)
        self._draw_labels(painter, self._state, source, zoom)

    def _draw_labels(
        self,
        painter: QPainter,
        state: ViewerState,
        source: QRectF,
        zoom: float,
    ) -> None:
        rgb = self._rgb
        image = state.image
        font = _label_font(widest_text(state), zoom)
        if font is None or rgb is None or image is None:
            return
        height, width = rgb.shape[:2]
        # Labels, colors, and the match mask are all clipped to the same bounds,
        # so they cannot disagree even if a cached buffer lags behind the state.
        x0 = max(int(source.x()), 0)
        y0 = max(int(source.y()), 0)
        x1 = min(x0 + int(source.width()), width, image.width)
        y1 = min(y0 + int(source.height()), height, image.height)
        columns = x1 - x0
        rows = y1 - y0
        if columns <= 0 or rows <= 0:
            return
        texts = region_texts(state, x0, y0, x1, y1)
        if not any(texts):
            return
        region = rgb[y0:y1, x0:x1].astype(np.uint32)
        bright = (region * _LUMA_WEIGHTS).sum(axis=-1) > _LUMA_THRESHOLD
        match = self._match
        if match is not None and match.shape != (height, width):
            match = None
        painter.setFont(font)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # The font is fitted to the template, so labels never leave their cell:
        # one clip for the whole visible area, no per-cell save/restore.
        painter.save()
        try:
            painter.setClipRect(QRectF(x0 * zoom, y0 * zoom, columns * zoom, rows * zoom))
            pen: QColor | None = None
            for index, label in enumerate(texts):
                if not label:
                    continue
                row, column = cell_of(index, columns)
                if match is not None and not match[y0 + row, x0 + column]:
                    continue
                rect = QRectF(
                    (x0 + column) * zoom,
                    (y0 + row) * zoom,
                    zoom,
                    zoom,
                )
                main = _DARK_TEXT if bright[row, column] else _LIGHT_TEXT
                if main is not pen:
                    painter.setPen(main)
                    pen = main
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        finally:
            painter.restore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._move_pan(event.globalPosition().toPoint())
            event.accept()
            return
        if self._left_press is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if (
                not self._left_dragging
                and self._drag_distance(event, self._left_press) > _DRAG_THRESHOLD
            ):
                self._left_dragging = True
                self._pan_origin = self._left_press
                self._scroll_origin = self._left_press_scroll
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._left_dragging:
                self._move_pan(event.globalPosition().toPoint())
            event.accept()
            return
        self._hover(event.position().toPoint())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_down
        ):
            self._start_pan(event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_press = event.globalPosition().toPoint()
            self._left_press_scroll = (
                self._scroll.horizontalScrollBar().value(),
                self._scroll.verticalScrollBar().value(),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._left_press is not None:
            dragging = self._left_dragging
            self._left_press = None
            self._left_dragging = False
            self.unsetCursor()
            if not dragging:
                coord = self._coord(event.position().toPoint())
                if coord is not None:
                    self.anchored.emit(coord.x, coord.y)
            event.accept()
            return
        if self._panning and event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        }:
            self._panning = False
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_down else Qt.CursorShape.ArrowCursor
            )
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        modifiers = event.modifiers()
        zooming = bool(
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        if zooming:
            step = 1 if event.angleDelta().y() > 0 else -1
            viewport_pos = self.mapTo(self._scroll.viewport(), event.position().toPoint())
            self.zoom_requested.emit(step, viewport_pos.x(), viewport_pos.y())
            event.accept()
            return
        dx, dy = _pan_delta(event)
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        horizontal.setValue(horizontal.value() - dx)
        vertical.setValue(vertical.value() - dy)
        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                self.file_dropped.emit(local)
                event.acceptProposedAction()
                return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            if not self._panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _hover(self, point: QPoint) -> None:
        coord = self._coord(point)
        if coord is None or coord == self._state.cursor:
            return
        self.hovered.emit(coord.x, coord.y)

    def _coord(self, point: QPoint) -> PixelCoord | None:
        image = self._state.image
        if image is None:
            return None
        return pixel_at(point.x(), point.y(), self._state.zoom, image.width, image.height)


@lru_cache(maxsize=64)
def _label_font(template: str, zoom: float) -> QFont | None:
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


def _visible_rects(
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


def _draw_grid(painter: QPainter, source: QRectF, zoom: float) -> None:
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


def _draw_marker(
    painter: QPainter,
    coord: PixelCoord | None,
    color: QColor,
    zoom: float,
    *,
    inset: int,
) -> None:
    if coord is None or zoom <= inset * 2:
        return
    pen = QPen(color)
    pen.setCosmetic(True)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    rect = QRectF(coord.x * zoom, coord.y * zoom, zoom, zoom)
    painter.drawRect(rect.adjusted(inset, inset, -1 - inset, -1 - inset))


def _pan_delta(event: QWheelEvent) -> tuple[int, int]:
    pixels = event.pixelDelta()
    if not pixels.isNull():
        return pixels.x(), pixels.y()
    return event.angleDelta().x() // 2, event.angleDelta().y() // 2

"""Zoomable image canvas. One image pixel maps to ``zoom`` logical pixels."""

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
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
_DARK_TEXT = QColor("#111111")
_LIGHT_TEXT = QColor("#f5f5f5")
_DARK_SHADOW = QColor(255, 255, 255, 150)
_LIGHT_SHADOW = QColor(0, 0, 0, 150)
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
        self._cache_key: tuple[object, ...] | None = None
        self._label_key: tuple[object, ...] | None = None
        self._target = QSize(320, 240)
        self._space_down = False
        self._panning = False
        self._pan_origin = QPoint()
        self._scroll_origin = (0, 0)
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

    def set_state(self, state: ViewerState) -> None:
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
            key = (id(image), state.selection)
            if key != self._cache_key:
                self._rgb = render_rgb(image, state.selection)
                self._image = qimage_from_rgb(self._rgb)
                self._cache_key = key
                rebuilt = True
            self._target = QSize(image.width * state.zoom, image.height * state.zoom)
        label_key = (
            state.value_format,
            state.value_mode,
            state.anchor,
            state.selection,
        )
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

    def _repaint_pixel(self, coord: PixelCoord | None, zoom: int) -> None:
        if coord is None:
            return
        pad = 2
        self.update(
            QRect(
                coord.x * zoom - pad,
                coord.y * zoom - pad,
                zoom + pad * 2,
                zoom + pad * 2,
            )
        )

    def sizeHint(self) -> QSize:
        return self._target

    def minimumSizeHint(self) -> QSize:
        return self._target

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(event.rect(), _BACKGROUND)
        qimage = self._image
        image = self._state.image
        if qimage is None or image is None:
            painter.setPen(QColor("#9a9a9a"))
            painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, text.CANVAS_HINT)
            painter.end()
            return
        zoom = self._state.zoom
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        source, dest = _visible_rects(event.rect(), zoom, qimage.width(), qimage.height())
        if source.width() > 0 and source.height() > 0:
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
        painter.end()

    def _draw_labels(
        self,
        painter: QPainter,
        state: ViewerState,
        source: QRect,
        zoom: int,
    ) -> None:
        template = widest_text(state)
        font = _label_font(template, zoom)
        if font is None or self._rgb is None:
            return
        x0, y0 = source.x(), source.y()
        columns = source.width()
        texts = region_texts(state, x0, y0, x0 + columns, y0 + source.height())
        if not any(texts):
            return
        region = self._rgb[y0 : y0 + source.height(), x0 : x0 + columns].astype(np.uint32)
        bright = (region * _LUMA_WEIGHTS).sum(axis=-1) > _LUMA_THRESHOLD
        painter.setFont(font)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        for index, label in enumerate(texts):
            if not label:
                continue
            column = index % columns
            row = index // columns
            rect = QRect((x0 + column) * zoom, (y0 + row) * zoom, zoom, zoom)
            if bright[row, column]:
                main, shadow = _DARK_TEXT, _LIGHT_SHADOW
            else:
                main, shadow = _LIGHT_TEXT, _DARK_SHADOW
            painter.save()
            painter.setClipRect(rect.adjusted(1, 1, -1, -1))
            painter.setPen(shadow)
            painter.drawText(rect.translated(1, 1), Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(main)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            painter.restore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.globalPosition().toPoint() - self._pan_origin
            self._scroll.horizontalScrollBar().setValue(self._scroll_origin[0] - delta.x())
            self._scroll.verticalScrollBar().setValue(self._scroll_origin[1] - delta.y())
            event.accept()
            return
        self._hover(event.position().toPoint())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        panning = event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_down
        )
        if panning:
            self._panning = True
            self._pan_origin = event.globalPosition().toPoint()
            self._scroll_origin = (
                self._scroll.horizontalScrollBar().value(),
                self._scroll.verticalScrollBar().value(),
            )
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            coord = self._coord(event.position().toPoint())
            if coord is not None:
                self.anchored.emit(coord.x, coord.y)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
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


def _label_font(template: str, zoom: int) -> QFont | None:
    if not label_fits(zoom, template):
        return None
    lines = template.split("\n")
    longest = max(lines, key=len)
    font = QFont()
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFamilies(["Menlo", "monospace"])
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


def _visible_rects(widget_rect: QRect, zoom: int, width: int, height: int) -> tuple[QRect, QRect]:
    src_x = max(widget_rect.x() // zoom, 0)
    src_y = max(widget_rect.y() // zoom, 0)
    src_right = min(widget_rect.right() // zoom + 1, width)
    src_bottom = min(widget_rect.bottom() // zoom + 1, height)
    source = QRect(src_x, src_y, max(src_right - src_x, 0), max(src_bottom - src_y, 0))
    dest = QRect(src_x * zoom, src_y * zoom, source.width() * zoom, source.height() * zoom)
    return source, dest


def _draw_grid(painter: QPainter, source: QRect, zoom: int) -> None:
    pen = QPen(_GRID)
    pen.setCosmetic(True)
    painter.setPen(pen)
    right = (source.x() + source.width()) * zoom
    bottom = (source.y() + source.height()) * zoom
    for column in range(source.x(), source.x() + source.width() + 1):
        x = column * zoom
        painter.drawLine(x, source.y() * zoom, x, bottom)
    for row in range(source.y(), source.y() + source.height() + 1):
        y = row * zoom
        painter.drawLine(source.x() * zoom, y, right, y)


def _draw_marker(
    painter: QPainter,
    coord: PixelCoord | None,
    color: QColor,
    zoom: int,
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
    rect = QRect(coord.x * zoom, coord.y * zoom, zoom, zoom)
    painter.drawRect(rect.adjusted(inset, inset, -1 - inset, -1 - inset))


def _pan_delta(event: QWheelEvent) -> tuple[int, int]:
    pixels = event.pixelDelta()
    if not pixels.isNull():
        return pixels.x(), pixels.y()
    return event.angleDelta().x() // 2, event.angleDelta().y() // 2

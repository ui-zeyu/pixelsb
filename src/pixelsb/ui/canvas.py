"""Zoomable image canvas. One image pixel maps to ``zoom`` logical pixels."""

import math
from enum import StrEnum
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QGestureEvent, QPinchGesture, QScrollArea, QWidget

from pixelsb.domain.geometry import GRID_ZOOM, pixel_at
from pixelsb.domain.labels import (
    LABEL_PAD,
    MIN_FONT,
    font_pixel_size,
    label_fits,
    region_texts,
    region_texts_at,
    widest_text,
)
from pixelsb.domain.match_view import MatchView, match_span, match_view
from pixelsb.domain.models import LoadedImage, PixelCoord, RgbArray, ViewerState
from pixelsb.domain.samples import render_rgb, render_rgb_at
from pixelsb.ui import text
from pixelsb.ui.painting import qimage_from_rgb
from pixelsb.ui.theme import CANVAS, TEXT, TEXT_MUTED

_BACKGROUND = QColor(CANVAS)
_CANVAS_RGB: tuple[int, int, int] = (
    QColor(CANVAS).red(),
    QColor(CANVAS).green(),
    QColor(CANVAS).blue(),
)
_CURSOR = QColor("#f59f00")
_CURSOR_HALO = QColor(255, 255, 255, 200)
_GRID = QColor(17, 20, 24, 26)
_DRAG_THRESHOLD = 4
_DIM_KEEP = 8
_DARK_TEXT = QColor("#111111")
_LIGHT_TEXT = QColor("#f5f5f5")
_LUMA_WEIGHTS = np.array([2126, 7152, 722], dtype=np.uint32)
_LUMA_THRESHOLD = 1_500_000


class CanvasMode(StrEnum):
    """What a plain left drag does: pan the view, or drag out a region."""

    PAN = "pan"
    SELECT = "select"


class ImageCanvas(QWidget):
    hovered = Signal(int, int)
    zoom_requested = Signal(int, int, int)
    zoom_scale_requested = Signal(float, int, int)
    file_dropped = Signal(str)
    region_selecting = Signal(int, int, int, int)
    region_selected = Signal(int, int, int, int)
    region_committed = Signal(int, int, int, int)
    region_canceled = Signal()

    def __init__(self, scroll: QScrollArea) -> None:
        super().__init__()
        self._scroll = scroll
        self._state = ViewerState()
        self._image: QImage | None = None
        self._rgb: RgbArray | None = None
        self._bright: NDArray[np.bool_] | None = None
        self._match: NDArray[np.bool_] | None = None
        self._view: MatchView | None = None
        self._no_match = False
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
        self._mode = CanvasMode.PAN
        self._marquee_origin: tuple[int, int] | None = None
        self._marquee: tuple[int, int, int, int] | None = None
        self.grabGesture(Qt.GestureType.PinchGesture)
        # Gestures over the letterbox around the image land on the viewport.
        scroll.viewport().installEventFilter(self)
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
        view = self._view
        if view is not None:
            return view.width, view.height
        return image.width, image.height

    def displayed_at(self, coord: PixelCoord) -> tuple[int, int] | None:
        """Where a source pixel sits on the canvas now; ``None`` when hidden."""
        view = self._view
        if view is not None:
            return view.display_of(coord)
        image = self._state.image
        if image is None or not 0 <= coord.x < image.width or not 0 <= coord.y < image.height:
            return None
        return coord.x, coord.y

    def set_mode(self, mode: CanvasMode) -> None:
        """Switch what a plain left drag does, and the cursor that announces it."""
        self._mode = mode
        if not self._panning and self._marquee_origin is None:
            self.setCursor(self._mode_cursor())

    def _mode_cursor(self) -> Qt.CursorShape:
        if self._mode is CanvasMode.SELECT:
            return Qt.CursorShape.CrossCursor
        return Qt.CursorShape.OpenHandCursor if self._space_down else Qt.CursorShape.ArrowCursor

    def _compacted_view(
        self,
        image: LoadedImage,
        state: ViewerState,
        match: NDArray[np.bool_] | None,
    ) -> MatchView | None:
        """Render only the matched pixels; ``None`` outside the compacted mode."""
        if not state.only_matched or match is None or match.shape != (image.height, image.width):
            return None
        span = match_span(match)
        if span is None:
            return None
        ys, xs = span
        rendered = render_rgb_at(image, state.selection, ys, xs)
        return match_view(rendered, match, ys, xs, _CANVAS_RGB)

    def set_state(self, state: ViewerState, match: NDArray[np.bool_] | None = None) -> None:
        previous = self._state
        image = state.image
        rebuilt = False
        if image is None:
            rebuilt = self._image is not None
            self._image = None
            self._rgb = None
            self._bright = None
            self._view = None
            self._no_match = False
            self._marquee_origin = None
            self._marquee = None
            self._cache_key = None
            self._target = QSize(320, 240)
        else:
            # The image object itself is the identity: id() values get recycled
            # after the previous image is freed, which would hit a stale cache.
            key = (image, state.selection, state.filter_expr, state.only_matched)
            if key != self._cache_key:
                view = self._compacted_view(image, state, match)
                if view is not None:
                    self._no_match = False
                    self._rgb = view.rgb
                    self._view = view
                    self._bright = None
                    self._image = qimage_from_rgb(view.rgb)
                else:
                    # Exact test for the empty state: a shape-stale match falls
                    # back to the plain render instead.
                    no_match = state.only_matched and match is not None and not match.any()
                    rgb = render_rgb(image, state.selection)
                    if match is not None and match.shape == rgb.shape[:2] and not no_match:
                        _fade_out(rgb, match)
                    self._no_match = no_match
                    self._rgb = None if no_match else rgb
                    self._view = None
                    self._bright = None
                    self._image = qimage_from_rgb(rgb) if self._rgb is not None else None
                self._cache_key = key
                rebuilt = True
            if self._no_match:
                self._target = QSize(320, 240)
            else:
                view = self._view
                width = view.width if view is not None else image.width
                height = view.height if view is not None else image.height
                self._target = QSize(
                    math.ceil(width * state.zoom),
                    math.ceil(height * state.zoom),
                )
        label_key = (
            state.value_format,
            state.selection,
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
        for coord in (previous.cursor, state.cursor):
            self._repaint_pixel(coord, state.zoom)

    def _repaint_pixel(self, coord: PixelCoord | None, zoom: float) -> None:
        if coord is None:
            return
        cell = self.displayed_at(coord)
        if cell is None:
            return
        dx, dy = cell
        pad = 2
        x = math.floor(dx * zoom) - pad
        y = math.floor(dy * zoom) - pad
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
            if self._no_match:
                _draw_empty_state(painter, self.rect(), hint=text.FILTER_NO_MATCH, shortcut="")
            else:
                _draw_empty_state(painter, self.rect())
            return
        zoom = self._state.zoom
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        source, dest = _visible_rects(QRectF(event.rect()), zoom, qimage.width(), qimage.height())
        if not dest.isEmpty():
            painter.drawImage(dest, qimage, source)
        if zoom >= GRID_ZOOM:
            _draw_grid(painter, source, zoom)
        _draw_marker(painter, self._view, self._state.cursor, zoom)
        _draw_marquee(painter, self._marquee, zoom)
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
        view = self._view
        if view is None:
            texts = region_texts(state, x0, y0, x1, y1)
            match = self._match
            if match is not None and match.shape != (height, width):
                match = None
        else:
            texts = region_texts_at(state, view.xs[x0:x1], view.ys[y0:y1])
            match = view.match
        if not any(texts):
            return
        if self._bright is None:
            self._bright = _bright_map(rgb)
        bright = self._bright[y0:y1, x0:x1]
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
        if self._marquee_origin is not None:
            self._move_marquee(event.position().toPoint())
            event.accept()
            return
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
        if event.button() == Qt.MouseButton.LeftButton and self._mode is CanvasMode.SELECT:
            self._start_marquee(event.position().toPoint())
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
        if self._marquee_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            self._finish_marquee()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._left_press is not None:
            self._left_press = None
            self._left_dragging = False
            self.setCursor(self._mode_cursor())
            event.accept()
            return
        if self._panning and event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        }:
            self._panning = False
            self.setCursor(self._mode_cursor())
        super().mouseReleaseEvent(event)

    def event(self, event: QEvent) -> bool:
        if self._handle_gesture(event):
            return True
        return super().event(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if self._handle_gesture(event):
            return True
        if isinstance(event, QWheelEvent) and self._zoom_modifier(event):
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)

    def _handle_gesture(self, event: QEvent) -> bool:
        """True for pinch events, whether they land on the canvas or the viewport."""
        if isinstance(event, QNativeGestureEvent):
            self._native_gesture(event)
            return True
        if isinstance(event, QGestureEvent):
            self._pinch_gesture(event)
            return True
        return False

    def _native_gesture(self, event: QNativeGestureEvent) -> None:
        """Trackpad pinch on macOS arrives as native magnification events."""
        if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self._emit_scale(1.0 + event.value())
        elif event.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture:
            self._emit_scale(2.0 if event.value() > 0 else 0.5)

    def _pinch_gesture(self, event: QGestureEvent) -> None:
        """Touchscreen pinch, delivered through Qt's gesture framework."""
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if isinstance(pinch, QPinchGesture) and pinch.state() == Qt.GestureState.GestureUpdated:
            self._emit_scale(pinch.scaleFactor())

    def _emit_scale(self, factor: float) -> None:
        """Scale around the pointer: while gesturing, that is the anchor."""
        if factor <= 0 or factor == 1.0:
            return
        anchor_x, anchor_y = self._cursor_in_viewport()
        self.zoom_scale_requested.emit(factor, anchor_x, anchor_y)

    def _cursor_in_viewport(self) -> tuple[int, int]:
        point = self._scroll.viewport().mapFromGlobal(QCursor.pos())
        return point.x(), point.y()

    def _zoom_modifier(self, event: QWheelEvent) -> bool:
        modifiers = event.modifiers()
        return bool(
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        zooming = self._zoom_modifier(event)
        if zooming:
            pixels = event.pixelDelta()
            if pixels.isNull():
                step = 1 if event.angleDelta().y() > 0 else -1
                anchor_x, anchor_y = self._cursor_in_viewport()
                self.zoom_requested.emit(step, anchor_x, anchor_y)
            else:  # a trackpad: zoom continuously with the fingers
                factor = math.exp(pixels.y() / 250.0)
                anchor_x, anchor_y = self._cursor_in_viewport()
                self.zoom_scale_requested.emit(factor, anchor_x, anchor_y)
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
            if local := url.toLocalFile():
                self.file_dropped.emit(local)
                event.acceptProposedAction()
                return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and (
            self._marquee_origin is not None or self._marquee is not None
        ):
            self._cancel_marquee()
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and self._marquee is not None
            and self._marquee_origin is None
        ):
            self._commit_selection()
            event.accept()
            return
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
            self.setCursor(self._mode_cursor())
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
        view = self._view
        width = view.width if view is not None else image.width
        height = view.height if view is not None else image.height
        cell = pixel_at(point.x(), point.y(), self._state.zoom, width, height)
        if cell is None:
            return None
        if view is not None:
            return view.source_at(cell.x, cell.y)
        return cell

    def _clamped_pixel(self, point: QPoint) -> tuple[int, int] | None:
        """The pixel under ``point``, clamped into the shown raster."""
        image = self._state.image
        if image is None:
            return None
        view = self._view
        width = view.width if view is not None else image.width
        height = view.height if view is not None else image.height
        zoom = self._state.zoom
        x = min(max(int(point.x() / zoom), 0), width - 1)
        y = min(max(int(point.y() / zoom), 0), height - 1)
        if view is not None:
            return int(view.xs[x]), int(view.ys[y])
        return x, y

    def _start_marquee(self, point: QPoint) -> None:
        origin = self._clamped_pixel(point)
        if origin is None:
            return
        self._marquee_origin = origin
        self._marquee = (*origin, *origin)
        self.region_selecting.emit(*self._marquee)
        self.update()

    def _move_marquee(self, point: QPoint) -> None:
        current = self._clamped_pixel(point)
        origin = self._marquee_origin
        if current is None or origin is None:
            return
        self._marquee = (
            min(origin[0], current[0]),
            min(origin[1], current[1]),
            max(origin[0], current[0]),
            max(origin[1], current[1]),
        )
        self.region_selecting.emit(*self._marquee)
        self.update()

    def _finish_marquee(self) -> None:
        """Drag finished: a real region stays active, a plain click is nothing."""
        rect = self._marquee
        self._marquee_origin = None
        if rect is None:
            return
        if rect[0] < rect[2] or rect[1] < rect[3]:
            self.region_selected.emit(*rect)
            return
        self._marquee = None
        self.region_canceled.emit()
        self.update()

    def _commit_selection(self) -> None:
        rect = self._marquee
        if rect is None or self._marquee_origin is not None:
            return
        self._marquee = None
        self.update()
        self.region_committed.emit(*rect)

    def _cancel_marquee(self) -> None:
        if self._marquee is None and self._marquee_origin is None:
            return
        self._marquee_origin = None
        self._marquee = None
        self.update()
        self.region_canceled.emit()


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


def _bright_map(rgb: RgbArray) -> NDArray[np.bool_]:
    """Per-pixel "light enough for dark text", computed once per rendered image."""
    weighted = rgb.astype(np.uint32) * _LUMA_WEIGHTS
    return weighted.sum(axis=-1) > _LUMA_THRESHOLD


def _fade_out(rgb: RgbArray, match: NDArray[np.bool_]) -> None:
    """Push the pixels that fail the filter toward the canvas color, in place."""
    rgb[~match] = 255 - (255 - rgb[~match]) // _DIM_KEEP


def _draw_empty_state(
    painter: QPainter,
    rect: QRect,
    hint: str = text.CANVAS_HINT,
    shortcut: str = text.CANVAS_SHORTCUT,
) -> None:
    """Centered lines: what to do, and the shortcut that does it."""
    font = QFont(painter.font())
    font.setPixelSize(15)
    painter.setFont(font)
    painter.setPen(QColor(TEXT))
    painter.drawText(_shifted(rect, -18), Qt.AlignmentFlag.AlignCenter, hint)
    if not shortcut:
        return
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(QColor(TEXT_MUTED))
    painter.drawText(_shifted(rect, 18), Qt.AlignmentFlag.AlignCenter, shortcut)


def _shifted(rect: QRect, offset: int) -> QRect:
    return QRect(rect.x(), rect.y() + offset, rect.width(), rect.height())


def _draw_marker(
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


def _draw_marquee(
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


def _pan_delta(event: QWheelEvent) -> tuple[int, int]:
    pixels = event.pixelDelta()
    if not pixels.isNull():
        return pixels.x(), pixels.y()
    return event.angleDelta().x() // 2, event.angleDelta().y() // 2

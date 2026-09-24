"""Zoomable image canvas. One image pixel maps to ``zoom`` logical pixels."""

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import override

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPaintEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QGestureEvent, QPinchGesture, QScrollArea, QWidget

from pixelsb.domain.geometry import GRID_ZOOM, pixel_at
from pixelsb.domain.labels import (
    region_texts,
    region_texts_at,
    widest_text,
)
from pixelsb.domain.match_view import MatchView, match_span, match_view
from pixelsb.domain.models import LoadedImage, PixelCoord, RgbArray, ViewerState
from pixelsb.domain.samples import render_rgb, render_rgb_at
from pixelsb.ui import text
from pixelsb.ui.painting import (
    bright_map,
    cell_of,
    draw_empty_state,
    draw_grid,
    draw_marker,
    draw_marquee,
    fade_out,
    label_color,
    label_font,
    qimage_from_rgb,
    visible_rects,
)
from pixelsb.ui.theme import CANVAS

_BACKGROUND = QColor(CANVAS)
_CANVAS_RGB: tuple[int, int, int] = (
    QColor(CANVAS).red(),
    QColor(CANVAS).green(),
    QColor(CANVAS).blue(),
)
_DRAG_THRESHOLD = 4
_EMPTY_TARGET = QSize(320, 240)


@dataclass(frozen=True, slots=True)
class _Frame:
    """Everything one viewer state draws, and the keys that say when to rebuild.

    ``qimage`` is what the painter blits, ``rgb`` and ``match`` are what the
    numeric labels read, and ``target`` is the canvas size the zoom asks for.
    The empty state has no pixels, so it holds a placeholder size instead.
    """

    qimage: QImage | None = None
    rgb: RgbArray | None = None
    view: MatchView | None = None
    match: NDArray[np.bool_] | None = None
    no_match: bool = False
    target: QSize = _EMPTY_TARGET
    content_key: tuple[object, ...] | None = None
    label_key: tuple[object, ...] | None = None

    @property
    def width(self) -> int:
        if self.view is not None:
            return self.view.width
        return 0 if self.rgb is None else self.rgb.shape[1]

    @property
    def height(self) -> int:
        if self.view is not None:
            return self.view.height
        return 0 if self.rgb is None else self.rgb.shape[0]

    def at_zoom(self, zoom: float) -> QSize:
        """The canvas size these pixels ask for at ``zoom``."""
        if self.qimage is None:
            return _EMPTY_TARGET
        return QSize(math.ceil(self.width * zoom), math.ceil(self.height * zoom))


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
        self._frame = _Frame()
        self._bright: NDArray[np.bool_] | None = None
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
        frame = self._frame
        if image is None or frame.qimage is None:
            return None
        return frame.width, frame.height

    def displayed_at(self, coord: PixelCoord) -> tuple[int, int] | None:
        """Where a source pixel sits on the canvas now; ``None`` when hidden."""
        view = self._frame.view
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

    def set_state(
        self,
        state: ViewerState,
        match: NDArray[np.bool_] | None = None,
    ) -> None:
        previous = self._state
        if state.image is None or state.image is not previous.image:
            # The region belongs to the image it was dragged on.
            self._marquee_origin = None
            self._marquee = None
        frame = self._frame_for(state, match)
        if frame.rgb is not self._frame.rgb:
            self._bright = None
        rebuilt = frame.content_key != self._frame.content_key
        relabelled = frame.label_key != self._frame.label_key
        resized = self.size() != frame.target
        if resized:
            self.resize(frame.target)
            self.updateGeometry()
        self._frame = frame
        self._state = state
        if rebuilt or relabelled or resized or previous.zoom != state.zoom:
            self.update()
            return
        for coord in (previous.cursor, state.cursor):
            self._repaint_pixel(coord, state.zoom)

    def _frame_for(
        self,
        state: ViewerState,
        match: NDArray[np.bool_] | None,
    ) -> _Frame:
        """The frame for one state, rendering again only when its content moved."""
        image = state.image
        if image is None:
            return _Frame()
        # The image object itself is the identity: id() values get recycled
        # after the previous image is freed, which would hit a stale cache.
        content = (image, state.selection, state.filter_expr, state.only_matched)
        labels = (state.value_format, state.selection, state.filter_expr, match is not None)
        frame = self._frame
        if content == frame.content_key:
            # The same pixels: only the zoom, the match, or the numbers changed.
            return replace(frame, match=match, label_key=labels, target=frame.at_zoom(state.zoom))
        return self._render_frame(image, state, match, content, labels)

    def _render_frame(
        self,
        image: LoadedImage,
        state: ViewerState,
        match: NDArray[np.bool_] | None,
        content: tuple[object, ...],
        labels: tuple[object, ...],
    ) -> _Frame:
        """Render one state's pixels, at whatever the current zoom asks for."""
        frame = self._rendered_frame(image, state, match, content, labels)
        return replace(frame, target=frame.at_zoom(state.zoom))

    def _rendered_frame(
        self,
        image: LoadedImage,
        state: ViewerState,
        match: NDArray[np.bool_] | None,
        content: tuple[object, ...],
        labels: tuple[object, ...],
    ) -> _Frame:
        view = self._compacted_view(image, state, match)
        if view is not None:
            return _Frame(
                qimage=qimage_from_rgb(view.rgb),
                rgb=view.rgb,
                view=view,
                match=match,
                content_key=content,
                label_key=labels,
            )
        if state.only_matched and match is not None and not match.any():
            return _Frame(match=match, no_match=True, content_key=content, label_key=labels)
        # A shape-stale match falls back to the plain render rather than fading
        # pixels it does not describe.
        rgb = render_rgb(image, state.selection)
        if match is not None and match.shape == rgb.shape[:2]:
            fade_out(rgb, match)
        return _Frame(
            qimage=qimage_from_rgb(rgb),
            rgb=rgb,
            match=match,
            content_key=content,
            label_key=labels,
        )

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

    @override
    def sizeHint(self) -> QSize:
        return self._frame.target

    @override
    def minimumSizeHint(self) -> QSize:
        return self._frame.target

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter, event)
        finally:
            # Never leave an active painter behind, even if drawing raises.
            painter.end()

    def _paint(self, painter: QPainter, event: QPaintEvent) -> None:
        painter.fillRect(event.rect(), _BACKGROUND)
        frame = self._frame
        qimage = frame.qimage
        image = self._state.image
        if qimage is None or image is None:
            if frame.no_match:
                draw_empty_state(painter, self.rect(), hint=text.FILTER_NO_MATCH, shortcut="")
            else:
                draw_empty_state(painter, self.rect())
            return
        zoom = self._state.zoom
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        source, dest = visible_rects(QRectF(event.rect()), zoom, qimage.width(), qimage.height())
        if not dest.isEmpty():
            painter.drawImage(dest, qimage, source)
        if zoom >= GRID_ZOOM:
            draw_grid(painter, source, zoom)
        draw_marker(painter, frame.view, self._state.cursor, zoom)
        draw_marquee(painter, self._marquee, zoom)
        self._draw_labels(painter, self._state, source, zoom)

    def _draw_labels(
        self,
        painter: QPainter,
        state: ViewerState,
        source: QRectF,
        zoom: float,
    ) -> None:
        frame = self._frame
        rgb = frame.rgb
        image = state.image
        font = label_font(widest_text(state), zoom)
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
        view = frame.view
        if view is None:
            texts = region_texts(state, x0, y0, x1, y1)
            match = frame.match
            if match is not None and match.shape != (height, width):
                match = None
        else:
            texts = region_texts_at(state, view.xs[x0:x1], view.ys[y0:y1])
            match = view.match
        if not any(texts):
            return
        if self._bright is None:
            self._bright = bright_map(rgb)
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
                main = label_color(bright[row, column])
                if main is not pen:
                    painter.setPen(main)
                    pen = main
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        finally:
            painter.restore()

    @override
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

    @override
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

    @override
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

    @override
    def event(self, event: QEvent) -> bool:
        if self._handle_gesture(event):
            return True
        return super().event(event)

    @override
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

    @override
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

    @override
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    @override
    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if local := url.toLocalFile():
                self.file_dropped.emit(local)
                event.acceptProposedAction()
                return
        event.ignore()

    @override
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

    @override
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
        view = self._frame.view
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
        view = self._frame.view
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


def _pan_delta(event: QWheelEvent) -> tuple[int, int]:
    pixels = event.pixelDelta()
    if not pixels.isNull():
        return pixels.x(), pixels.y()
    return event.angleDelta().x() // 2, event.angleDelta().y() // 2

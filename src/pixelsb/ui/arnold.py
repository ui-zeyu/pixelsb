"""The cat-map brute force, as a sidebar page: thumbnails, seated as they arrive.

A scrambled picture is judged by eye, so the gallery's candidates are
thumbnails, not lines of text: the search runs the transform over the canvas's
own pixels in a worker thread and seats each candidate the moment it lands, so
the restored picture shows up while the sweep is still running. Each thumbnail
carries the heuristic's smoothness score — the restored photo is smooth, a wrong
parameter set leaves confetti — and the grid is laid out smoothest first once
the sweep ends. Clicking one writes it into the stack as an ordinary 猫脸变换
operation; the page rewrites the layer it last wrote, so clicking through
candidates never piles up transforms, and the results it is clicking through
stay on the page until a new picture arrives.
"""

from collections.abc import Iterator
from functools import partial
from typing import cast, override

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.models import (
    ARNOLD_PARAM_LIMIT,
    COLOR_SLOTS,
    GRAY_SLOTS,
    LoadedImage,
    Raster,
    SampleArray,
    SamplePlane,
    plane_or_none,
)
from pixelsb.domain.samples import scale_to_byte, viewed_samples
from pixelsb.domain.stack import arnold_image
from pixelsb.ui import painting, text, theme
from pixelsb.ui.controls import drain, section_title
from pixelsb.ui.worker import Stoppable, shutdown

_THUMB = 96  # a candidate's thumbnail, square
_SPIN_WIDTH = 104  # wide enough for every digit ±2147483647 can ask for


class _BruteWorker(Stoppable):
    """The sweep over a parameter cube, one thumbnail per candidate.

    ``samples`` are the values the canvas shows — the selection's bits where it
    carries a plane — and ``tops`` are the ranges those values end at, so the
    thumbnails scale the way the canvas does. ``jobs`` is consumed lazily, so a
    wide range costs the page nothing until each candidate is actually
    transformed, and ``stop`` answers between them.
    """

    found = Signal(int, int, int, float, QImage)
    reached = Signal(int)

    def __init__(
        self,
        samples: SampleArray,
        tops: tuple[int, ...],
        planes: tuple[SamplePlane, ...],
        jobs: Iterator[tuple[int, int, int]],
        total: int,
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self._samples = samples
        self._tops = tops
        self._planes = planes
        self._jobs = jobs
        self.total = total
        self.count = 0

    @override
    def run(self) -> None:
        for times, a, b in self._jobs:
            if self._stopping:
                break
            decoded = arnold_image(self._samples, times, a, b)
            rgb = _as_rgb(decoded, self._planes, self._tops)
            self.found.emit(times, a, b, _smoothness(rgb), _thumbnail(rgb))
            self.count += 1
            self.reached.emit(self.count)


def _picture_indexes(planes: tuple[SamplePlane, ...]) -> tuple[int, ...]:
    """Which planes make the picture's color: the triplet, or one gray plane.

    Chosen by name the way the canvas chooses them, so a palette image shows
    the colors it looks up — taking the first three channels as RGB is what
    painted such thumbnails cyan.
    """
    red, green, blue = (plane_or_none(planes, name) for name in COLOR_SLOTS)
    if red is not None and green is not None and blue is not None:
        return red.index, green.index, blue.index
    gray = next(
        (plane for name in GRAY_SLOTS if (plane := plane_or_none(planes, name)) is not None),
        planes[0],
    )
    return (gray.index,)


def _as_rgb(
    samples: SampleArray, planes: tuple[SamplePlane, ...], tops: tuple[int, ...]
) -> NDArray[np.uint8]:
    """The picture as HxWx3 bytes, at the scale the canvas shows each plane.

    A gray picture repeats its one plane, so an alpha plane must never dress
    itself up as a color channel, which is what turned gray-plus-alpha thumbnails
    teal.
    """
    indexes = _picture_indexes(planes)
    scaled = [scale_to_byte(samples[:, :, index], tops[index]) for index in indexes]
    if len(scaled) == 1:
        scaled = scaled * 3
    return np.stack(scaled, axis=-1)


def _smoothness(rgb: NDArray[np.uint8]) -> float:
    """How picture-like a candidate looks, 1.0 for flat and near 0.0 for confetti.

    The restored photo is smooth almost everywhere, while a wrong parameter set
    leaves noise at every pixel, so the inverse mean absolute gradient sorts the
    grid the way the eye would — a heuristic, and the thumbnails themselves stay
    the final judge.
    """
    gray = rgb.astype(np.float64).mean(axis=-1)
    horizontal = float(np.abs(np.diff(gray, axis=1)).mean())
    vertical = float(np.abs(np.diff(gray, axis=0)).mean())
    return 1.0 / (1.0 + (horizontal + vertical) / 2.0)


def _thumbnail(rgb: NDArray[np.uint8]) -> QImage:
    """The picture at thumbnail size; nearest scaling keeps the pixels honest."""
    image = painting.qimage_from_rgb(rgb)
    return image.scaled(
        _THUMB,
        _THUMB,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


class ArnoldPanel(QWidget):
    """One sidebar page sweeping the cat map's parameter cube over the canvas.

    The page follows the canvas: whatever the recipe stack currently shows is
    what the sweep runs on. A run is started by hand, can be stopped, fills the
    grid while it runs, and seats what it found smoothest first once it is done.
    """

    picked = Signal(int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self._image: LoadedImage | None = None
        self._raster: Raster | None = None
        self._worker: _BruteWorker | None = None
        self._ranges: tuple[tuple[QSpinBox, QSpinBox], ...] = ()
        self._entries: list[tuple[float, int, int, int, QImage]] = []
        self._buttons: list[QToolButton] = []
        self._commands: dict[QToolButton, tuple[int, int, int]] = {}
        self._chosen: tuple[int, int, int] | None = None
        self._cell = _THUMB  # the width one thumbnail's column needs
        self._seats = 1  # how many of them a row holds at the page's width
        self._square = False
        self._built = False  # a resize before the page is built has nothing to seat
        self._progress = QLabel()
        self._progress.setObjectName("note")
        self._status = QLabel()
        self._status.setObjectName("note")
        self._status.setWordWrap(True)
        # One row per parameter, so the page stays narrow enough for the rail's
        # column: the ranges are wide, the page need not be.
        rows = QGridLayout()
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(6)
        for row, (label, low, high, first, last) in enumerate(
            (
                (text.ARNOLD_TIMES, 1, ARNOLD_PARAM_LIMIT, 1, 5),
                (text.ARNOLD_A, -ARNOLD_PARAM_LIMIT, ARNOLD_PARAM_LIMIT, 1, 5),
                (text.ARNOLD_B, -ARNOLD_PARAM_LIMIT, ARNOLD_PARAM_LIMIT, 1, 5),
            )
        ):
            name = QLabel(label)
            name.setObjectName("note")
            rows.addWidget(name, row, 0)
            pair: list[QSpinBox] = []
            for column, value in enumerate((first, last)):
                box = QSpinBox()
                box.setRange(low, high)
                box.setFixedWidth(_SPIN_WIDTH)
                box.setValue(value)
                pair.append(box)
                rows.addWidget(box, row, 1 + 2 * column)
            dash = QLabel(text.RANGE_DASH, self)
            dash.setObjectName("note")
            dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rows.addWidget(dash, row, 2)
            self._ranges += ((pair[0], pair[1]),)
        # All the page's spare width goes past the row, so the ranges hug the left.
        rows.setColumnStretch(4, 1)
        self._run = QPushButton(text.ARNOLD_START)
        self._run.setFixedHeight(theme.CONTROL_HEIGHT)
        self._run.clicked.connect(self._toggle_run)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        buttons.addWidget(self._run)
        buttons.addWidget(self._progress, 1)
        self._grid = QGridLayout()
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        host = QWidget()
        host.setLayout(self._grid)
        self._holder = QScrollArea()
        self._holder.setWidgetResizable(True)
        self._holder.setFrameShape(QScrollArea.Shape.NoFrame)
        self._holder.setWidget(host)
        self._holder.setMinimumHeight(_THUMB * 2 + 60)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(section_title(text.SECTION_ARNOLD))
        layout.addLayout(rows)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addWidget(self._holder, 1)
        self._built = True
        self._set_running(False)

    def set_source(self, image: LoadedImage | None, raster: Raster | None) -> None:
        """Point the page at the canvas's picture; only a new image clears the grid.

        The stack moving under the page is the ordinary case — picking a thumbnail
        rewrites it — and the gallery keeps its results then: the user is still
        clicking through one picture's brute force. Starting a new run sweeps the
        picture as the stack shows it now.
        """
        if image is self._image and raster is self._raster:
            return
        if image is not self._image:
            self._stop_run()
            self._clear_grid()
            self._chosen = None
        self._image = image
        self._raster = raster
        samples = None if raster is None else raster.samples
        self._square = samples is not None and samples.shape[0] == samples.shape[1]
        self._set_running(self._worker is not None)
        if samples is None:
            self._status.setText(text.NO_IMAGE)
            self._restyle_status(warn=False)
        elif self._square:
            self._status.setText("")
        else:
            self._status.setText(text.arnold_not_square(samples.shape[1], samples.shape[0]))
            self._restyle_status(warn=True)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        """A wider rail fits more thumbnails in a row, so the grid is laid out again."""
        super().resizeEvent(event)
        if self._built and self._columns() != self._seats:
            self._fill_grid()

    def shutdown(self) -> None:
        """Stop the sweep before the window goes away, so the thread joins."""
        self._stop_run()

    def _on_start(self) -> None:
        raster = self._raster
        if raster is None or self._worker is not None:
            return
        total = self._job_count()
        if not total:
            return
        self._clear_grid()
        self._seats = self._columns()
        samples, tops = viewed_samples(raster)
        self._worker = _BruteWorker(samples, tops, raster.planes, self._jobs(), total, self)
        self._worker.found.connect(self._on_found)
        self._worker.reached.connect(self._on_reached)
        self._worker.finished.connect(partial(self._on_done, self._worker))
        self._set_running(True)
        self._worker.start()

    def _job_count(self) -> int:
        """How many candidates the ranges ask for, without building them."""
        count = 1
        for low, high in self._ranges:
            count *= max(high.value() - low.value() + 1, 0)
        return count

    def _toggle_run(self) -> None:
        """The one button runs the sweep, or stops the run it is sweeping."""
        if self._worker is not None:
            self._worker.stop()
        else:
            self._on_start()

    def _jobs(self) -> Iterator[tuple[int, int, int]]:
        (times_from, times_to), (a_from, a_to), (b_from, b_to) = self._ranges
        for times in range(times_from.value(), times_to.value() + 1):
            for a in range(a_from.value(), a_to.value() + 1):
                for b in range(b_from.value(), b_to.value() + 1):
                    yield times, a, b

    def _on_found(self, times: int, a: int, b: int, score: float, image: QImage) -> None:
        """One candidate lands: it goes up right away, the order is settled at the end."""
        self._entries.append((score, times, a, b, image))
        button = self._button(times, a, b, score, image)
        if button.sizeHint().width() > self._cell:
            # A wider caption fits fewer thumbnails to a row, so the grid is laid
            # out again — a handful of times in a sweep, as the numbers grow.
            self._cell = button.sizeHint().width()
            self._fill_grid()
        else:
            self._seat(button, (times, a, b))

    def _on_reached(self, done: int) -> None:
        total = self._worker.total if self._worker is not None else done
        self._progress.setText(text.arnold_progress(done, total))

    def _on_done(self, worker: _BruteWorker) -> None:
        """The sweep ended and seats its candidates in the heuristic's order.

        The worker names itself, so a run the page already dropped — stopping and
        starting again, or a new picture — settles nothing here.
        """
        if worker is not self._worker:
            return
        self._worker = None
        self._progress.setText(
            text.arnold_cancelled(worker.count)
            if worker.count < worker.total
            else text.arnold_done(worker.count)
        )
        worker.deleteLater()
        self._fill_grid()
        self._set_running(False)

    def _fill_grid(self) -> None:
        """The candidates as thumbnails, smoothest first — the heuristic's order.

        The score only decides the seating; the tooltip carries it, and the eye
        still makes the call.
        """
        self._drain_grid()
        self._seats = self._columns()
        for score, times, a, b, image in sorted(self._entries, reverse=True):
            self._seat(self._button(times, a, b, score, image), (times, a, b))
        self._mark_chosen()

    def _columns(self) -> int:
        """How many thumbnails fit one row of the page as it is now.

        The rail's width is the user's to change, so the grid is laid out from it
        rather than from a fixed count, and a wide sidebar shows a wide gallery.
        A column is as wide as the widest thumbnail on the page: the buttons
        carry the command under the picture, and that caption is the wide part.
        """
        spacing = max(self._grid.horizontalSpacing(), 0)
        return max(1, (self._holder.viewport().width() + spacing) // (self._cell + spacing))

    def _button(self, times: int, a: int, b: int, score: float, image: QImage) -> QToolButton:
        button = QToolButton()
        button.setObjectName("galleryCard")
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setIcon(QPixmap.fromImage(image))
        button.setIconSize(image.size())
        button.setText(text.arnold_caption(times, a, b))
        button.setToolTip(text.arnold_pick_tip(times, a, b, score))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(partial(self._pick, times, a, b))
        return button

    def _pick(self, times: int, a: int, b: int) -> None:
        """One card is the user's choice: mark it, then write it into the stack."""
        self._chosen = (times, a, b)
        self._mark_chosen()
        self.picked.emit(times, a, b)

    def _mark_chosen(self) -> None:
        """Restyle the cards so the chosen one reads as chosen."""
        for button, command in self._commands.items():
            wanted = command == self._chosen
            if button.property("chosen") != wanted:
                button.setProperty("chosen", wanted)
                theme.repolish(button)

    def move_selection(self, down: int, right: int) -> bool:
        """One seat over, the edges holding: True when the page has one to move to.

        The canvas's arrow shortcuts stand down on this page: here the arrows
        browse candidates, and each one they land on is applied, so browsing is
        seeing. Starting from nothing chooses the first card.
        """
        if self._worker is not None or not self._commands:
            return False
        seats: dict[tuple[int, int], QToolButton] = {}
        for index in range(self._grid.count()):
            item = self._grid.itemAt(index)
            widget = None if item is None else item.widget()
            if isinstance(widget, QToolButton):
                row, column, _, _ = cast(
                    tuple[int, int, int, int], self._grid.getItemPosition(index)
                )
                seats[row, column] = widget
        here = next(
            (seat for seat, widget in seats.items() if self._commands.get(widget) == self._chosen),
            None,
        )
        if here is None:
            target = min(seats)
        else:
            row, column = here
            row = min(max(row + down, 0), max(seat_row for seat_row, _ in seats))
            columns_of_row = [seat_column for seat_row, seat_column in seats if seat_row == row]
            column = min(max(column + right, 0), max(columns_of_row))
            if (row, column) not in seats:  # the short last row
                column = max(columns_of_row) if right < 0 else min(columns_of_row)
            target = (row, column)
        widget = seats[target]
        if self._commands.get(widget) == self._chosen:
            return False
        widget.click()
        return True

    def _seat(self, button: QToolButton, command: tuple[int, int, int]) -> None:
        """Put one thumbnail after the ones already up, wrapping at the row's width."""
        self._buttons.append(button)
        self._commands[button] = command
        count = len(self._buttons) - 1
        self._grid.addWidget(button, count // self._seats, count % self._seats)

    def _drain_grid(self) -> None:
        drain(self._grid)
        self._buttons = []
        self._commands = {}

    def _clear_grid(self) -> None:
        """Nothing on the page: no candidates, no thumbnails, and a fresh row width."""
        self._drain_grid()
        self._entries = []
        self._cell = _THUMB
        self._progress.setText("")

    def _set_running(self, running: bool) -> None:
        """The button names the action a click would take now."""
        self._run.setText(text.ARNOLD_STOP if running else text.ARNOLD_START)
        self._run.setEnabled(self._square or running)

    def _restyle_status(self, *, warn: bool) -> None:
        """A non-square canvas is a warning; no image at all is just a note."""
        self._status.setObjectName("warnNote" if warn else "note")
        theme.repolish(self._status)

    def _stop_run(self) -> None:
        worker, self._worker = self._worker, None
        shutdown(worker)

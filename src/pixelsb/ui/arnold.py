"""The cat-map brute force, as a sidebar page: thumbnails sorted by plausibility.

A scrambled picture is judged by eye, so the gallery's candidates are
thumbnails, not lines of text: the search runs the transform over the canvas's
own pixels in a worker thread and fills the grid as candidates arrive. Each
thumbnail carries the heuristic's smoothness score — the restored photo is
smooth, a wrong parameter set leaves confetti — and the grid seats the
smoothest first. Clicking one writes it into the stack as an ordinary 猫脸变换
operation; the page rewrites the layer it last wrote, so clicking through
candidates never piles up transforms.
"""

from collections.abc import Iterator
from typing import override

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
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
    ARNOLD_TIMES_MAX,
    SampleArray,
    SamplePlane,
)
from pixelsb.domain.stack import arnold_image
from pixelsb.ui import painting, text
from pixelsb.ui.controls import drain

_THUMB = 96  # a candidate's thumbnail, square
_THUMB_COLUMNS = 3
_SPIN_WIDTH = 104  # wide enough for every digit ±2147483647 can ask for


class _BruteWorker(QThread):
    """The sweep over a parameter cube, one thumbnail per candidate."""

    found = Signal(int, int, int, float, QImage)
    reached = Signal(int)

    def __init__(
        self,
        samples: SampleArray,
        planes: tuple[SamplePlane, ...],
        jobs: list[tuple[int, int, int]],
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self._samples = samples
        self._planes = planes
        self._jobs = jobs
        self.total = len(jobs)
        self.count = 0
        self._stopping = False

    @override
    def run(self) -> None:
        for times, a, b in self._jobs:
            if self._stopping:
                break
            decoded = arnold_image(self._samples, times, a, b)
            rgb = _as_rgb(decoded, self._planes)
            self.found.emit(times, a, b, _smoothness(rgb), _thumbnail(rgb))
            self.count += 1
            self.reached.emit(self.count)

    def stop(self) -> None:
        self._stopping = True


def _as_rgb(samples: SampleArray, planes: tuple[SamplePlane, ...]) -> NDArray[np.uint8]:
    """The picture as HxWx3 bytes at the canvas's own channel scale.

    Three or more channels give their first three to the RGB; fewer mean gray,
    so the one channel repeats — an alpha plane must never dress itself up as a
    color channel, which is what turned gray-plus-alpha thumbnails teal.
    """
    if samples.shape[2] >= 3:
        scaled = [
            samples[:, :, index].astype(np.float64) * (255.0 / planes[index].maximum)
            for index in range(3)
        ]
    else:
        gray = samples[:, :, 0].astype(np.float64) * (255.0 / planes[0].maximum)
        scaled = [gray, gray, gray]
    return np.stack([channel.clip(0.0, 255.0).astype(np.uint8) for channel in scaled], axis=-1)


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
    what the sweep runs on, and a new picture clears the grid. A run is started
    by hand, can be stopped, and seats its candidates smoothest first.
    """

    picked = Signal(int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self._samples: SampleArray | None = None
        self._planes: tuple[SamplePlane, ...] = ()
        self._worker: _BruteWorker | None = None
        self._ranges: tuple[tuple[QSpinBox, QSpinBox], ...] = ()
        self._entries: list[tuple[float, int, int, int, QImage]] = []
        self._source = QLabel()
        self._source.setObjectName("note")
        self._source.setWordWrap(True)
        self._progress = QLabel()
        self._progress.setObjectName("note")
        # One row per parameter, so the page stays narrow enough for the rail's
        # column: the ranges are wide, the page need not be.
        rows = QGridLayout()
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(6)
        for row, (label, low, high, first, last) in enumerate(
            (
                (text.ARNOLD_TIMES, 1, ARNOLD_TIMES_MAX, 1, 5),
                (text.ARNOLD_A, -ARNOLD_PARAM_LIMIT, ARNOLD_PARAM_LIMIT, 1, 5),
                (text.ARNOLD_B, -ARNOLD_PARAM_LIMIT, ARNOLD_PARAM_LIMIT, 1, 5),
            )
        ):
            rows.addWidget(QLabel(label), row, 0)
            pair: list[QSpinBox] = []
            for column, value in enumerate((first, last)):
                box = QSpinBox()
                box.setRange(low, high)
                box.setFixedWidth(_SPIN_WIDTH)
                box.setValue(value)
                pair.append(box)
                rows.addWidget(box, row, 1 + column)
            self._ranges += ((pair[0], pair[1]),)
        self._go = QPushButton(text.ARNOLD_START)
        self._go.setObjectName("ghost")
        self._go.clicked.connect(self._on_start)
        self._halt = QPushButton(text.ARNOLD_STOP)
        self._halt.setObjectName("ghost")
        self._halt.clicked.connect(self._on_stop)
        rows.addWidget(self._go, 3, 0)
        rows.addWidget(self._halt, 3, 1)
        rows.addWidget(self._progress, 3, 2, 1, 2)
        self._grid = QGridLayout()
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        host = QWidget()
        host.setLayout(self._grid)
        holder = QScrollArea()
        holder.setWidgetResizable(True)
        holder.setFrameShape(QScrollArea.Shape.NoFrame)
        holder.setWidget(host)
        holder.setMinimumHeight(_THUMB * 2 + 60)
        note = QLabel(text.ARNOLD_SOURCE)
        note.setObjectName("note")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(note)
        layout.addWidget(self._source)
        layout.addLayout(rows)
        layout.addWidget(holder, 1)
        self._set_running(False)

    def set_source(self, samples: SampleArray | None, planes: tuple[SamplePlane, ...]) -> None:
        """Point the page at the canvas's pixels; a new picture clears the grid."""
        if samples is self._samples and planes == self._planes:
            return
        self._stop_run()
        self._samples = samples
        self._planes = planes
        self._clear_grid()
        square = samples is not None and samples.shape[0] == samples.shape[1]
        self._go.setEnabled(square)
        if samples is None:
            self._source.setText(text.NO_IMAGE)
        elif square:
            self._source.setText(
                text.arnold_source(samples.shape[0], samples.shape[1], samples.shape[2])
            )
        else:
            self._source.setText(text.arnold_not_square(samples.shape[1], samples.shape[0]))

    def shutdown(self) -> None:
        """Stop the sweep before the window goes away, so the thread joins."""
        self._stop_run()

    def _on_start(self) -> None:
        if self._samples is None or self._worker is not None:
            return
        jobs = list(self._jobs())
        if not jobs:
            return
        self._clear_grid()
        self._worker = _BruteWorker(self._samples, self._planes, jobs, self)
        self._worker.found.connect(self._on_found)
        self._worker.reached.connect(self._on_reached)
        self._worker.finished.connect(self._on_done)
        self._set_running(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _jobs(self) -> Iterator[tuple[int, int, int]]:
        (times_from, times_to), (a_from, a_to), (b_from, b_to) = self._ranges
        for times in range(times_from.value(), times_to.value() + 1):
            for a in range(a_from.value(), a_to.value() + 1):
                for b in range(b_from.value(), b_to.value() + 1):
                    yield times, a, b

    def _on_found(self, times: int, a: int, b: int, score: float, image: QImage) -> None:
        self._entries.append((score, times, a, b, image))

    def _on_reached(self, done: int) -> None:
        total = self._worker.total if self._worker is not None else done
        self._progress.setText(text.arnold_progress(done, total))

    def _on_done(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None and worker.count < worker.total:
            self._progress.setText(text.arnold_cancelled(worker.count))
        self._fill_grid()
        self._set_running(False)

    def _fill_grid(self) -> None:
        """The candidates as thumbnails, smoothest first — the heuristic's order.

        The score only decides the seating; the tooltip carries it, and the eye
        still makes the call.
        """
        drain(self._grid)
        for score, times, a, b, image in sorted(self._entries, reverse=True):
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIcon(QPixmap.fromImage(image))
            button.setIconSize(image.size())
            button.setText(text.arnold_caption(times, a, b))
            button.setToolTip(text.arnold_pick_tip(times, a, b, score))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, t=times, a=a, b=b: self.picked.emit(t, a, b)
            )
            count = self._grid.count()
            self._grid.addWidget(button, count // _THUMB_COLUMNS, count % _THUMB_COLUMNS)

    def _clear_grid(self) -> None:
        drain(self._grid)
        self._entries = []
        self._progress.setText("")

    def _set_running(self, running: bool) -> None:
        self._go.setEnabled(not running and self._samples is not None)
        self._halt.setEnabled(running)

    def _stop_run(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()
            worker.wait(2000)

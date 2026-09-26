"""The scan page: the zsteg sweep across channels, bits, and stream order.

The sweep runs on the image's own pixels — the file as zsteg would read it,
whatever the recipe stack is doing — one extraction per candidate, in a worker
thread, with progress as it goes. Every candidate already is a recipe, so a hit
is not a report to retype: clicking it writes its bits selection into the stack
and points the extract panel at the same order.
"""

from typing import override

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain import bytes_text
from pixelsb.domain.classify import StreamClassifier
from pixelsb.domain.detect import detect_patterns
from pixelsb.domain.extract import extract_bytes
from pixelsb.domain.models import LoadedImage, Raster, SampleArray, SamplePlane
from pixelsb.domain.scan import ScanCandidate, ScanHit, scan_candidates
from pixelsb.io.classify import stream_classifier
from pixelsb.ui import text, theme
from pixelsb.ui.controls import section_title
from pixelsb.ui.extract_view import dump_font

_HIT_TINT = 36  # the highlight's alpha over the warning color, for a result row
_PREVIEW_BYTES = 24  # the taste of each stream the preview column shows


class _SweepWorker(QThread):
    """One extraction per candidate, judged where the bytes land."""

    scored = Signal(object)  # a ScanHit
    reached = Signal(int)

    def __init__(
        self, samples: SampleArray, planes: tuple[SamplePlane, ...], parent: QWidget | None
    ) -> None:
        super().__init__(parent)
        self._samples = samples
        self._planes = planes
        self._candidates = scan_candidates(planes)
        self._stopping = False

    @override
    def run(self) -> None:
        classify = stream_classifier()
        for done, candidate in enumerate(self._candidates, start=1):
            if self._stopping:
                break
            self.scored.emit(self._judge(candidate, classify))
            self.reached.emit(done)

    def stop(self) -> None:
        self._stopping = True

    def _judge(self, candidate: ScanCandidate, classify: StreamClassifier) -> ScanHit:
        """The candidate's stream and what it looks like: a type, a flag, or nothing."""
        stream = extract_bytes(
            Raster(samples=self._samples, planes=self._planes, selection=candidate.selection),
            candidate.order,
        )
        if not stream:
            return ScanHit(candidate)
        classification = classify(stream)
        flags = tuple(detection.label for detection in detect_patterns(stream))
        return ScanHit(
            candidate,
            classification.label if classification else "",
            flags,
            bytes_text.preview(stream[:_PREVIEW_BYTES], _PREVIEW_BYTES),
        )


class ScanPanel(QWidget):
    """One sidebar page sweeping the file's own bit streams for recognizable ones.

    The candidate list is fixed by the image's planes; a run is started by hand,
    can be stopped, and leaves its hits in the tree with the noteworthy rows on
    top. A click anywhere on a row applies that candidate.
    """

    apply_requested = Signal(object)  # a ScanCandidate

    def __init__(self) -> None:
        super().__init__()
        self._image: LoadedImage | None = None
        self._worker: _SweepWorker | None = None
        self._hits: list[ScanHit] = []
        self._total = 0
        self._start = QPushButton(text.SWEEP_START)
        self._start.setObjectName("ghost")
        self._start.clicked.connect(self._on_start)
        self._stop = QPushButton(text.SWEEP_STOP)
        self._stop.setObjectName("ghost")
        self._stop.clicked.connect(self._on_stop)
        self._progress = QLabel()
        self._progress.setObjectName("note")
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(self._start)
        controls.addWidget(self._stop)
        controls.addWidget(self._progress, 1)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(text.SWEEP_COLUMNS))
        self._tree.setHeaderLabels(text.SWEEP_COLUMNS)
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.itemClicked.connect(self._on_clicked)
        header = self._tree.header()
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 148), (1, 92), (2, 176)):
            header.resizeSection(column, width)
        note = QLabel(text.SWEEP_NOTE)
        note.setObjectName("note")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(section_title(text.SECTION_SWEEP))
        layout.addWidget(note)
        layout.addLayout(controls)
        layout.addWidget(self._tree, 1)
        self._set_running(False)

    def set_image(self, image: LoadedImage | None) -> None:
        """Point the sweep at an image; a new file stops any run and clears the list."""
        if image is self._image:
            return
        self._halt()
        self._image = image
        self._hits = []
        self._tree.clear()
        self._total = len(scan_candidates(image.planes)) if image is not None else 0
        self._progress.setText(text.sweep_candidates(self._total))
        self._start.setEnabled(image is not None and self._total > 0)

    def shutdown(self) -> None:
        """Stop the sweep before the window goes away, so the thread joins."""
        self._halt()

    def _on_start(self) -> None:
        image = self._image
        if image is None or self._worker is not None:
            return
        self._hits = []
        self._tree.clear()
        self._worker = _SweepWorker(image.samples, image.planes, self)
        self._worker.scored.connect(self._on_scored)
        self._worker.reached.connect(self._on_reached)
        self._worker.finished.connect(self._on_done)
        self._set_running(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _on_scored(self, hit: ScanHit) -> None:
        self._hits.append(hit)

    def _on_reached(self, done: int) -> None:
        self._progress.setText(text.sweep_progress(done, self._total))

    def _on_done(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._populate()
        flagged = sum(1 for hit in self._hits if hit.flags)
        self._progress.setText(text.sweep_done(flagged, self._total))
        self._set_running(False)

    def _populate(self) -> None:
        """The whole run at once, strongest evidence first, original order within.

        A flag-shaped string outranks a bare type guess: the model reads noise
        as ISO images and STL binaries all day on periodic bit streams, while a
        flag is the finding the sweep is for — so only flag rows take the tint.
        """
        self._tree.clear()
        order = sorted(
            enumerate(self._hits),
            key=lambda pair: (
                0 if pair[1].flags else 1 if pair[1].label else 2,
                pair[0],
            ),
        )
        for _index, hit in order:
            item = QTreeWidgetItem(
                (
                    hit.candidate.command,
                    text.sweep_order(hit.candidate.order),
                    hit.preview,
                    text.sweep_result(hit),
                )
            )
            item.setFont(2, dump_font())
            if hit.flags:
                item.setForeground(3, QColor(theme.WARNING))
                tint = QColor(theme.WARNING)
                tint.setAlpha(_HIT_TINT)
                for column in range(4):
                    item.setBackground(column, tint)
            item.setData(0, Qt.ItemDataRole.UserRole, hit.candidate)
            self._tree.addTopLevelItem(item)

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        candidate = item.data(0, Qt.ItemDataRole.UserRole)
        if candidate is not None:
            self.apply_requested.emit(candidate)

    def _set_running(self, running: bool) -> None:
        self._start.setEnabled(not running and self._image is not None and self._total > 0)
        self._stop.setEnabled(running)

    def _halt(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.stop()
            worker.wait(2000)

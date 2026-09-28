"""The scan page: the zsteg sweep across channels, bits, and stream order.

The sweep runs on the current canvas — the pixels as the stack shows them now,
region masks included: only the pixels the extract panel would read go into a
candidate's stream, so a hit is judged against the very stream its recipe
produces. Every candidate already is a recipe, so a click writes its bits
selection into the stack and points the extract panel at the same order.
"""

from functools import partial
from typing import override

from PySide6.QtCore import Qt, Signal
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
from pixelsb.domain.detect import keyword_hits
from pixelsb.domain.extract import StreamSource
from pixelsb.domain.models import LoadedImage, Raster
from pixelsb.domain.scan import ScanCandidate, ScanHit, scan_candidates
from pixelsb.io.classify import stream_classifier
from pixelsb.ui import text, theme
from pixelsb.ui.controls import section_title
from pixelsb.ui.extract_view import dump_font
from pixelsb.ui.worker import Stoppable, shutdown

_HIT_TINT = 36  # the highlight's alpha over the warning color, for a result row
_PREVIEW_BYTES = 24  # the taste of each stream the preview column shows
_RANK_ROLE = Qt.ItemDataRole.UserRole + 1  # where a row keeps its evidence rank


def _rank(hit: ScanHit) -> int:
    """How strong the evidence is: a flag, a bare type guess, or nothing."""
    return 0 if hit.flags else 1 if hit.label else 2


class _SweepWorker(Stoppable):
    """One extraction per candidate, judged where the bytes land.

    The candidates read the canvas's own raster through one shared
    :class:`StreamSource`, so the rows and bit columns every candidate wants are
    gathered once instead of per candidate.
    """

    scored = Signal(object)  # a ScanHit
    reached = Signal(int)

    def __init__(self, raster: Raster, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._source = StreamSource(raster)
        self._candidates = scan_candidates(raster.planes)

    @override
    def run(self) -> None:
        classify = stream_classifier()
        for done, candidate in enumerate(self._candidates, start=1):
            if self._stopping:
                break
            self.scored.emit(self._judge(candidate, classify))
            self.reached.emit(done)

    def _judge(self, candidate: ScanCandidate, classify: StreamClassifier) -> ScanHit:
        """The candidate's stream and what it looks like: a type, a flag, or nothing."""
        stream = self._source.stream(candidate.selection, candidate.order)
        if not stream:
            return ScanHit(candidate)
        classification = classify(stream)
        flags = tuple(hit.label for hit in keyword_hits(stream))
        return ScanHit(
            candidate,
            classification.label if classification else "",
            flags,
            bytes_text.preview(stream[:_PREVIEW_BYTES], _PREVIEW_BYTES),
        )


class ScanPanel(QWidget):
    """One sidebar page sweeping the canvas's bit streams for recognizable ones.

    The candidate list is fixed by the canvas's planes; a run is started by hand,
    can be stopped, and lists its hits as they land, with the noteworthy rows on
    top. A click anywhere on a row applies that candidate.
    """

    apply_requested = Signal(object)  # a ScanCandidate

    def __init__(self) -> None:
        super().__init__()
        self._image: LoadedImage | None = None
        self._raster: Raster | None = None
        self._worker: _SweepWorker | None = None
        self._flagged = 0  # rows of the current run that carry a flag
        self._total = 0
        self._run = QPushButton(text.SWEEP_START)
        self._run.setFixedHeight(theme.CONTROL_HEIGHT)
        self._run.clicked.connect(self._toggle_run)
        self._progress = QLabel()
        self._progress.setObjectName("note")
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(self._run)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(section_title(text.SECTION_SWEEP))
        layout.addLayout(controls)
        layout.addWidget(self._tree, 1)
        self._progress.setText(text.sweep_candidates(self._total))
        self._set_running(False)

    def set_source(self, image: LoadedImage | None, raster: Raster | None) -> None:
        """Point the sweep at the canvas's pixels; only a new image clears the list.

        Clicking a hit writes its recipe into the stack, so the stack moving under
        the page is the ordinary case — the list stays up then, and starting a new
        run sweeps whatever the stack shows now.
        """
        if image is self._image and raster is self._raster:
            return
        if image is not self._image:
            self._halt()
            self._flagged = 0
            self._tree.clear()
        self._image = image
        self._raster = raster
        self._total = len(scan_candidates(raster.planes)) if raster is not None else 0
        if self._worker is None:  # a live run refreshes the label on its own
            self._progress.setText(text.sweep_candidates(self._total))
        self._set_running(self._worker is not None)

    def shutdown(self) -> None:
        """Stop the sweep before the window goes away, so the thread joins."""
        self._halt()

    def _on_start(self) -> None:
        raster = self._raster
        if raster is None or self._worker is not None:
            return
        self._flagged = 0
        self._tree.clear()
        self._worker = _SweepWorker(raster, self)
        self._worker.scored.connect(self._on_scored)
        self._worker.reached.connect(self._on_reached)
        self._worker.finished.connect(partial(self._on_done, self._worker))
        self._set_running(True)
        self._worker.start()

    def _toggle_run(self) -> None:
        """The one button runs the sweep, or stops the run it is sweeping."""
        if self._worker is not None:
            self._worker.stop()
        else:
            self._on_start()

    def _on_scored(self, hit: ScanHit) -> None:
        """One candidate judged: it takes its place in the list right away.

        A flag outranks a bare type guess outranks silence, and rows of equal
        rank keep the order they were tried in — so a finding shows up at the top
        the moment the sweep reaches it, while the run is still going.
        """
        rank = _rank(hit)
        if hit.flags:
            self._flagged += 1
        position = self._tree.topLevelItemCount()
        while position > 0:
            above = self._tree.topLevelItem(position - 1)
            if above is None or int(above.data(0, _RANK_ROLE)) <= rank:
                break
            position -= 1
        self._tree.insertTopLevelItem(position, self._row(hit))

    def _on_reached(self, done: int) -> None:
        self._progress.setText(text.sweep_progress(done, self._total))

    def _on_done(self, worker: _SweepWorker) -> None:
        if worker is not self._worker:
            return  # a run the page already dropped has nothing to report
        self._worker = None
        worker.deleteLater()
        self._progress.setText(text.sweep_done(self._flagged, self._total))
        self._set_running(False)

    def _row(self, hit: ScanHit) -> QTreeWidgetItem:
        """One hit as a row; only the rows that carry a flag take the tint.

        A flag-shaped string outranks a bare type guess: the model reads noise as
        ISO images and STL binaries all day on periodic bit streams, while a flag
        is the finding the sweep is for.
        """
        item = QTreeWidgetItem(
            (
                hit.candidate.command,
                text.sweep_order(hit.candidate.order),
                hit.preview,
                text.sweep_result(hit),
            )
        )
        item.setFont(2, dump_font())
        item.setData(0, Qt.ItemDataRole.UserRole, hit)
        item.setData(0, _RANK_ROLE, _rank(hit))
        if hit.flags:
            item.setForeground(3, QColor(theme.WARNING))
            tint = QColor(theme.WARNING)
            tint.setAlpha(_HIT_TINT)
            for column in range(4):
                item.setBackground(column, tint)
        return item

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        hit = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(hit, ScanHit):
            self.apply_requested.emit(hit.candidate)

    def _set_running(self, running: bool) -> None:
        """The button names the action a click would take now."""
        self._run.setText(text.SWEEP_STOP if running else text.SWEEP_START)
        self._run.setEnabled(self._raster is not None and self._total > 0)

    def _halt(self) -> None:
        worker, self._worker = self._worker, None
        shutdown(worker)

"""The histogram page: every channel's value census, and the chi-square read.

The chart draws the planes the raster carries — the values the stack leaves,
a projection's packing included, a crop already taken away — overlaid in one
log-scaled plot, where the shapes a stego tool leaves are spiky pairs and
missing values rather than a smooth hill. Under it, one chi-square row per
channel: the p of the value-pair test per shown bit, warning-tinted when a bit
reads as written.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from pixelsb.domain import histogram
from pixelsb.domain.models import LoadedImage, Raster, SampleArray, SamplePlane
from pixelsb.ui import text, theme
from pixelsb.ui.controls import drain, section_title
from pixelsb.ui.extract_view import dump_font

_BINS = 256  # display buckets; a 16-bit plane's 65536 counts fold into them
_CHART_HEIGHT = 170
_PAD_LEFT, _PAD_TOP, _PAD_RIGHT, _PAD_BOTTOM = 34.0, 10.0, 10.0, 16.0

_PLANE_COLORS = {
    "R": "#e5484d",
    "G": "#30a46c",
    "B": "#3b82f6",
    "L": "#8b8d98",
    "A": "#9a6bdf",
}
_FALLBACK_COLOR = "#8b8d98"


@dataclass(frozen=True, slots=True)
class _Series:
    """One painted plane's chart line: the log histogram of its shown values."""

    name: str
    color: QColor
    shape: NDArray[np.float64]  # log10(1 + count) per display bin
    top: int  # the plane's range's end, the x axis's candidate end


@dataclass(frozen=True, slots=True)
class _Verdicts:
    """One plane's chi-square row: the p per shown bit, over the stored values."""

    bits: tuple[int, ...]  # the source bits this row's cells read
    tails: tuple[tuple[float, int] | None, ...]  # per shown bit: (χ², df), or None
    p: tuple[float | None, ...]  # per shown bit: the tail's p


def _plane_color(name: str) -> QColor:
    return QColor(_PLANE_COLORS.get(name, _FALLBACK_COLOR))


def _series(counts: NDArray[np.int64], plane: SamplePlane) -> _Series:
    """The chart line of one plane, drawn from the census it shares with the row."""
    return _Series(
        name=plane.name,
        color=_plane_color(plane.name),
        shape=np.log10(_binned(counts[: plane.maximum + 1]) + 1.0),
        top=plane.maximum,
    )


def _verdicts(counts: NDArray[np.int64], plane: SamplePlane) -> _Verdicts:
    """The chi-square row of one plane, from the census it shares with the chart."""
    bits = histogram.shown_bits(plane)
    tails: list[tuple[float, int] | None] = []
    p: list[float | None] = []
    for bit in bits:
        verdict = histogram.pair_chi_square(counts, bit)
        tails.append(verdict)
        p.append(None if verdict is None else histogram.chi2_tail(*verdict))
    return _Verdicts(bits=bits, tails=tuple(tails), p=tuple(p))


def _binned(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """The histogram in display buckets; a narrow plane keeps its own bins."""
    if counts.size <= _BINS:
        return counts.astype(np.float64)
    return counts.reshape(_BINS, -1).sum(axis=1).astype(np.float64)


class _Chart(QWidget):
    """The value histograms, one overlaid line per plane, on a log count axis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._planes: tuple[_Series, ...] = ()
        self.setMinimumHeight(_CHART_HEIGHT)

    def set_planes(self, planes: tuple[_Series, ...]) -> None:
        self._planes = planes
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor(theme.FIELD))
        area = QRectF(
            _PAD_LEFT,
            _PAD_TOP,
            max(self.width() - _PAD_LEFT - _PAD_RIGHT, 1.0),
            max(self.height() - _PAD_TOP - _PAD_BOTTOM, 1.0),
        )
        peak = max((float(data.shape.max()) for data in self._planes), default=0.0)
        self._frame(painter, area, peak)
        if peak <= 0.0:
            return
        for data in self._planes:
            self._line(painter, area, data, peak)
        self._legend(painter, area)

    def _frame(self, painter: QPainter, area: QRectF, peak: float) -> None:
        """The hairline box, the decade gridlines, and the axis labels."""
        painter.setPen(QPen(QColor(theme.HAIRLINE), 1.0))
        painter.drawRect(area)
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        muted = QPen(QColor(theme.TEXT_MUTED), 1.0)
        for decade in range(int(np.ceil(peak)) + 1):
            y = area.bottom() - area.height() * (decade / peak if peak > 0 else 0.0)
            painter.setPen(muted)
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.drawText(
                QRectF(0.0, y - 6.0, _PAD_LEFT - 4.0, 12.0),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                "1" if decade == 0 else f"1e{decade}",
            )
        top = max((data.top for data in self._planes), default=0xFF)
        painter.setPen(muted)
        for tick in range(5):
            x = area.left() + area.width() * tick / 4
            painter.drawText(
                QRectF(x - 20.0, area.bottom() + 2.0, 40.0, 12.0),
                int(Qt.AlignmentFlag.AlignCenter),
                f"{round(top * tick / 4):X}",
            )

    def _line(self, painter: QPainter, area: QRectF, data: _Series, peak: float) -> None:
        """One plane's histogram as a polyline over the display bins.

        The scale is the chart's one peak, not the plane's own: a channel's
        ceiling must not pass for the picture's.
        """
        pen = QPen(data.color, 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        shape = data.shape
        step = area.width() / shape.size
        xs = area.left() + step * (np.arange(shape.size) + 0.5)
        ys = area.bottom() - area.height() * shape / peak
        painter.drawPolyline(
            QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])
        )

    def _legend(self, painter: QPainter, area: QRectF) -> None:
        """The planes' names in their own colors, tucked into the chart's corner."""
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        x = area.right()
        for data in reversed(self._planes):
            width = metrics.horizontalAdvance(data.name)
            x -= width
            painter.setPen(QPen(data.color, 1.0))
            painter.drawText(
                QRectF(x, area.top() + 2.0, width, 14.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                data.name,
            )
            x -= 6.0


class HistogramPanel(QWidget):
    """One sidebar page: the canvas's value histograms and the pair test."""

    def __init__(self) -> None:
        super().__init__()
        self._image: LoadedImage | None = None
        self._raster: Raster | None = None
        self._samples: SampleArray | None = None  # identity: what the numbers were read from
        self._live: NDArray[np.bool_] | None = None  # and which pixels they were read from
        self._signature: tuple[tuple[str, int, tuple[int, ...]], ...] = ()
        self._cells: dict[tuple[str, int], QLabel] = {}
        self._chart = _Chart()
        self._chart.setToolTip(text.HISTOGRAM_TIP)
        self._empty = QLabel(text.NO_IMAGE)
        self._empty.setObjectName("note")
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(section_title(text.SECTION_HISTOGRAM))
        layout.addWidget(self._chart)
        layout.addWidget(section_title(text.SECTION_CHI2))
        self._grid_host.setToolTip(text.CHI2_TIP)
        layout.addWidget(self._grid_host)
        layout.addWidget(self._empty)
        layout.addStretch(1)

    def set_source(self, image: LoadedImage | None, raster: Raster | None) -> None:
        """Point the page at the canvas's picture; it recomputes when the view moves.

        The counts and the pair test read the raster's samples and the region's
        kept pixels — the population about to be extracted — so a change in
        either recounts.
        """
        samples = None if raster is None else raster.samples
        live = None if raster is None else raster.live
        if image is self._image and samples is self._samples and live is self._live:
            self._raster = raster
            return
        self._image = image
        self._raster = raster
        self._samples = samples
        self._live = live
        planes: tuple[SamplePlane, ...] = ()
        signature: tuple[tuple[str, int, tuple[int, ...]], ...] = ()
        series: tuple[_Series, ...] = ()
        rows: tuple[tuple[SamplePlane, _Verdicts], ...] = ()
        if raster is not None and raster.width > 0 and raster.height > 0:
            planes = raster.planes
            signature = tuple(
                (plane.name, plane.bit_depth, histogram.shown_bits(plane)) for plane in planes
            )
            built: list[_Series] = []
            judged: list[tuple[SamplePlane, _Verdicts]] = []
            for plane in planes:
                counted, _top = histogram.plane_counts(raster, plane)  # one census feeds both
                built.append(_series(counted, plane))
                judged.append((plane, _verdicts(counted, plane)))
            series, rows = tuple(built), tuple(judged)
        if signature != self._signature:
            self._signature = signature
            self._build_grid(raster, planes)
        self._chart.set_planes(series)
        self._update_cells(rows)
        self._empty.setVisible(raster is None or not series)

    def _build_grid(self, raster: Raster | None, planes: tuple[SamplePlane, ...]) -> None:
        """One row per plane; the columns sit at the bits the rows actually read."""
        drain(self._grid)
        self._cells = {}
        shown = [histogram.shown_bits(plane) for plane in planes] if raster is not None else []
        width = max((bits[-1] for bits in shown if bits), default=-1)
        for bit in range(width + 1):
            head = QLabel(str(bit))
            head.setObjectName("note")
            self._grid.addWidget(head, 0, 1 + bit)
        for row, (plane, bits) in enumerate(zip(planes, shown, strict=True), start=1):
            name = QLabel(plane.name)
            name.setObjectName("note")
            self._grid.addWidget(name, row, 0)
            for bit in bits:
                cell = QLabel()
                cell.setFont(dump_font())
                self._cells[plane.name, bit] = cell
                self._grid.addWidget(cell, row, 1 + bit)
        self._grid_host.setVisible(bool(planes))

    def _update_cells(self, rows: tuple[tuple[SamplePlane, _Verdicts], ...]) -> None:
        """Fill the grid: the p per shown bit, tinted when the bit reads as written."""
        for plane, row in rows:
            for bit, verdict, p in zip(row.bits, row.tails, row.p, strict=True):
                cell = self._cells.get((plane.name, bit))
                if cell is None:
                    continue
                cell.setText("—" if p is None or verdict is None else f"{p:.2f}")
                cell.setToolTip(
                    ""
                    if verdict is None
                    else text.chi2_detail(plane.name, bit, verdict[0], verdict[1])
                )
                hot = p is not None and p >= 0.95
                if cell.property("hot") != hot:
                    cell.setProperty("hot", hot)
                    theme.repolish(cell)

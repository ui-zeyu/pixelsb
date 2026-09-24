"""Checkbox grid of channel bits. Bit 0 is the rightmost column.

Row and column checkboxes toggle whole channels and bit columns; they show a
partial state when only part of the group is selected.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QSizePolicy, QWidget

from pixelsb.domain.models import BitChoice, SamplePlane
from pixelsb.ui import text

CELL_SIZE = 20
HEADER_OBJECT = "headerBox"


class BitMatrix(QWidget):
    bit_clicked = Signal(str, int, bool)
    channel_toggle = Signal(str, bool)
    column_toggle = Signal(int, bool)

    def __init__(self) -> None:
        super().__init__()
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(3)
        self._layout.setVerticalSpacing(3)
        self._boxes: dict[tuple[str, int], QCheckBox] = {}
        self._row_boxes: dict[str, QCheckBox] = {}
        self._col_boxes: dict[int, QCheckBox] = {}
        self._signature: tuple[tuple[str, int], ...] = ()
        self.setToolTip(text.MATRIX_TIP)

    def set_layer(
        self,
        planes: tuple[SamplePlane, ...],
        chosen: frozenset[BitChoice] | None,
    ) -> None:
        """Sync the checkboxes. ``chosen is None`` means every bit is checked."""
        signature = tuple((plane.name, plane.bit_depth) for plane in planes)
        if signature != self._signature:
            self._rebuild(planes)
            self._signature = signature
        if not planes:
            return
        for (plane, bit), box in self._boxes.items():
            checked = True if chosen is None else BitChoice(plane, bit) in chosen
            _sync_box(box, checked)
        for name, box in self._row_boxes.items():
            plane = next(candidate for candidate in planes if candidate.name == name)
            members = _members(plane, plane.bit_depth, chosen)
            _sync_group(box, members)
        for bit, box in self._col_boxes.items():
            members = [
                True if chosen is None else BitChoice(plane.name, bit) in chosen
                for plane in planes
                if bit < plane.bit_depth
            ]
            _sync_group(box, members)

    def _rebuild(self, planes: tuple[SamplePlane, ...]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._boxes.clear()
        self._row_boxes.clear()
        self._col_boxes.clear()
        if not planes:
            return
        max_depth = max(plane.bit_depth for plane in planes)
        for bit in range(max_depth - 1, -1, -1):
            column = (max_depth - 1 - bit) + 1
            box = _header_box(str(bit))
            box.setToolTip(f"{text.COLUMN_TIP} bit {bit}")
            box.clicked.connect(
                lambda _checked=False, bit=bit: self.column_toggle.emit(bit, _checked)
            )
            self._layout.addWidget(box, 0, column)
            self._col_boxes[bit] = box
        for row, plane in enumerate(planes, start=1):
            row_box = _header_box(plane.name)
            row_box.setToolTip(text.CHANNEL_TIP)
            row_box.clicked.connect(
                lambda _checked=False, name=plane.name: self.channel_toggle.emit(name, _checked)
            )
            self._layout.addWidget(row_box, row, 0)
            self._row_boxes[plane.name] = row_box
            for bit in range(plane.bit_depth):
                column = (max_depth - 1 - bit) + 1
                box = QCheckBox()
                box.setFixedSize(CELL_SIZE, CELL_SIZE)
                box.setToolTip(f"{plane.name} bit {bit}")
                box.clicked.connect(
                    lambda _checked=False, name=plane.name, bit=bit: self._emit(name, bit)
                )
                self._layout.addWidget(box, row, column)
                self._boxes[(plane.name, bit)] = box
        # Spread the bit columns over whatever width the panel gives the grid;
        # the channel-label column keeps its own width.
        self._layout.setColumnStretch(0, 0)
        for column in range(1, max_depth + 1):
            self._layout.setColumnStretch(column, 1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Swallow clicks on empty grid cells so they do not reach the parent.
        if self.childAt(event.position().toPoint()) is None:
            event.accept()
            return
        super().mousePressEvent(event)

    def _emit(self, plane: str, bit: int) -> None:
        modifiers = QApplication.keyboardModifiers()
        exclusive = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
                | Qt.KeyboardModifier.ShiftModifier
            )
        )
        self.bit_clicked.emit(plane, bit, exclusive)


def _members(plane: SamplePlane, count: int, chosen: frozenset[BitChoice] | None) -> list[bool]:
    if chosen is None:
        return [True] * count
    return [BitChoice(plane.name, bit) in chosen for bit in range(count)]


def _header_box(label: str) -> QCheckBox:
    box = QCheckBox(label)
    box.setObjectName(HEADER_OBJECT)
    # Fixed, so the label column stays narrow while the bit columns spread.
    box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return box


def _sync_box(box: QCheckBox, checked: bool) -> None:
    state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    if box.checkState() != state:
        box.setCheckState(state)


def _sync_group(box: QCheckBox, members: list[bool]) -> None:
    if all(members):
        state = Qt.CheckState.Checked
    elif any(members):
        state = Qt.CheckState.PartiallyChecked
    else:
        state = Qt.CheckState.Unchecked
    if box.checkState() != state:
        box.setCheckState(state)

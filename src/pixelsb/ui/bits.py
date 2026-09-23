"""Checkbox grid of channel bits. Bit 0 is the rightmost column.

Left-click toggles a bit of the canvas layer. Right-click toggles a bit of the
number layer; right-click on a channel letter toggles the whole channel.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QLabel, QWidget

from pixelsb.domain.models import BitChoice, SamplePlane, ViewerState
from pixelsb.domain.selection import effective_selection
from pixelsb.ui import text


class BitMatrix(QWidget):
    bit_clicked = Signal(str, int, bool)
    bit_right_clicked = Signal(str, int)
    channel_right_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(2)
        self._layout.setVerticalSpacing(2)
        self._boxes: dict[tuple[str, int], QCheckBox] = {}
        self._cells: dict[QWidget, tuple[str, int]] = {}
        self._letters: dict[QWidget, str] = {}
        self._signature: tuple[tuple[str, int], ...] = ()
        self.setToolTip(text.MATRIX_TIP)

    def set_state(self, state: ViewerState) -> None:
        image = state.image
        planes = () if image is None else image.planes
        signature = tuple((plane.name, plane.bit_depth) for plane in planes)
        if signature != self._signature:
            self._rebuild(planes)
            self._signature = signature
        chosen = None if image is None else effective_selection(image, state.selection)
        for (plane, bit), box in self._boxes.items():
            checked = chosen is not None and BitChoice(plane, bit) in chosen
            if box.isChecked() != checked:
                box.blockSignals(True)
                box.setChecked(checked)
                box.blockSignals(False)

    def _rebuild(self, planes: tuple[SamplePlane, ...]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._boxes.clear()
        self._cells.clear()
        self._letters.clear()
        if not planes:
            return
        max_depth = max(plane.bit_depth for plane in planes)
        for column, bit in enumerate(range(max_depth - 1, -1, -1)):
            header = QLabel(str(bit))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._layout.addWidget(header, 0, column + 1)
        for row, plane in enumerate(planes, start=1):
            name = QLabel(plane.name)
            self._layout.addWidget(name, row, 0)
            self._letters[name] = plane.name
            for bit in range(plane.bit_depth):
                column = (max_depth - 1 - bit) + 1
                box = QCheckBox()
                box.setFixedSize(22, 22)
                box.setToolTip(f"{plane.name} bit {bit}")
                box.clicked.connect(
                    lambda _checked=False, name=plane.name, bit=bit: self._emit(name, bit)
                )
                self._layout.addWidget(box, row, column)
                self._boxes[(plane.name, bit)] = box
                self._cells[box] = (plane.name, bit)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        child = self.childAt(event.pos())
        cell = self._cells.get(child)
        if cell is not None:
            self.bit_right_clicked.emit(cell[0], cell[1])
            event.accept()
            return
        letter = self._letters.get(child)
        if letter is not None:
            self.channel_right_clicked.emit(letter)
            event.accept()
            return
        event.ignore()

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

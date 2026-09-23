"""Checkbox grid of channel bits. Bit 0 is the rightmost column.

One grid drives the canvas layer, another drives the number layer. Clicking a
checkbox toggles one bit; clicking a channel letter toggles the whole channel.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QWidget

from pixelsb.domain.models import BitChoice, SamplePlane
from pixelsb.ui import text


class BitMatrix(QWidget):
    bit_clicked = Signal(str, int, bool)
    channel_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(2)
        self._layout.setVerticalSpacing(2)
        self._boxes: dict[tuple[str, int], QCheckBox] = {}
        self._letters: dict[QWidget, str] = {}
        self._signature: tuple[tuple[str, int], ...] = ()
        self.setToolTip(text.MATRIX_TIP)

    def set_layer(
        self, planes: tuple[SamplePlane, ...], chosen: frozenset[BitChoice] | None
    ) -> None:
        """Sync the checkboxes. ``chosen is None`` means every bit is checked."""
        signature = tuple((plane.name, plane.bit_depth) for plane in planes)
        if signature != self._signature:
            self._rebuild(planes)
            self._signature = signature
        for (plane, bit), box in self._boxes.items():
            checked = True if chosen is None else BitChoice(plane, bit) in chosen
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
        self._letters.clear()
        if not planes:
            return
        max_depth = max(plane.bit_depth for plane in planes)
        for column, bit in enumerate(range(max_depth - 1, -1, -1)):
            header = QLabel(str(bit))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._layout.addWidget(header, 0, column + 1)
        for row, plane in enumerate(planes, start=1):
            letter = QLabel(plane.name)
            letter.setToolTip(text.CHANNEL_TIP)
            self._layout.addWidget(letter, row, 0)
            self._letters[letter] = plane.name
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            name = self._letters.get(self.childAt(event.position().toPoint()))
            if name is not None:
                self.channel_clicked.emit(name)
                event.accept()
                return
        super().mousePressEvent(event)

    def _emit(self, plane: str, bit: int) -> None:
        from PySide6.QtWidgets import QApplication

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

"""Bit matrix and the cursor / anchor readout."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from pixelsb.domain.models import ViewerState
from pixelsb.ui import text
from pixelsb.ui.bits import BitMatrix
from pixelsb.ui.text import readout_text


class Inspector(QWidget):
    bit_clicked = Signal(str, int, bool)
    original_requested = Signal()
    only_bit_requested = Signal()
    lsbs_requested = Signal()
    clear_bits_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._state = ViewerState()
        self._matrix = BitMatrix()
        self._matrix.bit_clicked.connect(self.bit_clicked.emit)
        self._detail = QLabel(text.NO_IMAGE)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail.setWordWrap(True)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setFamilies(["Menlo", "monospace"])
        self._detail.setFont(font)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for label, signal in (
            (text.ORIGINAL, self.original_requested),
            (text.ONLY_BIT, self.only_bit_requested),
            (text.ALL_LSB, self.lsbs_requested),
            (text.CLEAR_BITS, self.clear_bits_requested),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, signal=signal: signal.emit())
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(buttons)
        layout.addWidget(self._matrix)
        layout.addWidget(self._detail, 1)
        self.setMinimumWidth(280)

    def set_state(self, state: ViewerState) -> None:
        self._state = state
        self._matrix.set_state(state)
        self._detail.setText(readout_text(state))

    def detail_text(self) -> str:
        return readout_text(self._state)

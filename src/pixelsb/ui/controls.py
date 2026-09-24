"""Small controls the panels build from."""

from dataclasses import dataclass

from PySide6.QtCore import SignalInstance
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

from pixelsb.ui import theme

_SEGMENT_NAMES = ("segmentLeft", "segmentRight")


@dataclass(frozen=True, slots=True)
class Toggle:
    """One button of a segmented pair: the value it stands for, label, and tip."""

    value: str
    label: str
    tip: str


class TogglePair(QWidget):
    """Two exclusive buttons that report which one of the pair was clicked.

    They sit edge to edge, so the corners the stylesheet cuts make them read as
    one control. The first starts checked; the state decides everything after.
    """

    def __init__(
        self,
        parent: QWidget,
        toggles: tuple[Toggle, Toggle],
        signal: SignalInstance,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QPushButton] = {}
        for index, toggle in enumerate(toggles):
            button = QPushButton(toggle.label)
            button.setObjectName(_SEGMENT_NAMES[index])
            button.setCheckable(True)
            button.setToolTip(toggle.tip)
            button.setFixedHeight(theme.CONTROL_HEIGHT)
            button.clicked.connect(lambda _checked=False, value=toggle.value: signal.emit(value))
            button.setChecked(index == 0)
            self._group.addButton(button)
            self._buttons[toggle.value] = button
            layout.addWidget(button)

    def set_value(self, value: str) -> None:
        """Check the button standing for ``value`` without reporting it back."""
        for candidate, button in self._buttons.items():
            button.setChecked(candidate == value)

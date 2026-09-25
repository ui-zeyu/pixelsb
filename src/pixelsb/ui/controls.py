"""Small controls the panels build from."""

from dataclasses import dataclass

from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QWidget,
)

from pixelsb.ui import theme

_SEGMENT_NAMES = ("segmentLeft", "segmentRight")

# Both Enter keys: the main one and the keypad's, wherever one commits.
RETURN_KEYS = frozenset({Qt.Key.Key_Return, Qt.Key.Key_Enter})


def drain(layout: QLayout) -> None:
    """Empty ``layout`` immediately.

    The widget leaves its parent before the deferred deletion, so nothing stale
    answers a later search of the widget tree; it is hidden on the way out, so
    the orphan cannot stand in for a window while it waits to be destroyed.
    """
    while (item := layout.takeAt(0)) is not None:
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        inner = item.layout()
        if inner is not None:
            drain(inner)


def section_title(label: str) -> QLabel:
    """The small muted caption above a section of a panel."""
    title = QLabel(label)
    title.setObjectName("sectionTitle")
    return title


def hairline() -> QFrame:
    """The one-pixel rule between a panel's sections."""
    line = QFrame()
    line.setObjectName("hairline")
    line.setFixedHeight(1)
    return line


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

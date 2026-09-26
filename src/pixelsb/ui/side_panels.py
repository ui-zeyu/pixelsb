"""The right sidebar, Pixelmator-style: a slim rail of buttons deciding which
page the panel shows."""

from enum import StrEnum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_BUTTON_SIZE = 30  # a rail button, square like the strip it sits in
_RAIL_MARGIN = 6


class Panel(StrEnum):
    """The pages the rail switches between."""

    EXTRACT = "extract"
    INFO = "info"
    SCAN = "scan"
    ARNOLD = "arnold"


class SidePanels(QWidget):
    """The button rail beside a stack of pages; exactly one page is up.

    Pages carry the content, this class only carries the mechanism: ``add``
    pairs a page with its rail button, and the checked button decides the page.
    The page area scrolls as one, so the rail stays pinned to the edge while a
    tall page scrolls beside it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._keys: list[Panel] = []
        self._buttons: dict[Panel, QToolButton] = {}
        self._stack = QStackedWidget()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self._stack.setCurrentIndex)
        body = QScrollArea()
        body.setObjectName("panelArea")
        body.setWidgetResizable(True)
        body.setFrameShape(QFrame.Shape.NoFrame)
        body.setWidget(self._stack)
        rail = QFrame()
        rail.setObjectName("sideRail")
        self._rail_layout = QVBoxLayout(rail)
        self._rail_layout.setContentsMargins(_RAIL_MARGIN, 8, _RAIL_MARGIN, 8)
        self._rail_layout.setSpacing(4)
        self._rail_layout.addStretch(1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(body, 1)
        layout.addWidget(rail)

    def add(self, key: Panel, glyph: str, tip: str, page: QWidget) -> None:
        """Register a page and its rail button, in rail order; the first wins."""
        index = self._stack.addWidget(page)
        self._keys.append(key)
        button = QToolButton()
        button.setObjectName("railButton")
        button.setText(glyph)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setChecked(index == 0)
        button.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self._group.addButton(button, index)
        self._buttons[key] = button
        self._rail_layout.insertWidget(index, button, 0, Qt.AlignmentFlag.AlignHCenter)

    def set_current(self, key: Panel) -> None:
        """Raise the page for ``key``, as if its rail button had been clicked."""
        button = self._buttons[key]
        button.setChecked(True)
        self._stack.setCurrentIndex(self._group.id(button))

    @property
    def current(self) -> Panel:
        return self._keys[self._stack.currentIndex()]

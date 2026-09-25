"""The layer panel: the stack of masks over the picture, as a list.

The list reads top down, so the last mask applied is the first row, and the image
itself is the bottom row — it cannot be moved, removed, or switched off. A row
says what its mask holds; where a mask is edited is a sentence on the note line:
the value masks and the region mask in the filter box's command line, the bits in
the extract panel's grid.
"""

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixelsb.domain.models import (
    BitsMask,
    Layer,
    Mask,
    Raster,
    RegionMask,
    SamplePlane,
    ThresholdMask,
    ViewerState,
    XorMask,
)
from pixelsb.domain.transitions import bits_shadowed
from pixelsb.ui import text, theme
from pixelsb.ui.controls import drain, hairline, section_title

_CHECK_COLUMN = 21  # the switch column, so the image's own row lines up with the rest
_PANEL_MIN_WIDTH = 240  # what the layer list needs to read a row


class LayerPanel(QWidget):
    """The left sidebar: the mask stack, with a note about the selected mask."""

    add_requested = Signal(object)  # a Mask, added on top of the stack
    remove_requested = Signal(int)  # the layer's place in the stack
    move_requested = Signal(int, int)  # that place, and which way (-1 or +1)
    enabled_requested = Signal(int, bool)
    selection_changed = Signal()  # the layer whose note is on show

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("layerPanel")
        self.setMinimumWidth(_PANEL_MIN_WIDTH)
        self._state = ViewerState()
        self._failures: dict[int, str] = {}
        self._selected: int | None = None
        self._count = 0
        self._rows: dict[int | None, _LayerRow] = {}
        self._signature: tuple[object, ...] | None = None
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        self._note = _label(self, "", "note")
        self._note.setWordWrap(True)
        self._note.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(self._header())
        layout.addWidget(self._card())
        layout.addSpacing(6)
        layout.addWidget(hairline())
        layout.addSpacing(6)
        layout.addWidget(self._note, 0, Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)

    @property
    def selected(self) -> int | None:
        """The layer whose note is on show, as its place in the stack."""
        return self._selected

    def set_state(self, state: ViewerState, raster: Raster | None = None) -> None:
        self._state = state
        self._failures = {} if raster is None else {f.index: f.message for f in raster.failures}
        self._keep_selection()
        self._sync_rows()
        self._sync_editor()
        self._mark_selected()

    # --- the list ----------------------------------------------------------

    def _header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(section_title(text.SECTION_LAYERS))
        header.addStretch(1)
        self._add_button = QToolButton()
        self._add_button.setObjectName("ghostButton")
        self._add_button.setText(text.LAYER_ADD)
        self._add_button.setToolTip(text.LAYER_ADD_TIP)
        self._add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add_button.setMenu(QMenu(self))
        self._add_button.menu().setToolTipsVisible(True)
        self._add_button.menu().aboutToShow.connect(self._fill_menu)
        self._up = _flat_button(text.LAYER_UP, text.LAYER_UP_TIP, lambda: self._move(1))
        self._down = _flat_button(text.LAYER_DOWN, text.LAYER_DOWN_TIP, lambda: self._move(-1))
        self._remove = _flat_button(text.LAYER_REMOVE, text.LAYER_REMOVE_TIP, self._remove_selected)
        for button in (self._add_button, self._up, self._down, self._remove):
            button.setFixedHeight(theme.CONTROL_HEIGHT)
            header.addWidget(button)
        return header

    def _fill_menu(self) -> None:
        """Build the add list: only the masks that need no parameter belong here."""
        menu = self._add_button.menu()
        menu.clear()
        if not self._planes():
            return
        for mask in text.mask_menu():
            info = text.mask_info(mask)
            action = menu.addAction(info.label)
            action.setToolTip(info.tip)
            action.triggered.connect(
                lambda _checked=False, mask=mask: self.add_requested.emit(mask)
            )

    def _card(self) -> QFrame:
        card = _Card(self)
        self._card_frame = card
        card.setObjectName("card")
        card.clicked.connect(lambda: self._select(None))
        body = QVBoxLayout(card)
        body.setContentsMargins(6, 6, 6, 6)
        body.setSpacing(1)
        body.addLayout(self._rows_layout)
        return card

    def _planes(self) -> tuple[SamplePlane, ...]:
        return () if self._state.image is None else self._state.image.planes

    def _keep_selection(self) -> None:
        """Hold the selection on a layer that survives; a new layer takes it.

        Adding a mask is nearly always an intent to edit it, so the newest layer
        becomes the selected one — including a mask the filter box has just brought
        into being. The image's own row is a selection like any other: once it is
        picked it stays picked, until some mask is added on top.
        """
        count = len(self._state.layers)
        if count > self._count or (self._selected is not None and self._selected >= count):
            self._selected = count - 1 if count else None
        self._count = count

    def _sync_rows(self) -> None:
        # The image belongs in the key: the base row names the file, and every
        # row's parameters are written in the planes of the image they sit on.
        signature = (
            self._state.image,
            self._state.layers,
            tuple(sorted(self._failures.items())),
        )
        if signature != self._signature:
            self._signature = signature
            self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        drain(self._rows_layout)
        self._rows = {}
        layers = self._state.layers
        for index in reversed(range(len(layers))):
            layer = layers[index]
            row = _LayerRow(
                self._card_frame,
                text.mask_info(layer.mask).label,
                self._row_detail(index, layer),
                failed=index in self._failures,
            )
            row.set_enabled(layer.enabled)
            row.clicked.connect(lambda index=index: self._select(index))
            row.toggled.connect(lambda on, index=index: self.enabled_requested.emit(index, on))
            self._rows_layout.addWidget(row)
            self._rows[index] = row
        base = _LayerRow(self._card_frame, text.LAYER_BASE, self._base_detail(), switch=False)
        base.clicked.connect(lambda: self._select(None))
        self._rows_layout.addWidget(base)
        self._rows[None] = base

    def _row_detail(self, index: int, layer: Layer) -> str:
        """One mask's parameters, clipped by the label's own width."""
        detail = text.mask_detail(layer.mask, self._planes())
        return f"{text.WARNING_MARK} {detail}" if index in self._failures else detail

    def _base_detail(self) -> str:
        image = self._state.image
        return "" if image is None else text.file_facts(image)

    def _select(self, index: int | None) -> None:
        self._selected = index
        self._mark_selected()
        self._sync_editor()
        self.selection_changed.emit()

    def _mark_selected(self) -> None:
        for index, row in self._rows.items():
            row.set_selected(index == self._selected)
        editing = self._selected is not None
        for button in (self._up, self._down, self._remove):
            button.setEnabled(editing)

    def _move(self, step: int) -> None:
        """Ask for the move, and keep the selection on the mask that moves.

        The list is rebuilt from the new stack, so the selection has to travel with
        the mask: otherwise the next press of the same button moves it back, and
        the highlight ends up on a mask the user never picked.
        """
        index = self._selected
        if index is None:
            return
        target = index + step
        if not 0 <= target < len(self._state.layers):
            return  # the mask is already at that end
        self._selected = target
        self.move_requested.emit(index, step)

    def _remove_selected(self) -> None:
        if self._selected is not None:
            self.remove_requested.emit(self._selected)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """The blank space around the list deselects, like the image's own row."""
        event.accept()
        self._select(None)

    # --- the note ----------------------------------------------------------

    def _selected_mask(self) -> Mask | None:
        index = self._selected
        if index is None or not 0 <= index < len(self._state.layers):
            return None
        return self._state.layers[index].mask

    def _sync_editor(self) -> None:
        failure = self._failure()
        self._note.setText(failure or self._note_text(self._selected_mask()))
        _restyle(self._note, "warn", bool(failure))

    def _note_text(self, mask: Mask | None) -> str:
        """What the note says: where a mask is edited, or what it does."""
        if bits_shadowed(self._state.layers, self._selected):
            return text.MASK_BITS_SHADOWED
        if mask is None:
            return text.LAYER_BASE_NOTE
        if isinstance(mask, RegionMask):
            return text.LAYER_REGION_NOTE
        if isinstance(mask, BitsMask):
            return text.LAYER_BITS_NOTE
        if isinstance(mask, (ThresholdMask, XorMask)):
            return text.LAYER_COMMAND_NOTE
        return text.mask_info(mask).tip

    def _failure(self) -> str:
        return "" if self._selected is None else self._failures.get(self._selected, "")


class _Card(QFrame):
    """The list's frame; its blank space deselects, like the image's own row."""

    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        event.accept()  # a row's own press stops on the row and never arrives here
        self.clicked.emit()


class _LayerRow(QFrame):
    """One row of the list: a switch, a name, and a one-line summary."""

    clicked = Signal()
    toggled = Signal(bool)

    def __init__(
        self,
        parent: QWidget,
        name: str,
        detail: str,
        *,
        switch: bool = True,
        failed: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("layerRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 6, 2)
        layout.setSpacing(6)
        self._check: QCheckBox | None = None
        if switch:
            self._check = QCheckBox(self)
            self._check.setToolTip(text.LAYER_ENABLED_TIP)
            self._check.toggled.connect(self.toggled.emit)
            # The switch is part of the row, so flicking it picks the row as well;
            # the box takes the press itself, so the row never sees this click.
            self._check.clicked.connect(lambda _checked=False: self.clicked.emit())
            widget: QWidget = self._check
        else:
            # The image's own row keeps the column, so the names line up.
            widget = QWidget(self)
        widget.setFixedWidth(_CHECK_COLUMN)
        self._name = _label(self, name, "layerName")
        self._detail = _label(self, detail, "detailWarn" if failed else "layerDetail")
        # Whatever the panel width leaves is enough: a long expression clips rather
        # than wrapping, which would make its row taller than the rest.
        self._detail.setWordWrap(False)
        self._detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(widget)
        layout.addWidget(self._name)
        layout.addWidget(self._detail, 1)

    @property
    def switch(self) -> QCheckBox:
        """The row's own checkbox; the image's own row has none."""
        if self._check is None:
            raise RuntimeError("the image's own row carries no switch")
        return self._check

    def set_enabled(self, on: bool) -> None:
        if self._check is not None:
            self._check.blockSignals(True)
            self._check.setChecked(on)
            self._check.blockSignals(False)
        _restyle(self, "off", not on)
        for label in (self._name, self._detail):
            _restyle(label, "off", not on)

    def set_selected(self, on: bool) -> None:
        _restyle(self, "selected", on)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Pick this row, and stop the press here.

        A widget that leaves a press unhandled passes it on to its parent, and both
        the card and the panel behind it deselect — which would undo the pick as
        soon as it was made. Accepting the event keeps it on the row.
        """
        event.accept()
        self.clicked.emit()


def _flat_button(label: str, tip: str, slot: Callable[[], None]) -> QToolButton:
    button = QToolButton()
    button.setObjectName("ghostButton")
    button.setText(label)
    button.setToolTip(tip)
    button.clicked.connect(lambda _checked=False: slot())
    return button


def _label(parent: QWidget, content: str, name: str) -> QLabel:
    label = QLabel(content, parent)
    label.setObjectName(name)
    return label


def _restyle(widget: QWidget, name: str, on: bool) -> None:
    """Set a style flag and make the stylesheet see it."""
    if widget.property(name) == on:
        return
    widget.setProperty(name, on)
    theme.repolish(widget)

"""What a key press means when no text input holds the keyboard.

The mapping is pure: a key, its modifiers, the text it carries, and whether the
press is an auto-repeat, in; one command out — or ``None`` when the key means
nothing. The window executes the command: the gallery page stands in for the
canvas under the arrow keys, and everything else is one transition apiece.
"""

from dataclasses import dataclass

from PySide6.QtCore import Qt

from pixelsb.ui.side_panels import Panel

# Pixels an arrow covers when Shift is down, instead of one.
NUDGE_STEP = 8

# Chords the canvas never combines with: ⌘, Ctrl and Alt belong to the menu
# level, so a chord is either one of the commands here or somebody else's.
_COMMAND_MODIFIERS = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.MetaModifier
    | Qt.KeyboardModifier.AltModifier
)
_PAGE_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
_ARROW_STEPS = {
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
}
_FOCUS_BIT_KEYS = {Qt.Key.Key_BracketLeft: -1, Qt.Key.Key_BracketRight: 1}
_ZOOM_KEYS = {Qt.Key.Key_Plus: 1, Qt.Key.Key_Equal: 1, Qt.Key.Key_Minus: -1}
_PAGE_KEYS = {
    Qt.Key.Key_1: Panel.INFO,
    Qt.Key.Key_2: Panel.HISTOGRAM,
    Qt.Key.Key_3: Panel.EXTRACT,
    Qt.Key.Key_4: Panel.SCAN,
    Qt.Key.Key_5: Panel.ARNOLD,
}
_CHANNEL_LETTERS = ("R", "G", "B", "A", "L")
_ORDERED_LETTERS = tuple(str(index) for index in range(1, 10))


@dataclass(frozen=True, slots=True)
class Copy:
    """The cursor's readout to the clipboard."""


@dataclass(frozen=True, slots=True)
class Cursor:
    """One step along the arrows: the gallery's or the cursor's, per the page."""

    dx: int
    dy: int
    step: int  # pixels a nudge covers; the gallery ignores it


@dataclass(frozen=True, slots=True)
class FocusBit:
    """One bit up or down the channel in hand."""

    delta: int


@dataclass(frozen=True, slots=True)
class Zoom:
    """One zoom step in or out."""

    step: int


@dataclass(frozen=True, slots=True)
class Fit:
    """Zoom so the whole picture fits the window."""


@dataclass(frozen=True, slots=True)
class CycleFormat:
    """The readout's decimal / hex / binary cycle."""


@dataclass(frozen=True, slots=True)
class ChannelLsb:
    """One channel's lowest bit, alone."""

    name: str


@dataclass(frozen=True, slots=True)
class OrderedLsb:
    """The lowest bit of the channel at that position in the grid order."""

    position: int


@dataclass(frozen=True, slots=True)
class Page:
    """Raise that sidebar page."""

    panel: Panel


type KeyCommand = (
    Copy | Cursor | FocusBit | Zoom | Fit | CycleFormat | ChannelLsb | OrderedLsb | Page
)


def key_command(
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier,
    text: str,
    *,
    auto_repeat: bool = False,
) -> KeyCommand | None:
    """The command a press names, or ``None`` when it names none.

    The letter and digit commands answer the character the event carries, so
    they follow the keyboard layout; they fire once per physical press, while
    the arrows, the zoom and the fit repeat as long as the key is held.
    """
    if modifiers & _COMMAND_MODIFIERS:
        if key == Qt.Key.Key_C:
            return Copy()
        if modifiers & _PAGE_MODIFIERS and (panel := _PAGE_KEYS.get(key)) is not None:
            return Page(panel)
        return None
    if (arrows := _ARROW_STEPS.get(key)) is not None:
        step = NUDGE_STEP if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
        return Cursor(*arrows, step)
    if (delta := _FOCUS_BIT_KEYS.get(key)) is not None:
        return FocusBit(delta)
    if (step := _ZOOM_KEYS.get(key)) is not None:
        return Zoom(step)
    if key == Qt.Key.Key_0:
        return Fit()
    if auto_repeat:
        return None
    label = text.upper()
    if label == "F":
        return CycleFormat()
    if label in _CHANNEL_LETTERS:
        return ChannelLsb(label)
    if label in _ORDERED_LETTERS:
        return OrderedLsb(int(label) - 1)
    return None

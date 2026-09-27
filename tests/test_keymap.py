"""What a key press means, mapped away from any widget: the pure table."""

from PySide6.QtCore import Qt

from pixelsb.ui import keymap
from pixelsb.ui.side_panels import Panel

_NONE = Qt.KeyboardModifier.NoModifier


def command(key: Qt.Key, modifiers: Qt.KeyboardModifier = _NONE, text: str = "") -> object:
    return keymap.key_command(key, modifiers, text)


def test_the_arrows_step_and_shift_widens_the_step() -> None:
    assert command(Qt.Key.Key_Left) == keymap.Cursor(-1, 0, 1)
    assert command(Qt.Key.Key_Down) == keymap.Cursor(0, 1, 1)
    assert command(Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier) == keymap.Cursor(1, 0, 8)


def test_the_arrows_repeat_while_held() -> None:
    held = keymap.key_command(Qt.Key.Key_Up, _NONE, "", auto_repeat=True)
    assert held == keymap.Cursor(0, -1, 1)


def test_the_copy_chord_takes_either_command_key() -> None:
    assert isinstance(command(Qt.Key.Key_C, Qt.KeyboardModifier.MetaModifier), keymap.Copy)
    assert isinstance(command(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier), keymap.Copy)
    assert command(Qt.Key.Key_R, Qt.KeyboardModifier.MetaModifier) is None


def test_the_command_digits_raise_their_page() -> None:
    pages = {
        Qt.Key.Key_1: Panel.INFO,
        Qt.Key.Key_2: Panel.HISTOGRAM,
        Qt.Key.Key_3: Panel.EXTRACT,
        Qt.Key.Key_4: Panel.SCAN,
        Qt.Key.Key_5: Panel.ARNOLD,
    }
    for key, panel in pages.items():
        assert command(key, Qt.KeyboardModifier.ControlModifier) == keymap.Page(panel)
    assert command(Qt.Key.Key_6, Qt.KeyboardModifier.ControlModifier) is None
    # alt alone is not the page chord
    assert command(Qt.Key.Key_1, Qt.KeyboardModifier.AltModifier) is None


def test_the_view_keys_mean_their_view() -> None:
    assert command(Qt.Key.Key_BracketLeft) == keymap.FocusBit(-1)
    assert command(Qt.Key.Key_BracketRight) == keymap.FocusBit(1)
    assert command(Qt.Key.Key_Plus) == keymap.Zoom(1)
    assert command(Qt.Key.Key_Equal) == keymap.Zoom(1)
    assert command(Qt.Key.Key_Minus) == keymap.Zoom(-1)
    assert isinstance(command(Qt.Key.Key_0), keymap.Fit)


def test_the_letters_answer_the_text_the_key_carries() -> None:
    assert command(Qt.Key.Key_R, text="r") == keymap.ChannelLsb("R")
    assert command(Qt.Key.Key_R, Qt.KeyboardModifier.ShiftModifier, "R") == keymap.ChannelLsb("R")
    assert command(Qt.Key.Key_L, text="l") == keymap.ChannelLsb("L")
    assert command(Qt.Key.Key_3, text="3") == keymap.OrderedLsb(2)
    assert isinstance(command(Qt.Key.Key_F, text="f"), keymap.CycleFormat)


def test_the_letters_fire_once_per_press() -> None:
    assert keymap.key_command(Qt.Key.Key_B, _NONE, "b", auto_repeat=True) is None


def test_a_key_with_no_meaning_names_none() -> None:
    assert command(Qt.Key.Key_X, text="x") is None
    assert command(Qt.Key.Key_9, text="9") == keymap.OrderedLsb(8)
    assert command(Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier, "") is None

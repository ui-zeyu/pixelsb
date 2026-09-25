"""The command input's language: one line names one operation, of one kind."""

import pytest

from pixelsb.domain import commands
from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    GrayscaleMask,
    InvertMask,
    Mask,
    RegionMask,
    SampleOrigin,
    SamplePlane,
    ThresholdMask,
    XorMask,
)
from pixelsb.domain.selection import all_bits, channel_members
from tests.support import planes_rgb

_PLANES = planes_rgb()
_PLANES_16 = (SamplePlane("L", 0, 16, SampleOrigin.RAW),)

_WHOLE = {name: frozenset(channel_members(_PLANES, name)) for name in ("R", "G", "B")}


def test_a_channel_and_a_bit_show_that_one_bit() -> None:
    assert commands.parse("b.0", _PLANES) == BitsMask(frozenset({BitChoice("B", 0)}))
    assert commands.parse("R.7", _PLANES) == BitsMask(frozenset({BitChoice("R", 7)}))


def test_a_bare_channel_means_the_channel_whole() -> None:
    assert commands.parse("b", _PLANES) == BitsMask(_WHOLE["B"])
    assert commands.parse("G", _PLANES) == BitsMask(_WHOLE["G"])


def test_or_joins_selections() -> None:
    assert commands.parse("b or r", _PLANES) == BitsMask(_WHOLE["B"] | _WHOLE["R"])
    assert commands.parse("r or g.0", _PLANES) == BitsMask(_WHOLE["R"] | {BitChoice("G", 0)})
    assert commands.parse("r or g or b.0", _PLANES) == BitsMask(
        _WHOLE["R"] | _WHOLE["G"] | {BitChoice("B", 0)}
    )


def test_and_intersects_and_not_complements() -> None:
    assert commands.parse("all and not r", _PLANES) == BitsMask(_WHOLE["G"] | _WHOLE["B"])
    assert commands.parse("not b", _PLANES) == BitsMask(_WHOLE["R"] | _WHOLE["G"])


@pytest.mark.parametrize(
    ("text", "mask"),
    [
        ("thr 128", ThresholdMask(128)),
        ("thr 0x80", ThresholdMask(128)),
        ("thr 010", ThresholdMask(10)),
        ("xor 0xFF", XorMask(0xFF)),
        ("inv", InvertMask()),
        ("gray", GrayscaleMask()),
        ("crop", CropMask()),
        ("  thr 128  ", ThresholdMask(128)),  # the line is read trimmed
    ],
)
def test_each_verb_names_its_mask(text: str, mask: Mask) -> None:
    assert commands.parse(text, _PLANES) == mask


def test_a_region_condition_is_read_as_its_own_language() -> None:
    assert commands.parse("b > r", _PLANES) == RegionMask("b > r")
    assert commands.parse("b>r and b>g", _PLANES) == RegionMask("b > r and b > g")
    assert commands.parse("b > r and rect(0, 0, 10, 10)", _PLANES) == RegionMask(
        "b > r and rect(0, 0, 10, 10)"
    )
    assert commands.parse("B.0 == 1", _PLANES) == RegionMask("B.0 == 1")  # a bit field in a test
    assert commands.parse("x > 5", _PLANES) == RegionMask("x > 5")  # an unknown field, later
    assert commands.parse("R.12 == 1", _PLANES) == RegionMask("R.12 == 1")  # a bit, later


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("b>r and b, 0", "逗号"),  # a comma is Python's tuple, not a condition
        ("b.0, b.1", "逗号"),
        ("[1, 2]", "列表"),
        ("R[0]", "不支持的表达式元素"),
        ("lambda: 1", "不支持的表达式元素"),
        ("foo(1)", "不支持的函数"),
        ("rect(0, 0)", "4 个参数"),
    ],
)
def test_a_line_that_is_no_condition_at_all_is_refused(text: str, message: str) -> None:
    """The box judges the line; whether the image can supply a field is the stack's."""
    with pytest.raises(commands.CommandError, match=message):
        commands.parse(text, _PLANES)


@pytest.mark.parametrize(
    "text",
    [
        "b > r and b.0",  # a region condition and a selection
        "b.0 and b > r",
        "b.0 or r > 5",
        "all and left < 10",
        "not (b.0 or r > 5)",
    ],
)
def test_one_line_names_one_kind_of_operation(text: str) -> None:
    """``and``/``or``/``not`` join one kind: two kinds would be two operations."""
    with pytest.raises(commands.CommandError, match="一行只写一个操作"):
        commands.parse(text, _PLANES)


@pytest.mark.parametrize(
    "text",
    [
        "inv or gray",
        "not crop",
        "b.0 or inv",
        "not (thr 5)",
        "thr 128 and inv",
        "b.0 and thr 128",  # a verb that is not at the head of the line
        "thr 128 and b.0",
    ],
)
def test_a_verb_is_the_whole_line(text: str) -> None:
    with pytest.raises(commands.CommandError, match="单独一行"):
        commands.parse(text, _PLANES)


def test_a_half_typed_line_names_no_operation() -> None:
    """Mid-word, mid-operator, mid-call: the line is unfinished, so nothing applies."""
    assert commands.parse("R >= ", _PLANES) is None
    assert commands.parse("b.0 and", _PLANES) is None
    assert commands.parse("b>r and b.", _PLANES) is None
    assert commands.parse("rect(1, ", _PLANES) is None  # a call not yet finished
    assert commands.parse("", _PLANES) is None
    assert commands.parse("   ", _PLANES) is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("thr", "需要一个数值"),
        ("thr ff", "数值看不懂"),  # hexadecimal says 0x
        ("thr 0xzz", "数值看不懂"),
        ("thr -1", "超出"),
        ("thr 99999999", "超出"),
        ("xor", "需要一个数值"),
        ("inv 3", "不需要参数"),
        ("crop 2", "不需要参数"),
        ("b.9", "只有 8 位"),
        ("x.0", "没有 x 这样的通道"),
        ("x", "没有 x 这样的通道"),
    ],
)
def test_a_broken_command_says_what_is_wrong(text: str, message: str) -> None:
    with pytest.raises(commands.CommandError, match=message):
        commands.parse(text, _PLANES)


def test_a_level_past_the_widest_channel_of_this_image_is_refused() -> None:
    """A level the image cannot reach would be a mask that quietly does nothing."""
    assert commands.parse("thr 255", _PLANES) == ThresholdMask(255)
    assert commands.parse("xor 0xFFFF", _PLANES_16) == XorMask(0xFFFF)
    with pytest.raises(commands.CommandError, match=r"超出这张图的 0\.\.255"):
        commands.parse("thr 256", _PLANES)
    with pytest.raises(commands.CommandError, match=r"超出这张图的 0\.\.255"):
        commands.parse("xor 0xFF00", _PLANES)
    with pytest.raises(commands.CommandError, match=r"超出这张图的 0\.\.65535"):
        commands.parse("thr 0x10000", _PLANES_16)


@pytest.mark.parametrize(
    "mask",
    [
        BitsMask(frozenset({BitChoice("R", 3)})),
        BitsMask(channel_members(_PLANES, "G")),
        BitsMask(all_bits(_PLANES)),
        BitsMask(frozenset({BitChoice("R", 0), BitChoice("G", 1)})),
        RegionMask("rect(0, 0, 10, 10) and R > 3"),
        InvertMask(),
        GrayscaleMask(),
        ThresholdMask(200),
        XorMask(0x0F),
        CropMask(),
    ],
)
def test_a_mask_reads_back_as_the_command_that_means_it(mask: Mask) -> None:
    assert commands.parse(commands.text_of(mask, _PLANES) or "", _PLANES) == mask


def test_the_command_lines_are_in_the_house_style() -> None:
    assert commands.text_of(InvertMask(), _PLANES) == "inv"
    assert commands.text_of(ThresholdMask(200), _PLANES) == "thr 200"
    assert commands.text_of(XorMask(0x0F), _PLANES) == "xor 0x0F"
    assert commands.text_of(BitsMask(frozenset({BitChoice("B", 0)})), _PLANES) == "b.0"
    assert commands.text_of(BitsMask(channel_members(_PLANES, "B")), _PLANES) == "b"
    assert commands.text_of(BitsMask(all_bits(_PLANES)), _PLANES) == "all"
    assert (
        commands.text_of(
            BitsMask(frozenset({BitChoice("B", 0), BitChoice("B", 1)})),
            _PLANES,
        )
        == "b.0 or b.1"
    )


def test_a_bits_selection_the_text_cannot_say_has_no_line() -> None:
    assert commands.text_of(BitsMask(frozenset()), _PLANES) is None  # nothing to name


def test_a_selection_naming_another_images_channel_has_no_line() -> None:
    deep = BitsMask(
        frozenset({BitChoice("R", 0), BitChoice("L", 5)})  # L is no plane of this image
    )
    assert commands.text_of(deep, _PLANES) is None


def test_a_narrow_plane_takes_the_bits_it_has() -> None:
    planes = (
        SamplePlane("R", 0, 8, SampleOrigin.RAW),
        SamplePlane("A", 1, 1, SampleOrigin.RAW),
    )
    alpha = BitsMask(frozenset({BitChoice("A", 0)}))
    assert commands.parse("a", planes) == alpha
    assert commands.text_of(alpha, planes) == "a"  # the one bit is the channel whole

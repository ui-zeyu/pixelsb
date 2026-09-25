import numpy as np
import pytest

from pixelsb.domain.extract import (
    ExtractRow,
    applied_order,
    decode_row,
    decode_rows,
    extract_bytes,
    filter_extract,
    format_extract,
    order_choices,
    ordered_planes,
)
from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    ExtractEncoding,
    ExtractOrder,
    LoadedImage,
    SampleOrigin,
    SamplePlane,
    ScanOrder,
)
from tests.support import make_image, planes_rgb, planes_rgba


def _image(samples: list) -> LoadedImage:
    return make_image(np.array(samples, dtype=np.uint16), planes_rgb())


def test_extract_without_selection_uses_every_bit() -> None:
    image = _image([[[0xA5, 0x0F, 0x3C]]])
    assert extract_bytes(image, None) == bytes([0xA5, 0xF0, 0x3C])


def test_extract_only_packs_the_matching_pixels() -> None:
    image = _image([[[0b0001, 0b0001, 0], [0b0001, 0b0000, 0]]])
    chosen = frozenset({BitChoice("R", 0), BitChoice("G", 0)})
    assert extract_bytes(image, chosen) == bytes([0b11100000])
    first_only = np.array([[True, False]])
    assert extract_bytes(image, chosen, first_only) == bytes([0b11000000])
    second_only = np.array([[False, True]])
    assert extract_bytes(image, chosen, second_only) == bytes([0b10000000])
    none_match = np.array([[False, False]])
    assert extract_bytes(image, chosen, none_match) == b""


def test_format_rows_keep_the_offset_beside_the_text() -> None:
    data = b"MZ\x00" + bytes(range(3, 20))
    rows = format_extract(data)
    assert rows[0].offset == 0
    assert rows[0].offset_text == "00000000"
    assert rows[0].text.startswith("4d 5a 00 03 04")
    assert "  MZ.." in rows[0].text
    assert "00000000" not in rows[0].text
    assert rows[1].offset == 16
    assert rows[1].offset_text == "00000010"


def test_format_pads_the_last_row_and_handles_empty() -> None:
    rows = format_extract(b"\x01")
    assert rows[0].offset_text == "00000000"
    assert rows[0].text.startswith("01")
    assert rows[0].text.endswith(".")
    empty = format_extract(b"")
    assert [row.text for row in empty] == ["（无数据）"]
    assert empty[0].offset is None
    assert empty[0].offset_text == ""


def test_rows_keep_their_bytes_so_they_can_be_re_read() -> None:
    rows = format_extract(b"MZ\x00" + bytes(range(3, 20)))
    assert rows[0].data == b"MZ\x00" + bytes(range(3, 16))
    assert rows[1].data == bytes(range(16, 20))
    assert format_extract(b"")[0].data == b""  # a note row carries no bytes


def test_decoding_ascii_dots_everything_unprintable() -> None:
    row = format_extract(b"A\x00\n\xffZ")[0]
    assert decode_row(row, ExtractEncoding.ASCII) == "A...Z"


def test_decoding_utf8_reads_whole_sequences() -> None:
    row = format_extract("héllo".encode())[0]
    assert decode_row(row, ExtractEncoding.UTF8) == "héllo"
    broken = format_extract(b"\xff\xfeA")[0]
    assert decode_row(broken, ExtractEncoding.UTF8) == "\ufffd\ufffdA"


def test_decoding_utf16_takes_two_bytes_per_character() -> None:
    row = format_extract("中A".encode("utf-16-le"))[0]
    assert decode_row(row, ExtractEncoding.UTF16_LE) == "中A"
    assert decode_row(row, ExtractEncoding.UTF16_BE) == "ⵎ䄀"
    odd = format_extract(b"A\x00Z")[0]
    assert decode_row(odd, ExtractEncoding.UTF16_LE) == "A\ufffd"


def test_decoding_marks_a_character_split_by_the_row_boundary() -> None:
    data = b"A" * 15 + "中".encode() + b"B"
    rows = format_extract(data)
    assert decode_row(rows[0], ExtractEncoding.UTF8) == "A" * 15 + "\ufffd"
    assert decode_row(rows[1], ExtractEncoding.UTF8) == "\ufffd\ufffdB"  # its two tail bytes
    assert decode_row(ExtractRow(None, "（无数据）"), ExtractEncoding.UTF8) == ""


def test_a_whole_dump_reads_a_character_split_by_the_row_boundary() -> None:
    """Rows decode in sequence, so a comment does not break at every 16 bytes."""
    data = b"A" * 15 + "中".encode() + b"B"
    lines = decode_rows(format_extract(data), ExtractEncoding.UTF8)
    assert "\ufffd" not in "".join(lines)
    assert lines[0] == "A" * 15  # the character's first byte waits for the rest
    assert lines[1] == "中B"


def test_a_whole_dump_takes_utf16_pairs_across_rows() -> None:
    data = "AB".encode("utf-16-le") * 8  # a pair lands on the boundary
    lines = decode_rows(format_extract(data), ExtractEncoding.UTF16_LE)
    assert "".join(lines) == "AB" * 8


def test_a_row_shown_without_its_neighbour_stands_alone() -> None:
    """A search that hides rows cannot glue bytes that were never adjacent."""
    data = b"A" * 15 + "中".encode() + b"B"
    rows = format_extract(data)
    (alone,) = decode_rows([rows[1]], ExtractEncoding.UTF8)
    assert alone == "\ufffd\ufffdB"  # the tail bytes with no character to finish


def test_format_truncates_with_a_note() -> None:
    rows = format_extract(bytes(256), limit=4)
    assert len(rows) == 5
    assert rows[3].offset_text == "00000030"
    assert rows[4].offset is None
    assert rows[4].text == "…已截断，共 256 字节"


def test_format_does_not_claim_truncation_on_an_exact_fit() -> None:
    rows = format_extract(bytes(64), limit=4)
    assert [row.offset for row in rows] == [0, 16, 32, 48]
    assert all(row.offset is not None for row in rows)


def test_filter_matches_hex_ascii_or_offset() -> None:
    rows = format_extract(b"flag{abc}" + bytes([0xDE]) * 8)
    assert filter_extract(rows, "flag") == rows[:1]
    assert filter_extract(rows, "de") == rows
    assert filter_extract(rows, "00000010") == rows[1:]
    assert filter_extract(rows, "zzz") == []


def test_channel_order_reads_the_channels_in_the_sequence_it_names() -> None:
    # One bit per channel, each set to a distinguishable value.
    image = _image([[[0b0001, 0b0010, 0b0000]]])
    chosen = frozenset({BitChoice("R", 0), BitChoice("G", 1), BitChoice("B", 0)})
    assert extract_bytes(image, chosen) == bytes([0b11000000])
    reversed_order = ExtractOrder(planes=("B", "G", "R"))
    assert extract_bytes(image, chosen, order=reversed_order) == bytes([0b01100000])


def test_scan_order_reads_the_pixels_column_by_column() -> None:
    image = _image([[[1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
    chosen = frozenset({BitChoice("R", 0)})
    # Row by row the first column's two pixels are split by the second column's;
    # column by column they come together.
    assert extract_bytes(image, chosen) == bytes([0b10100000])
    columns = ExtractOrder(scan=ScanOrder.YZ)
    assert extract_bytes(image, chosen, order=columns) == bytes([0b11000000])


def test_bit_order_fills_each_byte_from_the_end_it_names() -> None:
    samples = np.zeros((8, 1, 3), dtype=np.uint16)
    samples[0, 0, 0] = 1  # only the first pixel's lowest red bit is set
    image = make_image(samples, planes_rgb())
    chosen = frozenset({BitChoice("R", 0)})
    assert extract_bytes(image, chosen) == bytes([0b10000000])
    low_first = ExtractOrder(bit_order=BitOrder.LSB)
    assert extract_bytes(image, chosen, order=low_first) == bytes([0b00000001])


def test_the_three_orders_compose_into_one_stream() -> None:
    # A 2x3 image whose pixels carry their own number 0..5 in the low three bits
    # of R, G, B, so every combination of the three orders reads differently.
    image = _image(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 1], [1, 0, 1]],
        ]
    )
    chosen = frozenset({BitChoice(name, 0) for name in ("R", "G", "B")})
    cases = {
        (("R", "G", "B"), BitOrder.MSB, ScanOrder.XY): bytes([0x11, 0x63, 0x40]),
        (("R", "G", "B"), BitOrder.LSB, ScanOrder.XY): bytes([0x88, 0xC6, 0x02]),
        (("R", "G", "B"), BitOrder.MSB, ScanOrder.YZ): bytes([0x1A, 0x15, 0x40]),
        (("R", "G", "B"), BitOrder.LSB, ScanOrder.YZ): bytes([0x58, 0xA8, 0x02]),
        (("B", "G", "R"), BitOrder.MSB, ScanOrder.XY): bytes([0x05, 0x39, 0x40]),
        (("B", "G", "R"), BitOrder.LSB, ScanOrder.XY): bytes([0xA0, 0x9C, 0x02]),
        (("B", "G", "R"), BitOrder.MSB, ScanOrder.YZ): bytes([0x0C, 0xC5, 0x40]),
        (("B", "G", "R"), BitOrder.LSB, ScanOrder.YZ): bytes([0x30, 0xA3, 0x02]),
    }
    for (planes, bit_order, scan), packed in cases.items():
        order = ExtractOrder(planes=planes, bit_order=bit_order, scan=scan)
        assert extract_bytes(image, chosen, order=order) == packed, (planes, bit_order, scan)


def test_order_choices_offer_every_arrangement_of_three_channels() -> None:
    chosen = frozenset({BitChoice("R", 0), BitChoice("G", 0), BitChoice("B", 0)})
    choices = order_choices(planes_rgb(), chosen)
    assert choices[0] == ("R", "G", "B")
    assert set(choices) == {
        ("R", "G", "B"),
        ("R", "B", "G"),
        ("G", "R", "B"),
        ("G", "B", "R"),
        ("B", "R", "G"),
        ("B", "G", "R"),
    }


def test_order_choices_only_use_the_channels_that_carry_bits() -> None:
    planes = planes_rgb()
    assert order_choices(planes, frozenset()) == ()
    assert order_choices(planes, frozenset({BitChoice("B", 0)})) == (("B",),)
    pair = frozenset({BitChoice("R", 0), BitChoice("B", 0)})
    assert order_choices(planes, pair) == (("R", "B"), ("B", "R"))


def test_order_choices_keep_the_order_and_its_reverse_beyond_three_channels() -> None:
    chosen = frozenset(BitChoice(name, 0) for name in ("R", "G", "B", "A"))
    assert order_choices(planes_rgba(), chosen) == (
        ("R", "G", "B", "A"),
        ("A", "B", "G", "R"),
    )


def test_the_channel_preference_keeps_unmentioned_planes_after_the_named_ones() -> None:
    planes = planes_rgba()
    named = ExtractOrder(planes=("B", "G", "R"))
    assert [plane.name for plane in ordered_planes(planes, named)] == ["B", "G", "R", "A"]
    assert [plane.name for plane in ordered_planes(planes, ExtractOrder())] == [
        "R",
        "G",
        "B",
        "A",
    ]


def test_the_channel_preference_ignores_names_the_image_does_not_have() -> None:
    gray = (SamplePlane("L", 0, 8, SampleOrigin.RAW),)
    assert ordered_planes(gray, ExtractOrder(planes=("B", "G", "R"))) == gray


def test_the_applied_order_lists_the_planes_that_carry_bits() -> None:
    chosen = frozenset({BitChoice("R", 0), BitChoice("B", 3)})
    order = ExtractOrder(planes=("B", "R"))
    assert applied_order(planes_rgb(), chosen, order) == ("B", "R")
    assert applied_order(planes_rgb(), frozenset(), order) == ()


def test_the_order_record_refuses_a_channel_listed_twice() -> None:
    with pytest.raises(ValueError, match="repeats a plane name"):
        ExtractOrder(planes=("R", "G", "R"))

import numpy as np

from pixelsb.domain.extract import extract_bytes, filter_extract, format_extract
from pixelsb.domain.models import BitChoice
from tests.support import make_image, planes_rgb


def _image(samples):
    return make_image(np.array(samples, dtype=np.uint16), planes_rgb())


def test_extract_packs_the_selected_bits_msb_first_in_plane_order() -> None:
    image = _image([[[0b1001, 0b0001, 0], [0b0000, 0b0000, 0]]])
    chosen = {BitChoice("R", 0), BitChoice("G", 0)}
    stream = [1, 1, 0, 0, 0, 0, 0, 0]
    assert extract_bytes(image, frozenset(chosen)) == bytes([0b11000000])
    assert stream[0] == 1


def test_extract_full_byte_matches_the_channel_value() -> None:
    image = _image([[[0xA5, 0, 0]]])
    assert extract_bytes(image, frozenset(BitChoice("R", bit) for bit in range(8))) == bytes([0xA5])


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


def test_format_truncates_with_a_note() -> None:
    rows = format_extract(bytes(256), limit=4)
    assert len(rows) == 5
    assert rows[3].offset_text == "00000030"
    assert rows[4].offset is None
    assert rows[4].text == "…已截断，共 256 字节"


def test_filter_matches_hex_ascii_or_offset() -> None:
    rows = format_extract(b"flag{abc}" + bytes([0xDE]) * 8)
    assert filter_extract(rows, "flag") == rows[:1]
    assert filter_extract(rows, "de") == rows
    assert filter_extract(rows, "00000010") == rows[1:]
    assert filter_extract(rows, "zzz") == []

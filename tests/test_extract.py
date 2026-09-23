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


def test_format_rows_show_offset_hex_and_ascii() -> None:
    data = b"MZ\x00" + bytes(range(3, 20))
    lines = format_extract(data)
    assert lines[0].startswith("00000000  4d 5a 00 03 04")
    assert "  MZ.." in lines[0]
    assert lines[1].startswith("00000010")


def test_format_pads_the_last_row_and_handles_empty() -> None:
    lines = format_extract(b"\x01")
    assert lines[0].startswith("00000000  01")
    assert lines[0].endswith(".")
    assert format_extract(b"") == ["（无数据）"]


def test_format_truncates_with_a_note() -> None:
    lines = format_extract(bytes(256), limit=4)
    assert len(lines) == 5
    assert lines[3].startswith("00000030")
    assert lines[4] == "…已截断，共 256 字节"


def test_filter_matches_hex_or_ascii_case_insensitive() -> None:
    lines = format_extract(b"flag{abc}" + bytes([0xDE]) * 8)
    assert filter_extract(lines, "flag") == lines[:1]
    assert filter_extract(lines, "de") == lines
    assert filter_extract(lines, "zzz") == []

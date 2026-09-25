"""The container census: block lists, planted data, and the EXIF tables."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational

from pixelsb.domain.container import BlockRole, PngHeader, render_blocks, scan_container
from pixelsb.io.inspect import exif_entries, inspect_container
from tests.support import chunk as _chunk

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # a 1x1 grayscale header


def _png(*chunks: bytes) -> bytes:
    return _PNG_SIGNATURE + b"".join(chunks)


def _scanlines(*pixels: bytes) -> bytes:
    return b"".join(b"\x00" + pixel for pixel in pixels)


def test_a_clean_png_lists_only_required_blocks() -> None:
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"IEND", b""),
        )
    )
    assert report.kind == "png"
    assert report.findings == ()
    assert [block.label for block in report.blocks] == ["IHDR", "IDAT", "IEND"]
    assert {block.role for block in report.blocks} == {BlockRole.REQUIRED}


def test_idats_split_inside_one_stream_share_a_group() -> None:
    """The legal multi-chunk layout: one zlib stream spread over several IDATs."""
    stream = zlib.compress(_scanlines(b"\xff"))
    middle = len(stream) // 2
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", stream[:middle]),
            _chunk(b"IDAT", stream[middle:]),
            _chunk(b"IEND", b""),
        )
    )
    assert report.findings == ()
    idat_groups = {block.group for block in report.blocks if block.label == "IDAT"}
    assert idat_groups == {1}  # both chunks serve the same stream


def test_a_lone_idat_chunk_is_still_stream_one() -> None:
    """One chunk holding the whole stream is stream 1 — the decoder's stream."""
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"IEND", b""),
        )
    )
    assert report.blocks[1].label == "IDAT"
    assert report.blocks[1].group == 1


def test_an_unreadable_idat_stream_gets_no_number() -> None:
    """A stream the audit cannot read whole is bytes, not a numbered stream."""
    report = scan_container(
        _png(_chunk(b"IHDR", _IHDR), _chunk(b"IDAT", b"not zlib"), _chunk(b"IEND", b""))
    )
    assert report.blocks[1].group == 0


def test_the_spec_bit_classifies_ancillary_chunks() -> None:
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"tEXt", b"Hint\x00look here"),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"IEND", b""),
        )
    )
    text_chunk = report.blocks[1]
    assert text_chunk.label == "tEXt"
    assert text_chunk.role is BlockRole.ANCILLARY
    assert "look here" in text_chunk.preview
    assert report.findings == ()  # ancillary alone is not an anomaly: the census explains it


def test_bytes_past_iend_travel_as_a_block() -> None:
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"IEND", b""),
        )
        + b"stowaway"
    )
    assert report.findings == ()  # the residual is a block to inspect, not a warning
    last = report.blocks[-1]
    assert last.label == "残留数据"
    assert last.role is BlockRole.ANCILLARY
    assert last.payload == b"stowaway"
    assert last.payload_at == last.offset
    assert [block.label for block in report.blocks[:-1]] == ["IHDR", "IDAT", "IEND"]


def test_a_png_stitched_after_the_end_stays_one_residual_block() -> None:
    """The census reads the first image; a second image is tail data, like a zip."""
    stream = zlib.compress(_scanlines(b"\xff"))
    first = _png(
        _chunk(b"IHDR", _IHDR),
        _chunk(b"IDAT", stream[: len(stream) // 2]),
        _chunk(b"IDAT", stream[len(stream) // 2 :]),
        _chunk(b"IEND", b""),
    )
    second = _png(
        _chunk(b"IHDR", _IHDR),
        _chunk(b"IDAT", zlib.compress(_scanlines(b"\x11"))),
        _chunk(b"IEND", b""),
    )
    report = scan_container(first + second + b"junk")
    assert [block.label for block in report.blocks] == ["IHDR", "IDAT", "IDAT", "IEND", "残留数据"]
    tail = report.blocks[-1]
    assert tail.offset == len(first)
    assert tail.payload == second + b"junk"
    assert tail.payload_at == tail.offset
    # The second image's own chunks are bytes, not structure: no IDAT stream
    # accounting for them, and nothing reads them as part of the first image.
    assert [block.group for block in report.blocks if block.label == "IDAT"] == [1, 1]
    assert report.findings == ()


def test_a_second_iend_still_raises_the_acropalypse_tell() -> None:
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"IEND", b""),
            _chunk(b"IEND", b""),
            b"\x00\x00\x00\x00",
        )
    )
    (finding,) = report.findings
    assert finding.kind == "duplicate-eof"
    assert finding.offset == report.blocks[-2].offset + 12  # right after the real IEND
    assert report.blocks[-1].label == "残留数据"


def test_each_fake_idat_stream_gets_its_own_group() -> None:
    """Two smuggled streams after the real one: three groups, two findings."""
    real = zlib.compress(_scanlines(b"\xff"))
    fake_one = zlib.compress(b"first fake")
    fake_two = zlib.compress(b"second fake flag{three_streams}")
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", real),
            _chunk(b"IDAT", fake_one),
            _chunk(b"IDAT", fake_two),
            _chunk(b"IEND", b""),
        )
    )
    groups = [block.group for block in report.blocks if block.label == "IDAT"]
    assert groups == [1, 2, 3]
    extra = [finding for finding in report.findings if finding.kind == "idat-extra"]
    assert len(extra) == 2
    assert extra[0].decoded == b"first fake"
    assert extra[1].decoded == b"second fake flag{three_streams}"


def test_surplus_pixels_inside_one_stream_are_reported() -> None:
    stream = zlib.compress(_scanlines(b"\xff") + b"\x00\xaa" * 5)  # five rows too many
    report = scan_container(
        _png(_chunk(b"IHDR", _IHDR), _chunk(b"IDAT", stream), _chunk(b"IEND", b""))
    )
    (finding,) = report.findings
    assert finding.kind == "idat-oversize"
    assert finding.length == 10  # five rows of two bytes: filter flag plus value
    assert finding.payload == b"\x00\xaa" * 5


def test_a_truncated_stream_is_reported() -> None:
    stream = zlib.compress(_scanlines(b"\xff"))[:-4]
    report = scan_container(
        _png(_chunk(b"IHDR", _IHDR), _chunk(b"IDAT", stream), _chunk(b"IEND", b""))
    )
    (finding,) = report.findings
    assert finding.kind == "stream-truncated"


def test_a_corrupt_stream_is_reported_not_raised() -> None:
    report = scan_container(
        _png(_chunk(b"IHDR", _IHDR), _chunk(b"IDAT", b"definitely not zlib"), _chunk(b"IEND", b""))
    )
    (finding,) = report.findings
    assert finding.kind == "stream-truncated"


def test_idat_groups_split_by_another_block_are_flagged() -> None:
    report = scan_container(
        _png(
            _chunk(b"IHDR", _IHDR),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
            _chunk(b"tEXt", b"gap"),
            _chunk(b"IDAT", zlib.compress(_scanlines(b"\x11"))),
            _chunk(b"IEND", b""),
        )
    )
    kinds = [finding.kind for finding in report.findings]
    assert "idat-gap" in kinds
    gap = next(finding for finding in report.findings if finding.kind == "idat-gap")
    assert gap.offset == report.blocks[3].offset  # the second group's chunk


def test_a_jpeg_census_lists_segments_and_the_residual_block(tmp_path: Path) -> None:
    path = tmp_path / "case.jpg"
    Image.new("RGB", (2, 2)).save(path)
    path.write_bytes(path.read_bytes() + b"hidden")
    report = inspect_container(path)
    assert report.kind == "jpeg"
    assert report.blocks[0].label == "SOI"
    assert report.blocks[-2].label == "EOI"
    assert report.blocks[-1].label == "残留数据"
    assert report.blocks[-1].payload == b"hidden"
    assert report.findings == ()


def test_a_png_pasted_onto_a_jpeg_is_one_residual_block(tmp_path: Path) -> None:
    path = tmp_path / "case.jpg"
    Image.new("RGB", (2, 2)).save(path)
    tail = _png(
        _chunk(b"IHDR", _IHDR),
        _chunk(b"IDAT", zlib.compress(_scanlines(b"\xff"))),
        _chunk(b"IEND", b""),
    )
    jpeg = path.read_bytes()
    path.write_bytes(jpeg + tail)
    report = scan_container(path.read_bytes())
    assert report.kind == "jpeg"
    assert report.blocks[-2].label == "EOI"
    assert report.blocks[-1].label == "残留数据"
    assert report.blocks[-1].payload == tail
    assert report.findings == ()


def test_an_unknown_container_stays_empty(tmp_path: Path) -> None:
    path = tmp_path / "case.bin"
    path.write_bytes(b"\x00\x01\x02")
    report = inspect_container(path)
    assert report.kind == ""
    assert report.findings == ()


def test_render_blocks_decodes_zlib_payloads() -> None:
    pixels = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))
    payload = zlib.compress(b"\x00" + pixels[:6] + b"\x00" + pixels[6:])
    rgb = render_blocks([payload], PngHeader(2, 2, 8, 2, 0))
    assert rgb.tolist() == [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]


def test_render_blocks_joins_payloads_but_stops_at_a_second_stream() -> None:
    """The real image renders even when the fake IDAT stays checked."""
    real = zlib.compress(b"\x00" + bytes((9, 8, 7)))
    fake = zlib.compress(b"not pixels")
    rgb = render_blocks([real, fake], PngHeader(1, 1, 8, 2, 0))
    assert rgb.tolist() == [[[9, 8, 7]]]


def test_render_blocks_accepts_raw_scanlines_without_zlib() -> None:
    rgb = render_blocks([b"\x00\x10\x20\x30"], PngHeader(1, 1, 8, 2, 0))
    assert rgb.tolist() == [[[16, 32, 48]]]


def test_render_blocks_truncates_a_long_stream_and_pads_a_short_one() -> None:
    long_stream = zlib.compress(b"\x00" + bytes(18))
    rgb = render_blocks([long_stream], PngHeader(1, 1, 8, 2, 0))
    assert rgb.tolist() == [[[0, 0, 0]]]
    rgb = render_blocks([zlib.compress(b"\x00\x01")], PngHeader(1, 1, 8, 2, 0))
    assert rgb.tolist() == [[[1, 0, 0]]]


def test_render_blocks_converts_gray_to_rgb() -> None:
    rgb = render_blocks([zlib.compress(b"\x00\x80")], PngHeader(1, 1, 8, 0, 0))
    assert rgb.tolist() == [[[128, 128, 128]]]


def test_render_blocks_looks_up_the_palette() -> None:
    payload = zlib.compress(b"\x00\x01")
    palette = bytes((10, 20, 30, 200, 100, 50))
    rgb = render_blocks([payload], PngHeader(1, 1, 8, 3, 0), palette)
    assert rgb.tolist() == [[[200, 100, 50]]]


def test_render_blocks_unpacks_sub_byte_depths() -> None:
    payload = zlib.compress(b"\x00\x80")  # one byte of 1-bit samples: white, then black
    rgb = render_blocks([payload], PngHeader(2, 1, 1, 0, 0))
    assert rgb.tolist() == [[[255, 255, 255], [0, 0, 0]]]


def test_render_blocks_reconstructs_filtered_scanlines() -> None:
    """Row two uses Up (filter 2); its bytes are deltas against row one."""
    stream = b"\x00\x01\x02\x03" + b"\x02\x01\x01\x01"
    rgb = render_blocks([stream], PngHeader(1, 2, 8, 2, 0))
    assert rgb.tolist() == [[[1, 2, 3]], [[2, 3, 4]]]


def test_render_blocks_treats_an_unknown_filter_as_none() -> None:
    rgb = render_blocks([b"\xff\x10\x20\x30"], PngHeader(1, 1, 8, 2, 0))
    assert rgb.tolist() == [[[16, 32, 48]]]


def test_render_blocks_wraps_long_filter_chains_modulo_256() -> None:
    """A real photo's Sub chain climbs past the int16 accumulator; the spec's
    modulo 256 is what keeps the click alive."""
    row = b"\x01" + b"\xff" * 300
    rgb = render_blocks([row], PngHeader(300, 1, 8, 0, 0))
    assert rgb[0, 0, 0] == 255
    assert rgb[0, 128, 0] == 129 * 255 % 256  # the old code overflowed int16 right here
    assert rgb[0, 299, 0] == 300 * 255 % 256


def test_render_blocks_declines_interlaced_data() -> None:
    with pytest.raises(ValueError, match="interlaced"):
        render_blocks([b""], PngHeader(1, 1, 8, 2, 1))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_rendering_a_file_matches_what_the_writer_put_in(tmp_path: Path, mode: str) -> None:
    """Every filter Pillow picks on noisy data, against the pixels it was given.

    A real encoder mixes the five filters over the rows, which is exactly what
    the reconstruction has to get right — including the two it cannot take in
    one pass. Interlacing aside, the pixels must come back as they went in.
    """
    rng = np.random.default_rng(11)
    channels = len(mode)
    samples = rng.integers(0, 256, (23, 37, channels), dtype=np.uint8).squeeze()
    path = tmp_path / f"written-{mode}.png"
    Image.fromarray(samples, mode).save(path, optimize=True)
    data = path.read_bytes()
    report = scan_container(data)
    assert report.header is not None
    payloads = [block.payload for block in report.blocks if block.label == "IDAT"]
    raw = zlib.decompress(b"".join(payloads))
    stride = report.header.pixel_stride()
    assert stride is not None
    kinds = {raw[row * (stride + 1)] for row in range(report.header.height)}
    assert len(kinds) >= 3  # the case is worth running only if the filters differ
    source = np.asarray(samples).reshape(report.header.height, report.header.width, channels)
    expected = np.repeat(source, 3, axis=2) if channels == 1 else source
    assert np.array_equal(render_blocks(payloads, report.header, report.palette), expected)


def test_exif_entries_name_the_tags(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[271] = "pixelsb"  # Make
    image = Image.new("RGB", (1, 1))
    path = tmp_path / "case.tif"
    image.save(path, exif=exif)
    assert ("Make", "pixelsb") in exif_entries(path)


def test_exif_entries_read_the_sub_ifds_and_skip_the_pointers(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif.get_ifd(IFD.Exif)[36867] = "2026:01:02 03:04:05"  # DateTimeOriginal
    exif.get_ifd(IFD.GPSInfo)[1] = "N"  # GPSLatitudeRef
    exif.get_ifd(IFD.GPSInfo)[2] = (IFDRational(31), IFDRational(14), IFDRational(0))
    path = tmp_path / "case.jpg"
    Image.new("RGB", (1, 1)).save(path, exif=exif)
    entries = dict(exif_entries(path))
    assert entries["DateTimeOriginal"] == "2026:01:02 03:04:05"
    assert entries["GPSLatitudeRef"] == "N"
    assert entries["GPSLatitude"] == "31.0, 14.0, 0.0"
    assert str(IFD.Exif) not in entries  # the pointer is not listed as a tag


def test_exif_entries_tolerate_a_container_without_exif(tmp_path: Path) -> None:
    path = tmp_path / "case.png"
    Image.new("RGB", (1, 1)).save(path)
    assert exif_entries(path) == ()

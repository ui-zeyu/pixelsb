"""The file-info page: the scan waits for the page, then findings and EXIF."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QLabel, QPushButton
from pytestqt.qtbot import QtBot

from pixelsb.domain.models import LoadedImage
from pixelsb.io.loading import image_from_rgb, load_image
from pixelsb.ui import text
from pixelsb.ui.info import InfoPanel
from tests.support import chunk as _chunk

_ZIP_START = b"PK\x03\x04"


def _census_png(
    tmp_path: Path, extra: tuple[bytes, ...] = (), name: str = "case.png", interlace: int = 0
) -> Path:
    """A hand-built 1x1 PNG: the census can then see whatever we weld on."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, interlace)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"\x00\xff"))
        + b"".join(extra)
        + _chunk(b"IEND", b"")
    )
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _split_idat_png(tmp_path: Path, name: str = "split.png") -> Path:
    """One stream over two IDAT chunks: the chunks render together."""
    stream = zlib.compress(b"\x00\xff")
    return _census_png(tmp_path, (_chunk(b"IDAT", stream[len(stream) // 2 :]),), name)


def _fake_idat_png(tmp_path: Path, name: str = "fake.png") -> Path:
    """Renders fine, but a second IDAT smuggles a flag no reader will show."""
    return _census_png(tmp_path, (_chunk(b"IDAT", zlib.compress(b"flag{fake_idat}")),), name)


def _png(tmp_path: Path, name: str = "case.png", exif: bool = False) -> Path:
    image = Image.new("RGB", (2, 2))
    path = tmp_path / name
    if exif:
        tags = Image.Exif()
        tags[271] = "pixelsb"  # Make
        image.save(path, exif=tags)
    else:
        image.save(path)
    return path


def _panel(qtbot: QtBot, path: Path) -> InfoPanel:
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_image(load_image(path))
    return panel


def _is_shown(panel: InfoPanel, row: int) -> bool:
    """Whether the census row's name button carries the canvas accent."""
    return bool(panel._name_buttons[row].property("shown"))


def _label_texts(panel: InfoPanel) -> list[str]:
    widgets = (*panel.findChildren(QLabel), *panel.findChildren(QPushButton))
    return [widget.text() for widget in widgets if widget.text()]


def test_the_scan_waits_until_the_page_is_seen(qtbot: QtBot, tmp_path: Path) -> None:
    path = _png(tmp_path)
    image = load_image(path)
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.set_image(image)
    assert panel._name.text() == ""  # hidden pages are not read
    panel.show()
    assert panel._name.text() == "case.png"
    assert panel._path.text() == str(path)
    assert panel._facts.text() == text.file_facts(image)
    assert text.INFO_NO_EXIF in _label_texts(panel)


def test_a_clean_file_stays_silent(qtbot: QtBot, tmp_path: Path) -> None:
    """No findings, no words: the block list and the canvas note are the report."""
    panel = _panel(qtbot, _png(tmp_path))
    assert panel._warn.isHidden()
    assert panel._hits.isHidden()
    report = panel._report
    assert report is not None
    assert panel._canvas_note.text() == text.canvas_note(report.blocks[1], 1)  # its lone IDAT


def test_the_block_census_is_listed(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _png(tmp_path))
    texts = _label_texts(panel)
    assert "IHDR" in texts
    assert "IDAT" in texts
    assert "IEND" in texts


def test_each_row_shows_a_text_taste_of_the_blocks_data(qtbot: QtBot, tmp_path: Path) -> None:
    """The right end of a row previews the payload as text, so a tail names itself."""
    path = _png(tmp_path)
    path.write_bytes(path.read_bytes() + b"\x89PNG\r\n\x1a\n" + b"rest of a second image")
    panel = _panel(qtbot, path)
    texts = _label_texts(panel)
    assert "............." in texts  # the IHDR payload is thirteen unprintable bytes
    assert ".PNG....rest of a second image" in texts  # the tail: another PNG


def test_a_comment_chunk_reads_in_the_preview(qtbot: QtBot, tmp_path: Path) -> None:
    path = _census_png(tmp_path, (_chunk(b"tEXt", b"Hint\x00look here"),))
    panel = _panel(qtbot, path)
    assert "Hint.look here" in _label_texts(panel)


def test_the_block_dump_can_be_searched(qtbot: QtBot, tmp_path: Path) -> None:
    """The block dump filters like the extract panel; the kinds are its domain test."""
    path = _census_png(tmp_path, (_chunk(b"tEXt", b"n33dle"),))
    panel = _panel(qtbot, path)
    report = panel._report
    assert report is not None
    text_row = next(index for index, block in enumerate(report.blocks) if block.label == "tEXt")
    panel._on_block_dumped(text_row)
    assert "n33dle" in panel._dump_view.text_pane.toPlainText()
    panel._dump_search.setText("n33dle")
    assert "n33dle" in panel._dump_view.text_pane.toPlainText()
    panel._dump_search.setText("no such bytes")
    assert panel._dump_view.hex_pane.toPlainText() == ""


def test_selecting_a_block_dumps_the_whole_chunk(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _png(tmp_path))  # 2x2 RGB: the IHDR chunk names the size
    panel._on_block_dumped(0)
    dump = panel._dump_view.hex_pane.toPlainText()
    # Length field, the IHDR type, the payload, then the CRC: the whole chunk.
    assert "00 00 00 0d 49 48 44 52 00 00 00 02 00 00 00 02" in dump
    assert panel._dump_view.hex_pane._rows[0].offset_text == "00000008"
    assert "块转储 · IHDR" in panel._dump_view._type_label.text()


def test_residual_data_dumps_with_a_guess_and_a_chip(qtbot: QtBot, tmp_path: Path) -> None:
    path = _png(tmp_path)
    path.write_bytes(path.read_bytes() + _ZIP_START + b"rest of archive")
    panel = _panel(qtbot, path)
    assert "残留数据" in _label_texts(panel)
    assert panel._warn.isHidden()  # the residual is a block now, not a warning
    assert panel._report is not None
    panel._on_block_dumped(len(panel._report.blocks) - 1)
    assert "50 4b 03 04" in panel._dump_view.hex_pane.toPlainText()
    assert "推测 zip" in panel._dump_view._type_label.text()
    chips = [button.text() for button in panel._dump_view.findChildren(QPushButton)]
    assert any(chip.startswith("ZIP @ 0x") for chip in chips)


def test_a_fake_idat_warns_and_shows_the_flag(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _fake_idat_png(tmp_path))
    assert "不会被渲染" in panel._warn.text()
    assert "疑似 flag" in panel._hits.text()


def test_clicking_a_block_name_renders_its_stream(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _png(tmp_path))
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._on_block_rendered(1, True)  # the IDAT: its stream is the real picture
    assert len(rendered) == 1
    image = rendered[0]
    assert (image.width, image.height) == (2, 2)
    assert [plane.name for plane in image.planes] == ["R", "G", "B"]


def test_a_stream_offers_one_radio_at_its_head(qtbot: QtBot, tmp_path: Path) -> None:
    """Three chunks, one picture: the radio belongs to the stream, not to each chunk."""
    panel = _panel(qtbot, _split_idat_png(tmp_path))
    # Rows: IHDR, the stream's two IDATs, IEND. Only the first IDAT carries a radio.
    assert sorted(panel._radios) == [0, 1, 3]
    assert panel._radios[1].isChecked()
    assert panel._radios[0].toolTip() == text.BLOCK_RENDER_TIP


def test_a_hard_scanline_chain_still_renders(qtbot: QtBot, tmp_path: Path) -> None:
    """Regression: a long Sub chain used to overflow int16 and kill the click."""
    path = tmp_path / "chain.png"
    ihdr = struct.pack(">IIBBBBB", 300, 1, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"\x01" + b"\xff" * 300))  # Sub: every byte adds
        + _chunk(b"IEND", b"")
    )
    panel = _panel(qtbot, path)
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._on_block_rendered(1, True)
    assert len(rendered) == 1


def test_a_failed_render_says_so_without_a_dialog(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _census_png(tmp_path, name="lace.png", interlace=1))
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._on_block_rendered(1, True)  # the IDAT: interlaced data is declined
    assert rendered == []
    assert text.RENDER_FAILED in panel._canvas_note.text()


def test_the_radio_renders_and_the_block_name_dumps(qtbot: QtBot, tmp_path: Path) -> None:
    """The two controls have one job each: the radio draws, the name shows bytes."""
    panel = _panel(qtbot, _fake_idat_png(tmp_path))  # rows: IHDR, IDAT, fake IDAT, IEND
    report = panel._report
    assert report is not None
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._name_buttons[0].click()  # the IHDR's name: the dump follows, the canvas does not
    assert rendered == []
    assert panel._selected == 0
    assert "块转储 · IHDR" in panel._dump_view._type_label.text()
    panel._radios[2].setChecked(True)  # the fake stream's radio: its pixels render
    assert len(rendered) == 1
    assert panel._canvas_note.text() == text.canvas_note(report.blocks[2], 1)
    assert "块转储 · IHDR" in panel._dump_view._type_label.text()  # the dump stayed put
    assert panel._selected == 0
    assert _is_shown(panel, 2)
    assert not _is_shown(panel, 1)  # the other stream is no longer the one on canvas


def test_a_fresh_file_checks_the_stream_it_shows_without_rendering(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Opening already shows stream 1; saying so must not re-render it."""
    image = load_image(_split_idat_png(tmp_path))
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show()
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel.set_image(image)  # the census runs with the spy watching
    assert rendered == []  # the default check is not a render
    assert panel._radios[1].isChecked()
    assert not panel._radios[0].isChecked()
    assert not panel._radios[3].isChecked()


def test_a_failed_render_leaves_the_radio_where_the_canvas_is(qtbot: QtBot, tmp_path: Path) -> None:
    """A stream that cannot be rendered must not leave the radio claiming it."""
    path = _census_png(
        tmp_path, (_chunk(b"IDAT", zlib.compress(b"\x00\xff")),), name="lace.png", interlace=1
    )
    panel = _panel(qtbot, path)
    report = panel._report
    assert report is not None
    assert [block.group for block in report.blocks if block.label == "IDAT"] == [1, 2]
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._radios[2].setChecked(True)  # the second stream: interlaced data is declined
    assert rendered == []
    assert text.RENDER_FAILED in panel._canvas_note.text()
    assert panel._radios[1].isChecked()  # back on the stream the canvas really shows
    assert not panel._radios[2].isChecked()


def test_a_render_keeps_the_census_and_the_selection(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fake_idat_png(tmp_path)
    panel = _panel(qtbot, path)
    panel._on_block_dumped(2)  # the fake IDAT picked for the dump
    rendered = image_from_rgb(path, np.zeros((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "pixelsb.ui.info.inspect_container",
        lambda _path: pytest.fail("a render re-scanned the file"),
    )
    panel.set_image(rendered)
    assert panel._selected == 2
    assert "块转储" in panel._dump_view._type_label.text()


def test_clicking_a_smuggled_stream_renders_the_smuggled_bytes(
    qtbot: QtBot, tmp_path: Path
) -> None:
    panel = _panel(qtbot, _fake_idat_png(tmp_path))
    rendered: list[LoadedImage] = []
    panel.render_requested.connect(rendered.append)
    panel._on_block_rendered(2, True)  # the fake IDAT: its own stream, its own pixels
    image = rendered[0]
    # The fake stream renders leniently: "f" reads as its filter byte, "l" as the pixel.
    assert int(image.samples[0, 0, 0]) == 0x6C


def test_a_fresh_file_marks_the_first_stream_as_shown(qtbot: QtBot, tmp_path: Path) -> None:
    """The canvas opens on the file itself: the census says so from the start."""
    panel = _panel(qtbot, _split_idat_png(tmp_path))
    report = panel._report
    assert report is not None
    assert panel._canvas_note.text() == text.canvas_note(report.blocks[1], 2)
    assert _is_shown(panel, 1)  # the stream's head, where its radio sits
    assert not _is_shown(panel, 2)
    assert not _is_shown(panel, 0)


def test_a_file_without_a_stream_claims_no_stream(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "case.bmp"  # a container the census does not read: no blocks
    Image.new("RGB", (2, 2)).save(path)
    panel = _panel(qtbot, path)
    assert panel._canvas_note.text() == text.original_note()
    assert panel._name_buttons == {}


def test_a_stitched_tail_is_dumped_and_guessed(qtbot: QtBot, tmp_path: Path) -> None:
    """A second PNG after the end is one residual block, not more structure."""
    path = _png(tmp_path)
    tail = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\x00" + bytes(6)))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(path.read_bytes() + tail)
    panel = _panel(qtbot, path)
    report = panel._report
    assert report is not None
    assert [block.label for block in report.blocks] == ["IHDR", "IDAT", "IEND", "残留数据"]
    panel._on_block_dumped(len(report.blocks) - 1)
    assert "89 50 4e 47 0d 0a 1a 0a" in panel._dump_view.hex_pane.toPlainText()
    assert "推测 png" in panel._dump_view._type_label.text()


def test_exif_tags_fill_the_table(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _png(tmp_path, exif=True))
    texts = _label_texts(panel)
    assert "Make" in texts
    assert "pixelsb" in texts
    assert text.INFO_NO_EXIF not in texts


def test_without_an_image_the_page_says_so(qtbot: QtBot) -> None:
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_image(None)
    assert panel._empty.isVisible()
    assert panel._empty.text() == text.NO_IMAGE
    assert not panel._form.isVisible()


def test_switching_images_leaves_no_stale_blocks(qtbot: QtBot, tmp_path: Path) -> None:
    panel = _panel(qtbot, _census_png(tmp_path, (_chunk(b"tEXt", b"Hint\x00first"),), "first.png"))
    panel.set_image(load_image(_fake_idat_png(tmp_path, "second.png")))
    texts = _label_texts(panel)
    assert "tEXt" not in texts  # the previous image's census is gone
    assert texts.count("IDAT") == 2


def test_the_same_image_is_not_read_twice(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = load_image(_png(tmp_path))
    panel = InfoPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_image(image)

    def exploding(_path: object) -> None:
        raise AssertionError("the page re-read an image it already holds")

    monkeypatch.setattr("pixelsb.ui.info.inspect_container", exploding)
    panel.set_image(image)

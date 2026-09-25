"""The extract panel's findings: the type note, the jump chips, and the tints."""

import pytest
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from pixelsb.domain.detect import Detection, detect_patterns
from pixelsb.domain.extract import format_extract
from pixelsb.domain.models import ExtractEncoding
from pixelsb.ui.extract_view import ExtractView, _row_span

_PNG = b"\x89PNG\r\n\x1a\n"


def _view(qtbot: QtBot, data: bytes) -> ExtractView:
    view = ExtractView()
    qtbot.addWidget(view)
    view.set_rows(format_extract(data), ExtractEncoding.ASCII)
    return view


def test_findings_name_the_type_and_build_one_chip_per_detection(qtbot: QtBot) -> None:
    view = _view(qtbot, b"flag{ok} " + _PNG)
    view.set_findings("类型 image/png · magic", detect_patterns(b"flag{ok} " + _PNG))
    chips = [button.text() for button in view.findChildren(QPushButton)]
    assert "flag{ok} @ 0x0" in chips
    assert "PNG @ 0x9" in chips
    assert view.hex_pane.extraSelections()


def test_findings_disappear_when_the_new_stream_has_none(qtbot: QtBot) -> None:
    view = _view(qtbot, b"flag{ok} " + _PNG)
    view.set_findings("类型 image/png · magic", detect_patterns(b"flag{ok} " + _PNG))
    view.set_rows(format_extract(b"\x00" * 32), ExtractEncoding.ASCII)
    view.set_findings("", ())
    assert not view._notice.isVisibleTo(view)
    assert not view.hex_pane.extraSelections()


def test_a_chip_click_selects_the_detection_bytes_in_the_dump(qtbot: QtBot) -> None:
    data = _PNG + b"flag{ok}"
    view = _view(qtbot, data)
    view.set_findings("", detect_patterns(data))
    chip = next(
        button for button in view.findChildren(QPushButton) if button.text().startswith("flag")
    )
    chip.click()
    assert view.hex_pane.textCursor().selectedText() == "66 6c 61 67 7b 6f 6b 7d"


def test_a_detection_filtered_out_of_the_dump_has_nothing_to_jump_to(qtbot: QtBot) -> None:
    data = _PNG + b"\x00" * 8 + b"flag{ok}"  # the flag lands in the second dump row
    view = _view(qtbot, data)
    rows = [row for row in format_extract(data) if row.offset == 0]
    view.set_rows(rows, ExtractEncoding.ASCII)
    view.set_findings("", detect_patterns(data))
    assert len(view.hex_pane.extraSelections()) == 1  # only the PNG header's row
    view._reveal(Detection(24, "flag{ok}", 8, flagged=True))
    assert view.hex_pane.textCursor().selectedText() == ""


def test_the_text_pane_reads_a_character_split_by_a_row_boundary(qtbot: QtBot) -> None:
    """The pane decodes the rows in sequence, so a comment survives 16-byte rows."""
    view = ExtractView()
    qtbot.addWidget(view)
    view.set_rows(format_extract(b"A" * 15 + "中".encode() + b"BC"), ExtractEncoding.UTF8)
    assert view.text_pane.toPlainText().splitlines() == ["A" * 15, "中BC"]


def test_highlights_survive_a_search_that_keeps_the_row(qtbot: QtBot) -> None:
    data = b"flag{ok}"
    view = _view(qtbot, data)
    view.set_findings("", detect_patterns(data))
    view.set_rows(format_extract(data), ExtractEncoding.ASCII)  # same rows, rebuilt document
    assert len(view.hex_pane.extraSelections()) == 1


@pytest.mark.parametrize(
    ("offset", "row", "column"),
    [(0, 0, 0), (15, 0, 15), (16, 1, 0), (33, 2, 1)],
)
def test_spans_map_stream_offsets_onto_dump_rows(offset: int, row: int, column: int) -> None:
    rows = tuple(format_extract(b"\x00" * 64))
    assert _row_span(rows, Detection(offset, "x", 1)) == (row, column, 1, False)


def test_a_span_past_the_rows_has_nothing(qtbot: QtBot) -> None:
    view = _view(qtbot, b"\x00" * 16)
    assert _row_span(view._rows, Detection(16, "x", 1)) is None

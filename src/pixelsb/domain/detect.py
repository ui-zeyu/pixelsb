"""Patterns worth flagging inside an extracted byte stream."""

import re
from dataclasses import dataclass
from itertools import islice

from pixelsb.domain.formatting import clip

_MAGIC: tuple[tuple[str, bytes], ...] = (
    ("PNG", b"\x89PNG\r\n\x1a\n"),
    ("JPEG", b"\xff\xd8\xff"),
    ("GIF", b"GIF8"),
    ("ZIP", b"PK\x03\x04"),
    ("RAR", b"Rar!\x1a\x07"),
    ("7Z", b"7z\xbc\xaf\x27\x1c"),
    ("PDF", b"%PDF-"),
    ("GZIP", b"\x1f\x8b"),
    ("ELF", b"\x7fELF"),
)
# flag{...}, ctf{...}, and the competition-prefixed spellings (DASCTF{...} and
# friends all carry "ctf" or "flag" inside the word). A bytes pattern keeps the
# scan ASCII-only, which is what a flag in a binary stream is.
_FLAG = re.compile(rb"\w*(?:flag|ctf)\w*\{[^}\n\r]{1,100}\}", re.IGNORECASE)
_CHIP_LABEL = 20
MAX_DETECTIONS = 128


@dataclass(frozen=True, slots=True)
class Detection:
    """One find: a file signature or a flag-shaped run, at its stream offset.

    ``label`` is what the chip says and is empty for a mark the dump only
    tints — a chunk's own header or CRC, which is a span rather than a find.
    """

    offset: int
    label: str = ""
    length: int = 0
    flagged: bool = False

    @property
    def chip(self) -> str:
        return f"{self.label} @ 0x{self.offset:x}"


def detect_patterns(data: bytes) -> tuple[Detection, ...]:
    """Every magic header and flag-shaped string in the stream, in stream order.

    The count is capped, so a pathological stream cannot flood the panel; the
    chips and the row highlights both come straight from this tuple.
    """
    magic = (
        Detection(offset, label, len(phrase))
        for label, phrase in _MAGIC
        for offset in _hits(data, phrase)
    )
    flags = (
        Detection(match.start(), _flag_label(match.group()), len(match.group()), flagged=True)
        for match in _FLAG.finditer(data)
    )
    found = sorted((*magic, *flags), key=lambda finding: (finding.offset, finding.flagged))
    return tuple(islice(found, MAX_DETECTIONS))


def _hits(data: bytes, phrase: bytes) -> list[int]:
    positions, start = [], 0
    while (hit := data.find(phrase, start)) != -1:
        positions.append(hit)
        start = hit + 1
    return positions


def _flag_label(flag: bytes) -> str:
    return clip(flag.decode("ascii", "replace"), _CHIP_LABEL)

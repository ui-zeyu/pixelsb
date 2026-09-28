"""Patterns worth flagging inside an extracted byte stream."""

from collections.abc import Iterator
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
# The keywords every flag format spells: flag, ctf, key, password, secret, read
# case-insensitively (the competition prefixes DASCTF{...} and friends carry
# "ctf" inside the word). Finding the keyword is all the scan promises — telling
# a real flag from a chance run of letters is the user's next step. Plain
# ``find`` is also what keeps the sweep fast: a regex over the whole stream once
# backtracked itself quadratic on a constant letter run, which is exactly what
# a constant bit plane extracts to, and froze sweeps on a single candidate.
_KEYWORDS = (b"flag", b"ctf", b"key", b"password", b"secret")
MAX_DETECTIONS = 128


@dataclass(frozen=True, slots=True)
class Detection:
    """One find: a file signature or a keyword hit, at its stream offset.

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
    """Every magic header and keyword hit in the stream, in stream order.

    The count is capped, so a pathological stream cannot flood the panel; the
    chips and the row highlights both come straight from this tuple.
    """
    magic = (
        Detection(offset, label, len(phrase))
        for label, phrase in _MAGIC
        for offset in _occurrences(data, phrase)
    )
    found = sorted(
        (*magic, *_keywords(data)), key=lambda finding: (finding.offset, finding.flagged)
    )
    return tuple(islice(found, MAX_DETECTIONS))


def keyword_hits(data: bytes) -> tuple[Detection, ...]:
    """Every keyword hit, in stream order, under the same cap as a full sweep.

    The sweep judges candidates by their keywords alone; the signature half of
    :func:`detect_patterns` would spend nine full passes per candidate on
    findings it then throws away.
    """
    return tuple(islice(sorted(_keywords(data), key=lambda hit: hit.offset), MAX_DETECTIONS))


def _keywords(data: bytes) -> Iterator[Detection]:
    """Every keyword occurrence, case-insensitively, as a flagged detection."""
    lowered = data.lower()
    for word in _KEYWORDS:
        for hit in _occurrences(lowered, word):
            yield Detection(hit, _chip_text(data[hit : hit + len(word)]), len(word), flagged=True)


def _occurrences(data: bytes, phrase: bytes) -> Iterator[int]:
    """Where ``phrase`` sits in ``data``, every occurrence, left to right."""
    start = 0
    while (hit := data.find(phrase, start)) != -1:
        yield hit
        start = hit + 1


def _chip_text(phrase: bytes) -> str:
    return clip(phrase.decode("ascii", "replace"), 20)

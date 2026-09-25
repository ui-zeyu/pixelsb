"""Bytes read as text: the printable tables, and the preview a dump row shows.

Two tables answer two different questions. A dump's ASCII column says which
bytes *are* text, so everything outside the printable ASCII range becomes a dot.
A census preview says what a block *holds*, and a comment chunk written in
UTF-8 spells its text in bytes above 0x7F: those are kept as the characters
latin-1 makes of them, so 邮戳 reads as its own bytes rather than a row of dots.
"""

_ACCENTED = 0xA0  # latin-1's printable range resumes above this point
_DOT = 0x2E

ASCII_TABLE = bytes(byte if 0x20 <= byte <= 0x7E else _DOT for byte in range(256))
LATIN_TABLE = bytes(
    byte if 0x20 <= byte <= 0x7E or byte >= _ACCENTED else _DOT for byte in range(256)
)


def as_text(data: bytes, table: bytes = ASCII_TABLE) -> str:
    """``data`` with the bytes ``table`` dots out replaced; never raises."""
    return data.translate(table).decode("latin-1")


def preview(data: bytes, limit: int = 32, table: bytes = LATIN_TABLE) -> str:
    """The first ``limit`` bytes as printable text, marked when it is cut short."""
    return as_text(data[:limit], table) + ("…" if len(data) > limit else "")

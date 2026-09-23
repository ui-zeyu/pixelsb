"""Decimal, hexadecimal, and binary formatting for channel samples."""

from pixelsb.domain.models import DisplayFormat


def format_sample(value: int, bit_depth: int, fmt: DisplayFormat) -> str:
    _check_sample(value, bit_depth)
    match fmt:
        case DisplayFormat.DECIMAL:
            return str(value)
        case DisplayFormat.HEX:
            digits = max(1, (bit_depth + 3) // 4)
            return f"{value:0{digits}X}"
        case DisplayFormat.BINARY:
            return f"{value:0{bit_depth}b}"
        case _:
            raise ValueError(f"unknown format: {fmt}")


def format_delta(delta: int, bit_depth: int, fmt: DisplayFormat) -> str:
    """Format a signed channel delta. Zero is ``0``; every other value keeps its sign."""
    if delta == 0:
        return "0"
    sign = "+" if delta > 0 else "-"
    return sign + format_sample(abs(delta), bit_depth, fmt)


def binary_bit_index(bit_depth: int, bit: int) -> int:
    """Index of ``bit`` in an MSB-first binary string. Bit 0 is the last character."""
    if not 1 <= bit_depth <= 256:
        raise ValueError("bit depth must be from 1 to 256")
    if not 0 <= bit < bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{bit_depth - 1}")
    return bit_depth - 1 - bit


def _check_sample(value: int, bit_depth: int) -> None:
    if not 1 <= bit_depth <= 256:
        raise ValueError("bit depth must be from 1 to 256")
    if not 0 <= value < (1 << bit_depth):
        raise ValueError(f"value {value} does not fit in {bit_depth} bits")

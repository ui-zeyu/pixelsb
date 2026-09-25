"""Formatting for display: channel samples as numbers, long text as a prefix."""

from pixelsb.domain.models import DisplayFormat


def clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, the cut marked with an ellipsis."""
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


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


def _check_sample(value: int, bit_depth: int) -> None:
    if not 1 <= bit_depth <= 256:
        raise ValueError("bit depth must be from 1 to 256")
    if not 0 <= value < (1 << bit_depth):
        raise ValueError(f"value {value} does not fit in {bit_depth} bits")

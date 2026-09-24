import pytest

from pixelsb.domain.formatting import format_sample
from pixelsb.domain.models import DisplayFormat


def test_format_sample_widths() -> None:
    assert format_sample(0, 8, DisplayFormat.DECIMAL) == "0"
    assert format_sample(1, 8, DisplayFormat.DECIMAL) == "1"
    assert format_sample(255, 8, DisplayFormat.DECIMAL) == "255"
    assert format_sample(255, 8, DisplayFormat.HEX) == "FF"
    assert format_sample(1, 8, DisplayFormat.BINARY) == "00000001"
    assert format_sample(65535, 16, DisplayFormat.HEX) == "FFFF"
    assert format_sample(1, 1, DisplayFormat.BINARY) == "1"


def test_format_rejects_values_outside_the_bit_depth() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        format_sample(256, 8, DisplayFormat.HEX)
    with pytest.raises(ValueError, match="does not fit"):
        format_sample(-1, 8, DisplayFormat.DECIMAL)

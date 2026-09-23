import numpy as np
import pytest

from pixelsb.domain.models import BitChoice
from pixelsb.domain.predicate import PredicateError, compile_filter, field_names
from tests.support import make_image, planes_rgb

_SAMPLES = np.array(
    [
        [[200, 100, 50], [0, 0, 200]],
        [[10, 10, 10], [0, 255, 0]],
    ],
    dtype=np.uint16,
)


def _image():
    return make_image(_SAMPLES, planes_rgb())


def _match(expression: str, chosen: frozenset[BitChoice] | None = None):
    return compile_filter(expression, planes_rgb()).evaluate(_image(), chosen)


def test_comparisons_and_logic() -> None:
    match = _match("B >= R and B >= G")
    assert match.tolist() == [[False, True], [True, False]]


def test_arithmetic_and_chained_comparison() -> None:
    assert _match("0 < R - G <= 100").tolist() == [[True, False], [False, False]]
    assert _match("R == 200 or R == 10").tolist() == [[True, False], [True, False]]
    assert _match("not (R > 0)").tolist() == [[False, True], [False, True]]


def test_coordinates_and_aliases() -> None:
    assert _match("left == 0 and top == 1").tolist() == [[False, False], [True, False]]
    assert _match("x == 1 and y == 0").tolist() == [[False, True], [False, False]]
    assert _match("left >= 1").tolist() == [[False, True], [False, True]]


def test_channel_values_follow_the_selection() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    assert _match("B == 0", low_bit).all()
    assert not _match("B == 1", low_bit).any()
    every_bit = frozenset(BitChoice("B", bit) for bit in range(8))
    assert _match("B == 50", every_bit).tolist() == [[True, False], [False, False]]
    assert _match("B == 50").tolist() == [[True, False], [False, False]]
    assert _match("B == 0", None).tolist() == [[False, False], [False, True]]


def test_identifiers_are_case_insensitive() -> None:
    assert _match("b >= r and b >= g").tolist() == [[False, True], [True, False]]


def test_bare_numeric_result_is_read_as_non_zero() -> None:
    assert _match("B - B").tolist() == [[False, False], [False, False]]
    assert _match("B").tolist() == [[True, True], [True, False]]


def test_errors_name_the_problem() -> None:
    with pytest.raises(PredicateError, match="语法错误"):
        compile_filter("B >=", planes_rgb())
    with pytest.raises(PredicateError, match="未知字段"):
        compile_filter("A > 0", planes_rgb())
    with pytest.raises(PredicateError, match="left"):
        compile_filter("foo == 1", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的表达式元素"):
        compile_filter("len(R) > 0", planes_rgb())


def test_field_names_list_aliases_and_planes() -> None:
    names = field_names(planes_rgb())
    assert names["x"] == "left"
    assert names["y"] == "top"
    assert names["b"] == "B"

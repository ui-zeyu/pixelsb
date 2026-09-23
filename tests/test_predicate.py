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


def test_coordinates_cover_both_axes() -> None:
    assert _match("left == 0 and top == 1").tolist() == [[False, False], [True, False]]
    assert _match("left >= 1").tolist() == [[False, True], [False, True]]
    assert _match("top == 0").tolist() == [[True, True], [False, False]]


def test_removed_aliases_are_unknown_fields() -> None:
    for expression in ("x == 1", "y == 0", "up == 1"):
        with pytest.raises(PredicateError, match="未知字段"):
            compile_filter(expression, planes_rgb())


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


def test_rect_selects_an_inclusive_rectangle() -> None:
    assert _match("rect(0, 1, 0, 1)").tolist() == [[False, False], [True, False]]
    assert _match("rect(0, 0, 1, 0)").tolist() == [[True, True], [False, False]]
    assert _match("rect(0, 0, 5, 5)").all()
    assert not _match("rect(2, 2, 3, 3)").any()


def test_rect_swaps_bounds_and_combines_with_conditions() -> None:
    assert _match("rect(1, 0, 1, 0)").tolist() == [[False, True], [False, False]]
    assert _match("rect(0, 0, 0, 0) or rect(1, 1, 1, 1)").tolist() == [
        [True, False],
        [False, True],
    ]
    assert _match("rect(0, 0, 1, 1) and B >= R and B >= G").tolist() == [
        [False, True],
        [True, False],
    ]
    assert _match("rect(0, 0, 0, 1) and B >= R and B >= G").tolist() == [
        [False, False],
        [True, False],
    ]


def test_rect_edges_can_be_named() -> None:
    positional = "rect(0, 1, 1, 1)"
    named = "rect(left=0, top=1, right=1, bottom=1)"
    assert _match(named).tolist() == _match(positional).tolist()
    assert _match(named).tolist() == [[False, False], [True, True]]
    swapped = _match("rect(left=1, top=1, right=0, bottom=1)")
    assert swapped.tolist() == [[False, False], [True, True]]


def test_rect_errors_are_clear() -> None:
    with pytest.raises(PredicateError, match="4 个坐标"):
        compile_filter("rect(0, 0, 1)", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的函数"):
        compile_filter("circle(0, 0, 1, 1)", planes_rgb())
    with pytest.raises(PredicateError, match="标量"):
        _match("rect(left, 0, 1, 1)")
    with pytest.raises(PredicateError, match="不能混用"):
        compile_filter("rect(0, 0, right=1, bottom=1)", planes_rgb())
    with pytest.raises(PredicateError, match="参数名只能是"):
        compile_filter("rect(left=0, top=0, right=1, width=1)", planes_rgb())
    with pytest.raises(PredicateError, match="缺少参数"):
        compile_filter("rect(left=0, top=0, right=1)", planes_rgb())


def test_errors_name_the_problem() -> None:
    with pytest.raises(PredicateError, match="语法错误"):
        compile_filter("B >=", planes_rgb())
    with pytest.raises(PredicateError, match="未知字段"):
        compile_filter("A > 0", planes_rgb())
    with pytest.raises(PredicateError, match="left"):
        compile_filter("foo == 1", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的函数"):
        compile_filter("len(R) > 0", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的表达式元素"):
        compile_filter("[R] == 1", planes_rgb())


def test_field_names_hold_one_name_per_field() -> None:
    names = field_names(planes_rgb())
    assert set(names.values()) == {"left", "top", "R", "G", "B"}
    assert names["b"] == "B"
    assert "x" not in names
    assert "up" not in names

import numpy as np
import pytest
from numpy.typing import NDArray

from pixelsb.domain import predicate
from pixelsb.domain.models import BitChoice, LoadedImage, SampleArray, SamplePlane
from pixelsb.domain.predicate import PredicateError, compile_filter, field_names
from tests.support import make_image, planes_rgb

_SAMPLES = np.array(
    [
        [[200, 100, 50], [0, 0, 200]],
        [[10, 10, 10], [0, 255, 0]],
    ],
    dtype=np.uint16,
)


def _image() -> LoadedImage:
    return make_image(_SAMPLES, planes_rgb())


def _match(expression: str, chosen: frozenset[BitChoice] | None = None) -> NDArray[np.bool_]:
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


def test_raw_attribute_ignores_the_selection() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    assert _match("B == 0", low_bit).all()
    assert _match("B.raw == 50", low_bit).tolist() == [[True, False], [False, False]]
    assert _match("B.raw == 0", low_bit).tolist() == [[False, False], [False, True]]
    assert _match("B.raw == 50", frozenset()).tolist() == [[True, False], [False, False]]
    assert _match("B.raw == 50").tolist() == [[True, False], [False, False]]


def test_bits_attribute_matches_the_bare_name() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    assert _match("B.bits == B", low_bit).all()
    assert _match("B.RAW >= 200", low_bit).tolist() == [[False, True], [False, False]]
    assert _match("B.raw > B.bits", low_bit).tolist() == [[True, True], [True, False]]


def test_identifier_errors_name_the_problem() -> None:
    with pytest.raises(PredicateError, match="属性只能是"):
        compile_filter("B.value > 0", planes_rgb())
    with pytest.raises(PredicateError, match="没有属性"):
        compile_filter("left.raw == 0", planes_rgb())
    with pytest.raises(PredicateError, match="未知字段"):
        compile_filter("A.raw == 0", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的表达式元素"):
        compile_filter("R.raw.raw == 1", planes_rgb())


def test_identifiers_are_case_insensitive() -> None:
    assert _match("b >= r and b >= g").tolist() == [[False, True], [True, False]]


def test_bare_numeric_result_is_read_as_non_zero() -> None:
    assert _match("B - B").tolist() == [[False, False], [False, False]]
    assert _match("B").tolist() == [[True, True], [True, False]]


def test_rect_selects_a_box() -> None:
    assert _match("rect(0, 0, 1, 1)").tolist() == [[True, False], [False, False]]
    assert _match("rect(0, 1, 1, 2)").tolist() == [[False, False], [True, False]]
    assert _match("rect(0, 0, 2, 2)").all()
    assert not _match("rect(2, 2, 3, 3)").any()


def test_rect_swaps_bounds_and_combines_with_conditions() -> None:
    assert _match("rect(1, 0, 2, 1)").tolist() == [[False, True], [False, False]]
    assert _match("rect(0, 0, 1, 1) or rect(1, 1, 2, 2)").tolist() == [
        [True, False],
        [False, True],
    ]
    assert _match("rect(0, 0, 2, 2) and B >= R and B >= G").tolist() == [
        [False, True],
        [True, False],
    ]
    assert _match("rect(0, 0, 1, 2) and B >= R and B >= G").tolist() == [
        [False, False],
        [True, False],
    ]


def test_rect_edges_can_be_named() -> None:
    positional = _match("rect(0, 1, 1, 2)")
    named = _match("rect(left=0, top=1, right=1, bottom=2)")
    assert named.tolist() == positional.tolist()
    assert named.tolist() == [[False, False], [True, False]]
    swapped = _match("rect(left=1, top=2, right=0, bottom=1)")
    assert swapped.tolist() == [[False, False], [True, False]]


def test_edge_fields_match_the_rect_form() -> None:
    assert _match("right == 1").tolist() == [[True, False], [True, False]]
    assert _match("bottom == 2").tolist() == [[False, False], [True, True]]
    edges = "left >= 1 and right <= 2 and top >= 0 and bottom <= 1"
    assert _match(edges).tolist() == _match("rect(1, 0, 2, 1)").tolist()


def test_a_filter_reports_the_fields_it_reads() -> None:
    def fields(expression: str) -> set[str]:
        return set(compile_filter(expression, planes_rgb()).fields)

    assert fields("rect(0, 0, 1, 1)") == {"left", "top", "right", "bottom"}
    assert fields("grid(0, 0, 2, 2)") == {"left", "top"}
    assert fields("R.raw > 1") == {"R.raw"}
    assert fields("R.bits + R") == {"R", "R.bits"}
    assert fields("B >= R and G < 100") == {"R", "G", "B"}
    # A call's arguments are read; the function name is not a field of its own.
    assert fields("rect(0, 0, 1, R.raw)") == {"left", "top", "right", "bottom", "R.raw"}


def test_a_coordinate_filter_never_builds_a_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    read = predicate._raw_values

    def spy(samples: SampleArray, plane: SamplePlane) -> NDArray[np.uint32]:
        seen.append(plane.name)
        return read(samples, plane)

    monkeypatch.setattr(predicate, "_raw_values", spy)
    assert _match("rect(0, 0, 1, 1)").tolist() == [[True, False], [False, False]]
    assert _match("grid(0, 0, 1, 1) and top == 0").tolist() == [[True, True], [False, False]]
    assert seen == []


def test_each_channel_is_built_once_per_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    read = predicate._raw_values

    def spy(samples: SampleArray, plane: SamplePlane) -> NDArray[np.uint32]:
        seen.append(plane.name)
        return read(samples, plane)

    monkeypatch.setattr(predicate, "_raw_values", spy)
    _match("R.raw > B.raw and B.bits > 0 and B > 1")
    # R.raw and B.raw need the stored values; both B flavours share the one copy.
    assert sorted(seen) == ["B", "R"]
    assert _match("0 <= left <= 0").tolist() == [[True, False], [True, False]]
    assert not _match("50 <= left <= 100").any()


def test_rect_errors_are_clear() -> None:
    with pytest.raises(PredicateError, match="4 个参数"):
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


def test_grid_steps_from_the_anchor() -> None:
    assert _match("grid(0, 0, 1, 1)").all()
    assert _match("grid(0, 0, 2, 2)").tolist() == [[True, False], [False, False]]
    assert _match("grid(0, 0, 1, 2)").tolist() == [[True, True], [False, False]]
    assert _match("grid(0, 0, 2, 1)").tolist() == [[True, False], [True, False]]
    assert _match("grid(1, 0, 1, 2)").tolist() == [[False, True], [False, False]]
    assert _match("grid(0, 1, 2, 1)").tolist() == [[False, False], [True, False]]


def test_grid_anchor_excludes_earlier_pixels() -> None:
    assert _match("grid(1, 1, 1, 1)").tolist() == [[False, False], [False, True]]
    assert not _match("grid(2, 2, 1, 1)").any()
    assert _match("grid(0, 0, 100, 100)").tolist() == [[True, False], [False, False]]


def test_grid_combines_with_other_conditions() -> None:
    match = _match("grid(0, 0, 1, 2) and B >= R and B >= G")
    assert match.tolist() == [[False, True], [False, False]]
    assert _match("rect(0, 0, 2, 2) and grid(1, 0, 1, 1)").tolist() == [
        [False, True],
        [False, True],
    ]


def test_grid_args_can_be_named_or_expressions() -> None:
    positional = _match("grid(1, 0, 1, 2)")
    assert _match("grid(x=1, y=0, step_x=1, step_y=2)").tolist() == positional.tolist()
    assert _match("grid(2 - 1, 0, 1, 4 / 2)").tolist() == positional.tolist()


def test_grid_errors_are_clear() -> None:
    # Argument count/name errors surface at compile time; step and anchor
    # errors surface at evaluation, like rect's scalar check.
    with pytest.raises(PredicateError, match="4 个参数"):
        compile_filter("grid(0, 0, 1)", planes_rgb())
    with pytest.raises(PredicateError, match="参数名只能是"):
        compile_filter("grid(x=0, y=0, step_x=1, stride=1)", planes_rgb())
    with pytest.raises(PredicateError, match="缺少参数"):
        compile_filter("grid(x=0, y=0, step_x=1)", planes_rgb())
    with pytest.raises(PredicateError, match="步长"):
        _match("grid(0, 0, 0, 1)")
    with pytest.raises(PredicateError, match="步长"):
        _match("grid(0, 0, -1, 1)")
    with pytest.raises(PredicateError, match="起点"):
        _match("grid(-1, 0, 1, 1)")
    with pytest.raises(PredicateError, match="标量"):
        _match("grid(left, 0, 1, 1)")


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
    assert set(names.values()) == {"left", "top", "right", "bottom", "R", "G", "B"}
    assert names["b"] == "B"
    assert "x" not in names
    assert "up" not in names
    assert "bottom" in names

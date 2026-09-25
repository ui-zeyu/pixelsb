import numpy as np
import pytest
from numpy.typing import NDArray

from pixelsb.domain import predicate
from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    LoadedImage,
    RegionMask,
    SampleArray,
    SamplePlane,
)
from pixelsb.domain.predicate import Field, PredicateError, compile_filter, field_names
from tests.support import make_image, planes_rgb, raster

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
    """The mask an expression makes of the sample image, over that bit selection."""
    stack = () if chosen is None else (BitsMask(chosen),)
    return compile_filter(expression, planes_rgb()).evaluate(raster(_image(), *stack))


def test_comparisons_and_logic() -> None:
    match = _match("B >= R and B >= G")
    assert match.tolist() == [[False, True], [True, False]]


def test_arithmetic_and_chained_comparison() -> None:
    assert _match("0 < R - G <= 100").tolist() == [[True, False], [False, False]]
    assert _match("R == 200 or R == 10").tolist() == [[True, False], [True, False]]
    assert _match("not (R > 0)").tolist() == [[False, True], [False, True]]


def test_boolean_operands_commute_over_the_same_pixels() -> None:
    """Both operands read the one incoming raster, so the order cannot matter."""
    forward = _match("r > 0 and b.1")
    assert forward.tolist() == [[True, False], [True, False]]
    assert np.array_equal(forward, _match("b.1 and r > 0"))
    assert np.array_equal(forward, _match("not (not r > 0 or not b.1)"))


def test_coordinates_cover_both_axes() -> None:
    assert _match("left == 0 and top == 1").tolist() == [[False, False], [True, False]]
    assert _match("left >= 1").tolist() == [[False, True], [False, True]]


def test_coordinates_name_the_source_pixels_a_cropped_raster_came_from() -> None:
    cropped = raster(_image(), RegionMask("left >= 1"), CropMask())
    filter_ = compile_filter("left == 1", planes_rgb())
    assert filter_.evaluate(cropped).tolist() == [[True], [True]]
    assert compile_filter("left == 0", planes_rgb()).evaluate(cropped).tolist() == [
        [False],
        [False],
    ]
    assert _match("top == 0").tolist() == [[True, True], [False, False]]


def test_removed_aliases_are_unknown_fields() -> None:
    for expression in ("x == 1", "y == 0", "up == 1"):
        with pytest.raises(PredicateError, match="未知字段"):
            compile_filter(expression, planes_rgb())


def test_a_bare_channel_is_the_value_as_stored() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    # The selection drives the canvas and the dump, not a bare channel name.
    assert _match("B == 50", low_bit).tolist() == [[True, False], [False, False]]
    assert _match("B == 0", low_bit).tolist() == [[False, False], [False, True]]
    assert not _match("B == 1", low_bit).any()
    assert _match("B == 50", frozenset()).tolist() == [[True, False], [False, False]]


def test_bits_attribute_is_the_value_of_the_selection() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    assert _match("B.bits == 0", low_bit).all()
    assert not _match("B.bits == 1", low_bit).any()
    every_bit = frozenset(BitChoice("B", bit) for bit in range(8))
    assert _match("B.bits == 50", every_bit).tolist() == [[True, False], [False, False]]
    assert _match("B.bits == 50").tolist() == [[True, False], [False, False]]
    assert _match("B.bits == 0", None).tolist() == [[False, False], [False, True]]


def test_bits_and_the_bare_name_are_different_values() -> None:
    low_bit = frozenset({BitChoice("B", 0)})
    # B = 50, 200, 10, 0: bit 0 of each is 0, 0, 0, 0, so only the last agrees.
    assert _match("B.bits == B", low_bit).tolist() == [[False, False], [False, True]]
    assert _match("B.bits <= B", low_bit).all()


def test_a_bit_attribute_reads_one_bit_of_the_channel() -> None:
    # B = 50, 200, 10, 0 and G = 100, 0, 10, 255.
    assert _match("B.1 == 1").tolist() == [[True, False], [True, False]]
    assert _match("B.3 == 1").tolist() == [[False, True], [True, False]]
    assert _match("B.5 == 1").tolist() == [[True, False], [False, False]]
    assert _match("G.0 == 1").tolist() == [[False, False], [False, True]]
    assert _match("G.7 == 1 or B.7 == 1").tolist() == [[False, True], [False, True]]


def test_the_bit_attributes_add_back_up_to_the_stored_value() -> None:
    weighted = " + ".join(f"B.{bit} * {1 << bit}" for bit in range(8))
    assert _match(f"{weighted} == B").all()


def test_a_bit_past_the_channels_depth_is_rejected() -> None:
    with pytest.raises(PredicateError, match="只有 8 位"):
        compile_filter("R.8 == 1", planes_rgb())
    deep = tuple(SamplePlane(plane.name, plane.index, 16, plane.origin) for plane in planes_rgb())
    assert compile_filter("R.15 == 1", deep).fields == {"R.15"}


def test_identifier_errors_name_the_problem() -> None:
    with pytest.raises(PredicateError, match="属性只能是"):
        compile_filter("B.value > 0", planes_rgb())
    with pytest.raises(PredicateError, match="没有属性"):
        compile_filter("left.0 == 0", planes_rgb())
    with pytest.raises(PredicateError, match="未知字段"):
        compile_filter("A.0 == 0", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的表达式元素"):
        compile_filter("R.3.bits == 1", planes_rgb())


def test_a_bit_index_is_marked_up_for_the_parser() -> None:
    assert predicate._with_bit_attributes("R.3 == 1") == "R._3 == 1"
    assert predicate._with_bit_attributes("R.0\nB.1") == "R._0\nB._1"
    # Nothing to mark: a float after a keyword or a space is left alone.
    assert predicate._with_bit_attributes("R and .5") == "R and .5"
    assert predicate._with_bit_attributes("left > .5") == "left > .5"
    assert predicate._with_bit_attributes("R .5") == "R .5"


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
    assert fields("R > 1") == {"R"}
    assert fields("R.bits + R") == {"R", "R.bits"}
    assert fields("R.3 + B.7") == {"R.3", "B.7"}
    assert fields("B >= R and G < 100") == {"R", "G", "B"}
    # A call's arguments are read; the function name is not a field of its own.
    assert fields("rect(0, 0, 1, R.7)") == {"left", "top", "right", "bottom", "R.7"}


def test_a_coordinate_filter_never_builds_a_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    read = predicate._stored_values

    def spy(samples: SampleArray, plane: SamplePlane) -> NDArray[np.uint32]:
        seen.append(plane.name)
        return read(samples, plane)

    monkeypatch.setattr(predicate, "_stored_values", spy)
    assert _match("rect(0, 0, 1, 1)").tolist() == [[True, False], [False, False]]
    assert _match("grid(0, 0, 1, 1) and top == 0").tolist() == [[True, True], [False, False]]
    assert seen == []


def test_each_channel_is_built_once_per_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    read = predicate._stored_values

    def spy(samples: SampleArray, plane: SamplePlane) -> NDArray[np.uint32]:
        seen.append(plane.name)
        return read(samples, plane)

    monkeypatch.setattr(predicate, "_stored_values", spy)
    _match("R > B.7 and B.bits > 0 and B.3 > 1 and B > 1")
    # Every flavour of B, bit fields included, shares the one copy of the channel.
    assert sorted(seen) == ["B", "R"]


def test_a_bit_only_filter_never_builds_the_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def spy(
        _channel: NDArray[np.uint32],
        plane: str,
        _chosen: frozenset[BitChoice] | None,
    ) -> None:
        seen.append(plane)

    monkeypatch.setattr(predicate, "_selected_values", spy)
    assert _match("R.0 == 1 or B.7 == 0").tolist() == [[True, False], [True, True]]
    assert seen == []
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
    # A bit index glued to a name is marked up for the parser, one with a space
    # before it is not, and neither is a number that only follows a name.
    with pytest.raises(PredicateError, match="语法错误"):
        compile_filter("R .5", planes_rgb())
    assert _match("R and .5").tolist() == [[True, False], [True, False]]
    with pytest.raises(PredicateError, match="未知字段"):
        compile_filter("A > 0", planes_rgb())
    with pytest.raises(PredicateError, match="left"):
        compile_filter("foo == 1", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的函数"):
        compile_filter("len(R) > 0", planes_rgb())
    with pytest.raises(PredicateError, match="不支持的表达式元素"):
        compile_filter("[R] == 1", planes_rgb())


def test_field_names_hold_one_field_per_name() -> None:
    names = field_names(planes_rgb())
    assert {name.name for name in names.values()} == {
        "left",
        "top",
        "right",
        "bottom",
        "R",
        "G",
        "B",
    }
    assert names["b"] == Field("B", 8)
    # The coordinate fields take no attributes: they hold no bits.
    assert names["left"] == Field("left")
    assert "x" not in names
    assert "up" not in names

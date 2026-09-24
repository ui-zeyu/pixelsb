"""tshark-style display filter over pixel fields, compiled to vector operations.

Fields: ``left``, ``top``, ``right``, ``bottom`` (a pixel occupies
``[left, right) x [top, bottom)``), and the channel names. A bare channel name is
the current selection masked onto the plane, and channels without any selected bit
read as 0; ``R.raw`` is the channel as stored, ``R.bits`` spells the selection
value out. The expression is parsed with :mod:`ast` and every node is compiled
into a numpy closure — strings are never evaluated.
"""

import ast
from collections.abc import Callable
from functools import reduce
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import BitChoice, LoadedImage, SampleArray, SamplePlane
from pixelsb.domain.selection import bits_for

type Value = NDArray[Any]
type FieldEnv = dict[str, Value]
type Evaluator = Callable[[FieldEnv], Value]
type BinaryOp = Callable[[Value, Value], Value]
type CompareOp = Callable[[Value, Value], Value]


_RECT = "rect"
_RECT_EDGES = ("left", "top", "right", "bottom")
_GRID = "grid"
_GRID_PARAMS = ("x", "y", "step_x", "step_y")
_FUNCTIONS = (_RECT, _GRID)
_BITS = "bits"
_RAW = "raw"
# Attributes a channel name accepts: the selection value, and the stored value.
_CHANNEL_ATTRIBUTES = (_BITS, _RAW)


class PredicateError(Exception):
    """The filter expression could not be compiled."""


_BINARY_OPS: dict[type[ast.operator], BinaryOp] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide,
    ast.BitAnd: np.bitwise_and,
    ast.BitOr: np.bitwise_or,
    ast.BitXor: np.bitwise_xor,
}

_COMPARISONS: dict[type[ast.cmpop], CompareOp] = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}


def field_names(planes: tuple[SamplePlane, ...]) -> dict[str, str]:
    """Lowercase field name -> key prefix in the field environment.

    The names are ``left``, ``top``, ``right``, ``bottom``, and the channel
    names; matching ignores case, so ``b`` and ``B`` are the same field. Channel
    values come in two flavours, named by their attribute (see
    :func:`_value_key`).
    """
    names = {"left": "left", "top": "top", "right": "right", "bottom": "bottom"}
    for plane in planes:
        names.setdefault(plane.name.lower(), plane.name)
    return names


def _value_key(plane: str, attribute: str) -> str:
    """Environment key for one flavour of a channel value, e.g. ``B.raw``."""
    return f"{plane}.{attribute}"


class Filter:
    """A compiled display filter, evaluated over one image at a time."""

    def __init__(self, expression: str, evaluator: Evaluator) -> None:
        self.expression = expression
        self._evaluator = evaluator

    def evaluate(
        self,
        image: LoadedImage,
        chosen: frozenset[BitChoice] | None,
    ) -> NDArray[np.bool_]:
        """Return an HxW boolean mask; expressions over coordinates alone broadcast."""
        values = _field_values(image, chosen)
        with np.errstate(all="ignore"):
            result = self._evaluator(values)
        if result.dtype != np.bool_:
            result = result != 0
        return np.broadcast_to(result, (image.height, image.width))


def compile_filter(expression: str, planes: tuple[SamplePlane, ...]) -> Filter:
    """Compile ``expression`` against the planes' field names."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise PredicateError(f"语法错误：{exc.msg}") from exc
    names = field_names(planes)
    return Filter(expression, _compile_node(tree.body, names))


def _compile_node(node: ast.expr, names: dict[str, str]) -> Evaluator:
    match node:
        case ast.Constant(value=int() | float() as value):
            constant = np.asarray(value)
            return lambda _env: constant
        case ast.Name():
            canonical = _canonical_name(node.id, names)
            return lambda env: env[canonical]
        case ast.Attribute(value=ast.Name(id=field), attr=attribute):
            return _compile_attribute(field, attribute, names)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            inner = _compile_node(operand, names)
            return lambda env: np.logical_not(inner(env))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            inner = _compile_node(operand, names)
            return lambda env: np.negative(inner(env))
        case ast.BinOp(op=op, left=left, right=right) if type(op) in _BINARY_OPS:
            combine = _BINARY_OPS[type(op)]
            left_eval = _compile_node(left, names)
            right_eval = _compile_node(right, names)
            return lambda env: combine(left_eval(env), right_eval(env))
        case ast.BoolOp(op=op, values=values):
            operands = [_compile_node(value, names) for value in values]
            return _compile_bool_op(op, operands)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            operands = [_compile_node(left, names)]
            operands.extend(_compile_node(item, names) for item in comparators)
            return _compile_compare(ops, operands)
        case ast.Call(func=ast.Name(id=name), args=call_args, keywords=call_keywords):
            lowered = name.lower()
            if lowered == _RECT:
                return _compile_rect(
                    _function_args(_RECT, _RECT_EDGES, call_args, call_keywords, names)
                )
            if lowered == _GRID:
                return _compile_grid(
                    _function_args(_GRID, _GRID_PARAMS, call_args, call_keywords, names)
                )
            raise PredicateError(f"不支持的函数：{name}（可用：{'、'.join(_FUNCTIONS)}）")
        case ast.Call():
            raise PredicateError(f"不支持的函数调用（可用：{_FUNCTIONS[0]}(x0, y0, x1, y1) 等）")
        case _:
            raise PredicateError(f"不支持的表达式元素：{type(node).__name__}")


def _canonical_name(field: str, names: dict[str, str]) -> str:
    """The field's canonical name, or a clear error naming the alternatives."""
    canonical = names.get(field.lower())
    if canonical is None:
        available = ", ".join(sorted(set(names.values())))
        raise PredicateError(f"未知字段：{field}（可用：{available}）")
    return canonical


def _compile_attribute(field: str, attribute: str, names: dict[str, str]) -> Evaluator:
    """A channel attribute: ``R.raw`` (as stored) or ``R.bits`` (the selection)."""
    canonical = _canonical_name(field, names)
    if canonical in _RECT_EDGES:
        raise PredicateError(f"字段 {canonical} 没有属性")
    lowered = attribute.lower()
    if lowered not in _CHANNEL_ATTRIBUTES:
        options = "、".join(_CHANNEL_ATTRIBUTES)
        raise PredicateError(f"字段 {canonical} 的属性只能是 {options}（收到：{attribute}）")
    key = _value_key(canonical, lowered)
    return lambda env: env[key]


def _function_args(
    name: str,
    params: tuple[str, ...],
    args: list[ast.expr],
    keywords: list[ast.keyword],
    names: dict[str, str],
) -> list[Evaluator]:
    """One compiled evaluator per parameter, given positionally or by name."""
    if keywords:
        if args:
            raise PredicateError(f"{name} 不能混用位置参数和命名参数")
        provided: dict[str, ast.expr] = {}
        for keyword in keywords:
            if keyword.arg is None or keyword.arg not in params:
                received = keyword.arg or "**"
                raise PredicateError(
                    f"{name} 的参数名只能是 {'、'.join(params)}（收到：{received}）"
                )
            provided[keyword.arg] = keyword.value
        missing = [param for param in params if param not in provided]
        if missing:
            raise PredicateError(f"{name} 缺少参数：{'、'.join(missing)}")
        ordered = [provided[param] for param in params]
    else:
        if len(args) != len(params):
            usage = f"{name}({', '.join(params)})"
            raise PredicateError(
                f"{name} 需要 {len(params)} 个参数：{usage} 或 {name}("
                + ", ".join(f"{param}=" for param in params)
                + ")"
            )
        ordered = list(args)
    return [_compile_node(item, names) for item in ordered]


def _compile_rect(bounds: list[Evaluator]) -> Evaluator:
    """The box [left, right) x [top, bottom), matching the four edge fields."""

    def evaluate(env: FieldEnv) -> Value:
        x0, y0, x1, y1 = (_scalar(bound(env)) for bound in bounds)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        return (
            (env["left"] >= x0) & (env["right"] <= x1) & (env["top"] >= y0) & (env["bottom"] <= y1)
        )

    return evaluate


def _scalar(value: Value) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise PredicateError("参数必须是标量表达式")
    return int(array.item())


def _compile_grid(bounds: list[Evaluator]) -> Evaluator:
    """The lattice anchored at (x, y), taking one pixel every step_x / step_y."""

    def evaluate(env: FieldEnv) -> Value:
        x, y, step_x, step_y = (_scalar(bound(env)) for bound in bounds)
        if x < 0 or y < 0:
            raise PredicateError(f"{_GRID} 的起点坐标不能为负")
        if step_x < 1 or step_y < 1:
            raise PredicateError(f"{_GRID} 的步长必须至少为 1")
        # left/top are unsigned, so the subtraction wraps left of the anchor;
        # the >= guards pin those cells to False before the modulo can matter.
        return (
            (env["left"] >= x)
            & ((env["left"] - x) % step_x == 0)
            & (env["top"] >= y)
            & ((env["top"] - y) % step_y == 0)
        )

    return evaluate


def _compile_bool_op(op: ast.boolop, operands: list[Evaluator]) -> Evaluator:
    combine: BinaryOp = np.logical_and if isinstance(op, ast.And) else np.logical_or

    def evaluate(env: FieldEnv) -> Value:
        parts = [operand(env).astype(bool) for operand in operands]
        return reduce(combine, parts)

    return evaluate


def _compile_compare(ops: list[ast.cmpop], operands: list[Evaluator]) -> Evaluator:
    comparisons = [_COMPARISONS[type(op)] for op in ops]

    def evaluate(env: FieldEnv) -> Value:
        values = [operand(env) for operand in operands]
        parts = [
            compare(a, b) for compare, (a, b) in zip(comparisons, pairwise(values), strict=True)
        ]
        return reduce(np.logical_and, parts, np.asarray(True))

    return evaluate


def _field_values(
    image: LoadedImage,
    chosen: frozenset[BitChoice] | None,
) -> FieldEnv:
    height, width = image.height, image.width
    left = np.arange(width, dtype=np.uint32)[None, :]
    top = np.arange(height, dtype=np.uint32)[:, None]
    # A pixel occupies [left, right) x [top, bottom), so its far edges are +1.
    values: FieldEnv = {
        "left": left,
        "top": top,
        "right": left + np.uint32(1),
        "bottom": top + np.uint32(1),
    }
    for plane in image.planes:
        channel = _raw_values(image.samples, plane)
        selected = _selected_values(channel, plane.name, chosen)
        values[plane.name] = selected
        values[_value_key(plane.name, _BITS)] = selected
        values[_value_key(plane.name, _RAW)] = channel
    return values


def _raw_values(samples: SampleArray, plane: SamplePlane) -> Value:
    """The channel exactly as stored, ignoring the bit selection."""
    return samples[:, :, plane.index].astype(np.uint32)


def _selected_values(
    channel: Value,
    plane: str,
    chosen: frozenset[BitChoice] | None,
) -> Value:
    """The channel masked onto the selected bits; no selected bit reads as 0."""
    if chosen is None:
        return channel
    bits = bits_for(chosen, plane)
    if not bits:
        return np.zeros(channel.shape, dtype=np.uint32)
    if len(bits) == 1:
        return (channel >> np.uint32(bits[0])) & np.uint32(1)
    return channel & np.uint32(sum(1 << bit for bit in bits))

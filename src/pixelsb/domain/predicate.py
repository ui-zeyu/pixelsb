"""tshark-style display filter over pixel fields, compiled to vector operations.

Fields: ``left`` / ``top`` (aliases ``x`` / ``y``) and channel names. Channel
values are the current selection masked onto each plane; channels without any
selected bit read as 0. The expression is parsed with :mod:`ast` and every node
is compiled into a numpy closure — strings are never evaluated.
"""

import ast
from collections.abc import Callable
from functools import reduce
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
    """Lowercase alias -> key in the field environment. Names are case-insensitive."""
    names = {
        "left": "left",
        "x": "left",
        "top": "top",
        "y": "top",
        "up": "top",
    }
    for plane in planes:
        names.setdefault(plane.name.lower(), plane.name)
    return names


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
            canonical = names.get(node.id.lower())
            if canonical is None:
                available = ", ".join(sorted(set(names.values())))
                raise PredicateError(f"未知字段：{node.id}（可用：{available}）")
            return lambda env: env[canonical]
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
        case ast.Call(func=ast.Name(id=name), args=call_args, keywords=[]):
            if name.lower() != _RECT:
                raise PredicateError(f"不支持的函数：{name}（可用：{_RECT}）")
            if len(call_args) != 4:
                raise PredicateError(f"{_RECT} 需要 4 个参数：x0, y0, x1, y1")
            return _compile_rect([_compile_node(item, names) for item in call_args])
        case ast.Call():
            raise PredicateError(f"不支持的函数调用（可用：{_RECT}(x0, y0, x1, y1)）")
        case _:
            raise PredicateError(f"不支持的表达式元素：{type(node).__name__}")


def _compile_rect(bounds: list[Evaluator]) -> Evaluator:
    """``rect(x0, y0, x1, y1)``: the inclusive rectangle, swapped bounds allowed."""

    def evaluate(env: FieldEnv) -> Value:
        x0, y0, x1, y1 = (_scalar(bound(env)) for bound in bounds)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        left = env["left"]
        top = env["top"]
        return (left >= x0) & (left <= x1) & (top >= y0) & (top <= y1)

    return evaluate


def _scalar(value: Value) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise PredicateError(f"{_RECT} 的边界必须是标量表达式")
    return int(array.item())


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
            compare(a, b)
            for compare, a, b in zip(comparisons, values[:-1], values[1:], strict=True)
        ]
        return reduce(np.logical_and, parts, np.asarray(True))

    return evaluate


def _field_values(
    image: LoadedImage,
    chosen: frozenset[BitChoice] | None,
) -> FieldEnv:
    height, width = image.height, image.width
    values: FieldEnv = {
        "left": np.arange(width, dtype=np.uint32)[None, :],
        "top": np.arange(height, dtype=np.uint32)[:, None],
    }
    for plane in image.planes:
        values[plane.name] = _plane_values(image.samples, plane, chosen)
    return values


def _plane_values(
    samples: SampleArray,
    plane: SamplePlane,
    chosen: frozenset[BitChoice] | None,
) -> Value:
    channel = samples[:, :, plane.index].astype(np.uint32)
    if chosen is None:
        return channel
    bits = bits_for(chosen, plane.name)
    if not bits:
        return np.zeros(channel.shape, dtype=np.uint32)
    if len(bits) == 1:
        return (channel >> np.uint32(bits[0])) & np.uint32(1)
    return channel & np.uint32(sum(1 << bit for bit in bits))

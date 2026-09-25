"""The command input's language: one line, one operation.

A line names a single operation, and its first word says which kind it is:

* **操作动词** — ``thr 128``, ``xor 0xFF``, ``inv``, ``gray``, ``crop``: the line
  is the verb and its argument, nothing else. A verb is not a truth value, so it
  never joins an ``and``/``or``; another operation is another line.
* **位选择** — channel atoms only: ``b`` (the channel whole), ``b.0`` (one bit),
  ``all``. They combine with ``or`` (union), ``and`` (intersection) and ``not``
  (complement) into the one selection the canvas shows and the stream packs.
* **区域条件** — anything else: comparisons, arithmetic, coordinates, ``rect`` /
  ``grid``, and the bit fields inside them (``b > r``, ``left < 10``,
  ``B.0 == 1``). The text is handed to :mod:`pixelsb.domain.predicate`, which
  reads it as its own expression language and decides which pixels it names. A
  line whose shape is no condition at all — the tuple a stray comma makes, a
  list, a call to something that is not a filter function — is refused outright,
  so a typo never lands as an operation that only reports itself broken.

``and`` / ``or`` / ``not`` join operands of one kind, the way a typed language
joins one type: a line that mixes a selection with a region condition is refused,
because it would name two operations. Operations are combined by the stack
instead — one line each, folded bottom up.
"""

import ast
import re
from functools import reduce
from typing import assert_never

from pixelsb.domain.models import (
    BitChoice,
    BitsMask,
    CropMask,
    GrayscaleMask,
    InvertMask,
    Mask,
    RegionMask,
    SamplePlane,
    ThresholdMask,
    XorMask,
    level_ceiling,
)
from pixelsb.domain.predicate import _with_bit_attributes, condition_error
from pixelsb.domain.selection import all_bits, channel_members, whole

# A verb at the head of a line: the word itself, so a word that merely contains
# one — or follows a dot — is left to the expression grammar.
_VERB = re.compile(r"(?i)(?P<verb>thr|xor|inv|gray|crop)(?![\w.])")
_CONNECTIVES = frozenset({"and", "or", "not"})

_VALUE_MASKS: dict[str, type[ThresholdMask] | type[XorMask]] = {
    "thr": ThresholdMask,
    "xor": XorMask,
}
_BARE_MASKS: dict[str, type[InvertMask] | type[GrayscaleMask] | type[CropMask]] = {
    "inv": InvertMask,
    "gray": GrayscaleMask,
    "crop": CropMask,
}

_ONE_OPERATION = "一行只写一个操作：位选择（b、b.0、all）和区域条件不能组合在一起，请分成两行"
_VERB_APART = "操作命令要单独一行：thr、xor、inv、gray、crop 不能和别的内容组合"


class CommandError(ValueError):
    """A command line that cannot be read as a mask."""


def parse(text: str, planes: tuple[SamplePlane, ...]) -> Mask | None:
    """The one operation the box's text names, or ``None`` when it names none yet.

    ``None`` means the line is not finished — mid-word, mid-operator, mid-call —
    so nothing applies and the stack keeps what it had. The distinction is the
    grammar's: a line the expression syntax reads is a region condition, and the
    box judges the *line* — a shape no condition can have (a stray comma's tuple,
    a list, an unknown function) is refused, and the typist's words stay put to
    be fixed. What the line *names* is the stack's business: a condition about a
    field this image lacks is complete, and lands as a layer that says so.
    """
    line = text.strip()
    if not line:
        return None
    mask = _verb_line(line, level_ceiling(planes))
    if mask is not None:
        return mask
    if _VERB.search(line):  # a verb the head test did not claim is a verb out of place
        raise CommandError(_VERB_APART)
    try:
        tree = ast.parse(_with_bit_attributes(line), mode="eval").body
    except SyntaxError:
        return None
    part = _combine(tree, planes)
    if isinstance(part, frozenset):
        return BitsMask(part)
    if (reason := condition_error(part)) is not None:
        raise CommandError(reason)
    return RegionMask(_expression_text(part))


def text_of(mask: Mask, planes: tuple[SamplePlane, ...]) -> str | None:
    """The command line that means this mask again, or ``None`` when text falls short.

    A bits selection has a line while it touches channels this image has: every bit
    of every channel is ``all``, each whole channel one bare name (``b``), each
    single bit its own ``b.0``, joined by ``or``. Only an emptied selection — or
    one carried over from an image with other channels — keeps the command input
    empty.
    """
    match mask:
        case BitsMask(selection=selection):
            return _bits_line(planes, selection)
        case RegionMask(expression=expression):
            return expression
        case InvertMask():
            return "inv"
        case GrayscaleMask():
            return "gray"
        case CropMask():
            return "crop"
        case ThresholdMask(level=level):
            return f"thr {level}"
        case XorMask(value=value):
            return f"xor 0x{value:02X}"
        case _ as unknown:
            assert_never(unknown)


def _verb_line(line: str, ceiling: int) -> Mask | None:
    """The operation a line that opens with a verb names, or ``None`` for another kind."""
    match = _VERB.match(line)
    if match is None:
        return None
    words = line[match.end() :].split()
    argument = words.pop(0) if words and words[0].lower() not in _CONNECTIVES else None
    mask = _verb_mask(match["verb"].lower(), argument, ceiling)
    if words:  # the verb and its argument are the whole line
        raise CommandError(_VERB_APART)
    return mask


def _combine(node: ast.expr, planes: tuple[SamplePlane, ...]) -> ast.expr | frozenset[BitChoice]:
    """A subtree folded to what it says: a selection of bits, or expression material.

    Both kinds combine by their own algebra at once, so a subtree carrying both
    would name two operations: that is what :data:`_ONE_OPERATION` refuses.
    """
    match node:
        case ast.BoolOp(op=op, values=values):
            parts = [_combine(value, planes) for value in values]
            selections = [part for part in parts if isinstance(part, frozenset)]
            if selections and len(selections) != len(parts):
                raise CommandError(_ONE_OPERATION)
            if not selections:
                return node  # a region condition, left as written
            combine = frozenset.union if isinstance(op, ast.Or) else frozenset.intersection
            return reduce(combine, selections)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            part = _combine(operand, planes)
            return all_bits(planes) - part if isinstance(part, frozenset) else node
        case _:
            selection = _selector_leaf(node, planes)
            return node if selection is None else selection


def _selector_leaf(
    node: ast.expr,
    planes: tuple[SamplePlane, ...],
) -> frozenset[BitChoice] | None:
    """The selection a bare atom names, or ``None`` when it is predicate material."""
    match node:
        case ast.Name(id=name):
            if name.lower() == "all":
                return all_bits(planes)
            return frozenset(channel_members(planes, _plane(name, planes).name))
        case ast.Attribute(value=ast.Name(id=name), attr=number) if number.removeprefix(
            "_"
        ).isdigit():
            bit = int(number.removeprefix("_"))
            plane = _plane(name, planes)
            if bit >= plane.bit_depth:
                raise CommandError(
                    f"{plane.name} 只有 {plane.bit_depth} 位：位号是 0～{plane.bit_depth - 1}"
                    f"（收到 {name}.{bit}）"
                )
            return frozenset({BitChoice(plane.name, bit)})
        case _:
            return None


def _plane(name: str, planes: tuple[SamplePlane, ...]) -> SamplePlane:
    found = next((item for item in planes if item.name.lower() == name.lower()), None)
    if found is None:
        named = "、".join(item.name for item in planes)
        raise CommandError(f"没有 {name} 这样的通道（这张图有：{named}）")
    return found


def _expression_text(expression: ast.expr) -> str:
    """The region condition back as text, written out in the house style."""
    return re.sub(r"\._(\d+)", r".\1", ast.unparse(expression))


def _bits_line(
    planes: tuple[SamplePlane, ...],
    selection: frozenset[BitChoice],
) -> str | None:
    if whole(planes, selection):
        return "all"  # every bit of every channel: the one word that says it
    parts: list[str] = []
    for plane in planes:
        bits = sorted(choice.bit for choice in selection if choice.plane == plane.name)
        if not bits:
            continue
        if len(bits) == plane.bit_depth:
            parts.append(plane.name.lower())
        else:
            parts.extend(f"{plane.name.lower()}.{bit}" for bit in bits)
    known = {plane.name for plane in planes}
    if any(choice.plane not in known for choice in selection):
        return None  # a mask carried over from an image with other channels
    return " or ".join(parts) or None


def _verb_mask(verb: str, argument: str | None, ceiling: int) -> Mask:
    if (kind := _VALUE_MASKS.get(verb)) is not None:
        if argument is None:
            raise CommandError(f"{verb} 需要一个数值：{verb} 128 或 {verb} 0xFF")
        return kind(_level(argument, ceiling))
    if (kind := _BARE_MASKS.get(verb)) is not None:
        if argument is not None:
            raise CommandError(f"{verb} 不需要参数（收到：{argument}）")
        return kind()
    raise CommandError(f"看不懂的命令：{verb}")  # the verb pattern only lets these through


def _level(argument: str, ceiling: int) -> int:
    """The number a value command names: decimal, or hexadecimal with a ``0x``.

    A level is compared against the channels as stored, so one past the widest
    channel of *this* image would mean a mask that does nothing: it is refused
    rather than quietly left dark.
    """
    try:
        value = int(argument, 16) if argument.lower().startswith("0x") else int(argument, 10)
    except ValueError:
        raise CommandError(f"数值看不懂：{argument}（十进制 128，或十六进制 0x80）") from None
    if not 0 <= value <= ceiling:
        raise CommandError(f"数值超出这张图的 0..{ceiling}：{argument}")
    return value

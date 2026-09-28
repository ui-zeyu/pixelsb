"""The command input's language: one line, one operation.

A line names a single operation, and its first word says which kind it is:

* **操作动词** — ``thr 128``, ``xor 0xFF``, ``inv``, ``gray``, ``crop``,
  ``fft r g``, ``arnold 1 2 3``, ``comb xor b.png``: the line is the verb and
  its argument, nothing else. A verb is not a truth value, so it never joins
  an ``and``/``or``; another operation is another line.
* **位选择** — channel atoms only: ``b`` (the channel whole), ``b.0`` (one bit),
  ``all``. They combine with ``or`` (union), ``and`` (intersection) and ``not``
  (complement) into the one selection the projection rebuilds the channels
  from. Like every operation it lands in the stack: what it makes becomes the
  picture and the bytes the operations above it read.
* **区域条件** — anything else: comparisons, arithmetic, coordinates, ``rect`` /
  ``grid``, and the bit fields inside them (``b > r``, ``left < 10``,
  ``B.0 == 1``). The text is handed to :mod:`pixelsb.domain.predicate`, which
  reads it as its own expression language and decides which pixels it names. A
  line whose shape is no condition at all — the tuple a stray comma makes, a
  list, a call to something that is not a filter function — is refused outright,
  so a typo never lands as an operation that only reports itself broken.

``and`` / ``or`` / ``not`` join operands of one kind, the way a typed language
joins one type: a line that mixes a selection with a region condition is refused,
because it would name two things. Operations are combined by the stack
instead — one line each, folded bottom up.
"""

import ast
import os
import re
from collections.abc import Callable
from functools import reduce
from pathlib import Path
from typing import assert_never

from pixelsb.domain.models import (
    ARNOLD_PARAM_LIMIT,
    ArnoldMask,
    BitChoice,
    BitsMask,
    CombineMask,
    CombineOp,
    CropMask,
    FftMask,
    GrayscaleMask,
    InvertMask,
    LoadedImage,
    Mask,
    RegionMask,
    SamplePlane,
    ThresholdMask,
    XorMask,
    level_ceiling,
)
from pixelsb.domain.predicate import condition_error, with_bit_attributes
from pixelsb.domain.selection import all_bits, channel_members, whole

# A verb at the head of a line: the word itself, so a word that merely contains
# one — or follows a dot — is left to the expression grammar.
_VERB = re.compile(r"(?i)(?P<verb>thr|xor|inv|gray|crop|fft|arnold|comb)(?![\w.])")
_CONNECTIVES = frozenset({"and", "or", "not"})

_VALUE_MASKS: dict[str, type[ThresholdMask] | type[XorMask]] = {
    "thr": ThresholdMask,
    "xor": XorMask,
}
_BARE_MASKS: dict[str, type[InvertMask] | type[GrayscaleMask] | type[CropMask] | type[FftMask]] = {
    "inv": InvertMask,
    "gray": GrayscaleMask,
    "crop": CropMask,
    "fft": FftMask,
}

_ONE_OPERATION = "一行只写一个操作：位选择（b、b.0、all）和区域条件不能组合在一起，请分成两行"
_VERB_APART = "操作命令要单独一行：thr、xor、inv、gray、crop、fft、arnold、comb 不能和别的内容组合"

# How a comb line reaches its file: the caller's loader, so the domain never reads.
type Companion = Callable[[str], LoadedImage]

_COMB_OPS = tuple(op.value for op in CombineOp)


class CommandError(ValueError):
    """A command line that cannot be read as a mask."""


def parse(
    text: str,
    planes: tuple[SamplePlane, ...],
    *,
    companion: Companion | None = None,
) -> Mask | None:
    """The one operation the box's text names, or ``None`` when it names none yet.

    ``None`` means the line is not finished — mid-word, mid-operator, mid-call —
    so nothing applies and the stack keeps what it had. The distinction is the
    grammar's: a line the expression syntax reads is a region condition, and the
    box judges the *line* — a shape no condition can have (a stray comma's tuple,
    a list, an unknown function) is refused, and the typist's words stay put to
    be fixed. What the line *names* is the stack's business: a condition about a
    field this image lacks is complete, and lands as a layer that says so. A
    line of channel atoms names the projection, whose bits it builds. A ``comb``
    line reaches its file through ``companion``, the loader the caller supplies.
    """
    line = text.strip()
    if not line:
        return None
    mask = _verb_line(line, planes, companion)
    if mask is not None:
        return mask
    if _VERB.search(line):  # a verb the head test did not claim is a verb out of place
        raise CommandError(_VERB_APART)
    try:
        tree = ast.parse(with_bit_attributes(line), mode="eval").body
    except SyntaxError:
        return None
    part = _combine(tree, planes)
    if isinstance(part, frozenset):
        return BitsMask(part)
    if (reason := condition_error(part)) is not None:
        raise CommandError(reason)
    return RegionMask(_expression_text(part))


def text_of(mask: Mask, planes: tuple[SamplePlane, ...], *, base: Path | None = None) -> str:
    """The command line that means this mask again, in the house style.

    ``base`` is where the open picture lives: a combine spells its other file
    relative to it, so the line names the folder the challenge ships in.
    """
    match mask:
        case BitsMask(selection=selection):
            return _bits_line(planes, selection) or ""
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
        case FftMask(planes=names):
            if names is None:
                return "fft"
            return "fft " + " ".join(name.lower() for name in names)
        case CombineMask(op=op, other=other):
            return f"comb {op.value} {_companion_word(other.path, base)}"
        case ArnoldMask(times=times, a=a, b=b):
            return f"arnold {times} {a} {b}"
        case _ as unknown:
            assert_never(unknown)


def _companion_word(path: Path, base: Path | None) -> str:
    """The other picture as the line spells it: a bare name beside the open file."""
    if base is None:
        return path.name
    return os.path.relpath(path, base)


def bits_text(planes: tuple[SamplePlane, ...], selection: frozenset[BitChoice]) -> str | None:
    """The command line that sets this projection, or ``None`` when text falls short.

    Every bit of every channel is ``all``, each whole channel one bare name
    (``b``), each single bit its own ``b.0``, joined by ``or``. An emptied
    selection — or one carried over from an image with other channels — has no
    line, which keeps the command input free for a fresh one.
    """
    return _bits_line(planes, selection)


def _verb_line(
    line: str,
    planes: tuple[SamplePlane, ...],
    companion: Companion | None,
) -> Mask | None:
    """The operation a line that opens with a verb names, or ``None`` for another kind."""
    match = _VERB.match(line)
    if match is None:
        return None
    verb = match["verb"].lower()
    rest = line[match.end() :].strip()
    words = rest.split()
    taken = _parameter_words(verb, words)
    if not taken:
        argument = None
    elif taken == len(words):
        argument = rest  # every word is a parameter, spelled as the typist wrote it
    else:
        argument = " ".join(words[:taken])
    mask = _verb_mask(verb, argument, level_ceiling(planes), companion)
    if taken < len(words):  # the verb and its parameters are the whole line
        raise CommandError(_VERB_APART)
    return mask


def _parameter_words(verb: str, words: list[str]) -> int:
    """How many words this verb reads as its parameters: three, one, or all.

    A connective in the first word's place is a line trying to combine, so no
    parameter is taken and the leftover words refuse the line below — unless
    the verb is ``comb`` and the word is one of its operation names, since
    ``comb or b.png`` is an operation of its own. ``fft`` reads every word as a
    channel name, so ``fft r g`` stays one operation; ``comb`` reads every
    word too, because a path may carry spaces.
    """
    if verb == "comb" and words and words[0].lower() in _COMB_OPS:
        return len(words)
    if not words or words[0].lower() in _CONNECTIVES:
        return 0
    if verb == "arnold":
        return 3
    return len(words) if verb in {"fft", "comb"} else 1


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


def _verb_mask(
    verb: str,
    argument: str | None,
    ceiling: int,
    companion: Companion | None,
) -> Mask:
    if (kind := _VALUE_MASKS.get(verb)) is not None:
        if argument is None:
            raise CommandError(f"{verb} 需要一个数值：{verb} 128 或 {verb} 0xFF")
        return kind(_level(argument, ceiling))
    if (kind := _BARE_MASKS.get(verb)) is not None:
        if verb == "fft":
            return _fft_mask(argument)
        if argument is not None:
            raise CommandError(f"{verb} 不需要参数（收到：{argument}）")
        return kind()
    if verb == "arnold":
        return _arnold_mask(argument)
    if verb == "comb":
        return _comb_mask(argument, companion)
    raise CommandError(f"看不懂的命令：{verb}")  # the verb pattern only lets these through


_PLANE_NAME = re.compile(r"(?i)^[a-z]+$")


def _fft_mask(argument: str | None) -> FftMask:
    """The spectrum's line: bare, or the channel names it works on.

    The names are this mask's own parameters, so ``fft r g`` keeps doing the
    same thing when another frame carries other channels; whether the image
    has them is the stack's business, reported against the layer.
    """
    if argument is None:
        return FftMask()
    names = tuple(argument.split())
    for name in names:
        if _PLANE_NAME.match(name) is None:
            raise CommandError(f"fft 的参数是通道名：fft r g（收到：{argument}）")
    return FftMask(tuple(name.upper() for name in names))


def _arnold_mask(argument: str | None) -> ArnoldMask:
    """The cat map's line: three integers, times first.

    The operation is the recovery direction, so the same ``(a, b)`` a challenge's
    encoder used is what goes here; the refusals spell the shape out so a
    half-typed line says what it is missing.
    """
    if argument is None:
        raise CommandError("arnold 需要三个整数：arnold 1 2 3（次数 a b）")
    words = argument.split()
    if len(words) != 3:
        raise CommandError(f"arnold 需要三个整数：arnold 1 2 3（收到 {len(words)} 个）")
    times, a, b = (_signed(word) for word in words)
    if not 1 <= times <= ARNOLD_PARAM_LIMIT:
        raise CommandError(f"次数要在 1..{ARNOLD_PARAM_LIMIT}：arnold {argument}")
    if abs(a) > ARNOLD_PARAM_LIMIT or abs(b) > ARNOLD_PARAM_LIMIT:
        raise CommandError(f"a、b 要在 ±{ARNOLD_PARAM_LIMIT} 内：arnold {argument}")
    return ArnoldMask(times, a, b)


def _comb_mask(argument: str | None, companion: Companion | None) -> CombineMask:
    """The combine's line: an operation word, then the other picture's path.

    The path runs to the end of the line, spaces and all, and is resolved by
    the caller's loader — beside the open file, or absolute. Whether the two
    pictures fit together is judged where both are known, when the layer lands,
    which is also why an unreadable file is refused here: the loader raises.
    """
    if argument is None:
        raise CommandError("comb 需要一个操作和一张图：comb xor b.png")
    op, _, rest = argument.partition(" ")
    if op.lower() not in _COMB_OPS:
        raise CommandError(
            f"comb 的操作是 {'、'.join(_COMB_OPS)} 之一：comb xor b.png（收到：{op}）"
        )
    path = rest.strip()
    if not path:
        raise CommandError("comb 的最后是另一张图的路径：comb xor b.png")
    if companion is None:
        raise CommandError("comb 要读另一张图，这一行没有可用的文件来源")
    return CombineMask(companion(path), CombineOp(op.lower()))


def _integer(word: str, complaint: str) -> int:
    """An integer with an optional sign: decimal, or hexadecimal with a ``0x``.

    ``complaint`` is the error a malformed word raises — the verbs word theirs
    differently for thresholds and cat-map parameters. ``int`` itself reads the
    sign and, at base 16, the ``0x`` prefix.
    """
    try:
        return int(word, 16) if "0x" in word.lower() else int(word, 10)
    except ValueError:
        raise CommandError(complaint) from None


def _signed(word: str) -> int:
    return _integer(word, f"整数看不懂：{word}（十进制 3，或十六进制 0x1F）")


def _level(argument: str, ceiling: int) -> int:
    """The number a value command names: decimal, or hexadecimal with a ``0x``.

    A level is compared against the channels as stored, so one past the widest
    channel of *this* image would mean a mask that does nothing: it is refused
    rather than quietly left dark.
    """
    value = _integer(argument, f"数值看不懂：{argument}（十进制 128，或十六进制 0x80）")
    if not 0 <= value <= ceiling:
        raise CommandError(f"数值超出这张图的 0..{ceiling}：{argument}")
    return value

"""Which channel bits are visible."""

from pixelsb.domain.models import BitChoice, SamplePlane, plane_named


def all_bits(planes: tuple[SamplePlane, ...]) -> frozenset[BitChoice]:
    return frozenset(
        BitChoice(plane.name, bit) for plane in planes for bit in range(plane.bit_depth)
    )


def lsb_bits(planes: tuple[SamplePlane, ...]) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane.name, 0) for plane in planes)


def channel_members(planes: tuple[SamplePlane, ...], name: str) -> frozenset[BitChoice]:
    """Every bit of one plane; ``KeyError`` when the planes hold no such channel."""
    plane = plane_named(planes, name)
    return frozenset(BitChoice(name, bit) for bit in range(plane.bit_depth))


def column_members(planes: tuple[SamplePlane, ...], bit: int) -> frozenset[BitChoice]:
    """One bit of every plane wide enough to have it."""
    return frozenset(BitChoice(plane.name, bit) for plane in planes if bit < plane.bit_depth)


def plane_ladder(planes: tuple[SamplePlane, ...]) -> tuple[BitChoice, ...]:
    """Every single bit plane in display order: R7 … R0, G7 … G0, B7 … B0.

    Matches how the bit grid reads — high bits on the left, channels top to
    bottom — so stepping the ladder walks the grid left to right, top to bottom.
    """
    return tuple(
        BitChoice(plane.name, bit) for plane in planes for bit in range(plane.bit_depth - 1, -1, -1)
    )


def bits_for(selection: frozenset[BitChoice], plane: str) -> tuple[int, ...]:
    return tuple(sorted(choice.bit for choice in selection if choice.plane == plane))


def active_bits(
    planes: tuple[SamplePlane, ...],
    selection: frozenset[BitChoice],
) -> tuple[BitChoice, ...]:
    """The selected bits in plane order, then low to high."""
    return tuple(
        BitChoice(plane.name, bit) for plane in planes for bit in bits_for(selection, plane.name)
    )


def whole(planes: tuple[SamplePlane, ...], selection: frozenset[BitChoice]) -> bool:
    """Whether the selection is every bit of every plane: the original image."""
    return selection == all_bits(planes)


def mask_of(bits: tuple[int, ...]) -> int:
    return sum(1 << bit for bit in bits)


def bit_value(sample: int, bit: int) -> int:
    return (sample >> bit) & 1


def shown_channel_value(sample: int, bits: tuple[int, ...]) -> tuple[int, int]:
    """Return the number to display for one plane, and the bit width used to format it.

    A single selected bit is ``0`` or ``1``. Several bits keep their original weights,
    which is exactly the stored sample masked onto the selected bits.
    """
    if len(bits) == 1:
        return bit_value(sample, bits[0]), 1
    return sample & mask_of(bits), max(bits) + 1

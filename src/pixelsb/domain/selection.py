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


def grid_planes(planes: tuple[SamplePlane, ...]) -> tuple[SamplePlane, ...]:
    """The channels the bit grid shows, in its row order: alpha first, then the rest."""
    return tuple(sorted(planes, key=lambda plane: plane.name != "A"))


def bits_for(selection: frozenset[BitChoice], plane: str) -> tuple[int, ...]:
    return tuple(sorted(choice.bit for choice in selection if choice.plane == plane))


def whole(planes: tuple[SamplePlane, ...], selection: frozenset[BitChoice]) -> bool:
    """Whether the selection is every bit of every plane: the original image."""
    return selection == all_bits(planes)

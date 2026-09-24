"""Which channel bits are visible."""

from pixelsb.domain.models import BitChoice, LoadedImage, ViewerState


def all_bits(image: LoadedImage) -> frozenset[BitChoice]:
    return frozenset(
        BitChoice(plane.name, bit) for plane in image.planes for bit in range(plane.bit_depth)
    )


def lsb_bits(image: LoadedImage) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane.name, 0) for plane in image.planes if plane.bit_depth > 0)


def effective_selection(
    image: LoadedImage,
    selection: frozenset[BitChoice] | None,
) -> frozenset[BitChoice]:
    if selection is None:
        return all_bits(image)
    return selection


def number_bits(state: ViewerState) -> frozenset[BitChoice] | None:
    """Bits driving the pixel numbers. While attached they mirror the canvas layer."""
    if not state.detached:
        return state.selection
    return state.readout


def bits_for(selection: frozenset[BitChoice], plane: str) -> tuple[int, ...]:
    return tuple(sorted(choice.bit for choice in selection if choice.plane == plane))


def mask_of(bits: tuple[int, ...]) -> int:
    return sum(1 << bit for bit in bits)


def stored_selection(
    image: LoadedImage,
    chosen: frozenset[BitChoice],
) -> frozenset[BitChoice] | None:
    if chosen == all_bits(image):
        return None
    return chosen


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

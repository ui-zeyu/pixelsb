"""The zsteg sweep: candidates over channels, bits, and stream order.

Every candidate *is* a recipe — a bits selection and the extract order that goes
with it — so a hit applies to the stack unchanged and the sweep is the extract
panel's own run across the whole parameter space. This module only enumerates
what to try; judging a stream is the caller's job.
"""

from dataclasses import dataclass

from pixelsb.domain.models import (
    BitChoice,
    BitOrder,
    ExtractOrder,
    SamplePlane,
    ScanOrder,
)

# The channel sets worth sweeping: each channel alone, and the packed orders a
# flag hunt usually means. A set naming a channel the image lacks drops out.
# Index is what a palette image actually stores, so it gets swept on its own.
_COMBOS: tuple[tuple[str, ...], ...] = (
    ("R",),
    ("G",),
    ("B",),
    ("A",),
    ("L",),
    ("Index",),
    ("L", "A"),
    ("Index", "A"),
    ("R", "G", "B"),
    ("B", "G", "R"),
    ("R", "G", "B", "A"),
    ("A", "B", "G", "R"),
)
_BIT_ORDERS = (BitOrder.MSB, BitOrder.LSB)
_SCANS = (ScanOrder.XY, ScanOrder.YZ)


@dataclass(frozen=True, slots=True)
class ScanCandidate:
    """One thing worth trying: some channels' one bit, read in one order.

    ``selection`` is the bits recipe, ``order`` the extract order that goes with
    it, and ``command`` the line that writes the recipe into the stack.
    """

    selection: frozenset[BitChoice]
    order: ExtractOrder
    command: str


@dataclass(frozen=True, slots=True)
class ScanHit:
    """One candidate's verdict: what its stream turned out to be.

    ``label`` names a format the stream was recognized as, ``flags`` lists the
    flag-shaped strings in it, and ``preview`` is the stream's first bytes as
    printable text — the eye's own check on whether the model's read is real.
    Both label and flags empty means nothing showed up.
    """

    candidate: ScanCandidate
    label: str = ""
    flags: tuple[str, ...] = ()
    preview: str = ""

    @property
    def noteworthy(self) -> bool:
        return bool(self.flags or self.label)


def scan_candidates(planes: tuple[SamplePlane, ...]) -> tuple[ScanCandidate, ...]:
    """Every candidate this image has: each channel set's common bit, in every order.

    A set naming a channel the image lacks drops out whole — rgba on an RGB file
    would only repeat rgb — and the bit runs over the narrowest channel of its
    set, so nothing names a bit no plane carries. Each selection pairs with both
    bit orders and both scan orders, the four ways a stream can be read.
    """
    named = {plane.name: plane for plane in planes}
    candidates: list[ScanCandidate] = []
    for combo in _COMBOS:
        if any(name not in named for name in combo):
            continue
        involved = [named[name] for name in combo]
        names = tuple(plane.name for plane in involved)
        orders = [
            ExtractOrder(names, bit_order, scan) for bit_order in _BIT_ORDERS for scan in _SCANS
        ]
        candidates.extend(
            ScanCandidate(
                frozenset(BitChoice(plane.name, bit) for plane in involved),
                order,
                " or ".join(f"{name.lower()}.{bit}" for name in names),
            )
            for bit in range(min(plane.bit_depth for plane in involved))
            for order in orders
        )
    return tuple(candidates)

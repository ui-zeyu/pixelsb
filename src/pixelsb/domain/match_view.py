"""Compacted canvas view: only the pixels that pass the display filter."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixelsb.domain.models import PixelCoord, RgbArray

type IndexArray = NDArray[np.intp]
type Background = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MatchView:
    """The rendered image compressed to its matching rows and columns.

    ``rgb`` is ``len(ys) x len(xs)``: display cell ``(dx, dy)`` shows source
    pixel ``(xs[dx], ys[dy])`` where ``match`` is set, and the canvas
    background where it is not — a ``grid()`` filter yields the exact strided
    sub-image, a ``rect()`` filter a plain crop.
    """

    xs: IndexArray
    ys: IndexArray
    rgb: RgbArray
    match: NDArray[np.bool_]

    @property
    def width(self) -> int:
        return int(self.xs.size)

    @property
    def height(self) -> int:
        return int(self.ys.size)

    def source_at(self, dx: int, dy: int) -> PixelCoord:
        """The source pixel shown at display cell ``(dx, dy)``."""
        return PixelCoord(int(self.xs[dx]), int(self.ys[dy]))

    def display_of(self, coord: PixelCoord) -> tuple[int, int] | None:
        """The display cell showing ``coord``, or ``None`` when it is hidden."""
        dx = int(np.searchsorted(self.xs, coord.x))
        if dx >= self.xs.size or self.xs[dx] != coord.x:
            return None
        dy = int(np.searchsorted(self.ys, coord.y))
        if dy >= self.ys.size or self.ys[dy] != coord.y:
            return None
        return dx, dy


def match_span(match: NDArray[np.bool_]) -> tuple[IndexArray, IndexArray] | None:
    """The rows and columns the match touches; ``None`` when nothing matches."""
    xs = np.flatnonzero(match.any(axis=0))
    ys = np.flatnonzero(match.any(axis=1))
    if xs.size == 0 or ys.size == 0:
        return None
    return ys, xs


def match_view(
    dense_rgb: RgbArray,
    match: NDArray[np.bool_],
    ys: IndexArray,
    xs: IndexArray,
    background: Background,
) -> MatchView:
    """Assemble the view from ``dense_rgb`` already gathered at ``(ys, xs)``.

    ``dense_rgb`` normally comes straight from :func:`render_rgb_at`, so the
    matching pixels are rendered once at view size and never re-gathered.
    """
    dense_match = match[np.ix_(ys, xs)]
    dense = np.where(dense_match[..., None], dense_rgb, np.array(background, dtype=dense_rgb.dtype))
    return MatchView(xs=xs, ys=ys, rgb=dense, match=dense_match)

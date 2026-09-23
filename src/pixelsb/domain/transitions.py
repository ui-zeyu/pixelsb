"""Pure transitions for viewer state."""

from dataclasses import replace

from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    BitChoice,
    DisplayFormat,
    LoadedImage,
    PixelCoord,
    ValueMode,
    ViewerState,
    ensure_inside,
)
from pixelsb.domain.selection import all_bits, lsb_bits, stored_selection

_FORMATS = (DisplayFormat.DECIMAL, DisplayFormat.HEX, DisplayFormat.BINARY)


def open_image(state: ViewerState, image: LoadedImage, *, zoom: int | None = None) -> ViewerState:
    chosen = state.zoom if zoom is None else zoom
    if not MIN_ZOOM <= chosen <= MAX_ZOOM:
        raise ValueError(f"zoom {chosen} is outside {MIN_ZOOM}..{MAX_ZOOM}")
    if not image.planes:
        raise ValueError("image has no planes")
    return ViewerState(
        image=image,
        cursor=None,
        anchor=None,
        selection=None,
        focus=BitChoice(image.planes[0].name, 0),
        value_format=state.value_format,
        value_mode=state.value_mode,
        zoom=chosen,
    )


def set_cursor(state: ViewerState, coord: PixelCoord) -> ViewerState:
    ensure_inside(_image(state), coord)
    return replace(state, cursor=coord)


def move_cursor(state: ViewerState, dx: int, dy: int) -> ViewerState:
    image = state.image
    if image is None:
        return state
    if state.cursor is None:
        return replace(state, cursor=PixelCoord(0, 0))
    x = min(max(state.cursor.x + dx, 0), image.width - 1)
    y = min(max(state.cursor.y + dy, 0), image.height - 1)
    return replace(state, cursor=PixelCoord(x, y))


def set_anchor(state: ViewerState, coord: PixelCoord) -> ViewerState:
    ensure_inside(_image(state), coord)
    return replace(state, anchor=coord, cursor=coord)


def clear_anchor(state: ViewerState) -> ViewerState:
    return replace(state, anchor=None)


def select_only(state: ViewerState, plane: str, bit: int) -> ViewerState:
    image = _image(state)
    choice = _choice(image, plane, bit)
    return replace(
        state,
        selection=stored_selection(image, frozenset({choice})),
        focus=choice,
    )


def toggle_bit(state: ViewerState, plane: str, bit: int) -> ViewerState:
    image = _image(state)
    choice = _choice(image, plane, bit)
    current = all_bits(image) if state.selection is None else state.selection
    updated = current - {choice} if choice in current else current | {choice}
    return replace(state, selection=stored_selection(image, updated), focus=choice)


def select_focus_only(state: ViewerState) -> ViewerState:
    image = state.image
    focus = state.focus
    if image is None or focus is None:
        return state
    return select_only(state, focus.plane, focus.bit)


def select_lsbs(state: ViewerState) -> ViewerState:
    image = _image(state)
    chosen = lsb_bits(image)
    focus = state.focus if state.focus is not None else next(iter(chosen), None)
    return replace(state, selection=stored_selection(image, chosen), focus=focus)


def select_all_bits(state: ViewerState) -> ViewerState:
    if state.image is None:
        return state
    return replace(state, selection=None)


def clear_selection(state: ViewerState) -> ViewerState:
    if state.image is None:
        return state
    return replace(state, selection=frozenset())


def step_focus_bit(state: ViewerState, delta: int) -> ViewerState:
    image = state.image
    if image is None:
        return state
    focus = state.focus or BitChoice(image.planes[0].name, 0)
    plane = image.plane(focus.plane)
    bit = min(max(focus.bit + delta, 0), plane.bit_depth - 1)
    return select_only(state, plane.name, bit)


def select_lsb(state: ViewerState, name: str) -> ViewerState:
    image = state.image
    if image is None or all(plane.name != name for plane in image.planes):
        return state
    return select_only(state, name, 0)


def select_lsb_at(state: ViewerState, index: int) -> ViewerState:
    image = state.image
    if image is None or not 0 <= index < len(image.planes):
        return state
    return select_only(state, image.planes[index].name, 0)


def set_format(state: ViewerState, fmt: DisplayFormat) -> ViewerState:
    return replace(state, value_format=fmt)


def cycle_format(state: ViewerState) -> ViewerState:
    index = _FORMATS.index(state.value_format)
    return replace(state, value_format=_FORMATS[(index + 1) % len(_FORMATS)])


def set_value_mode(state: ViewerState, mode: ValueMode) -> ViewerState:
    return replace(state, value_mode=mode)


def toggle_value_mode(state: ViewerState) -> ViewerState:
    mode = ValueMode.OFFSET if state.value_mode is ValueMode.ABSOLUTE else ValueMode.ABSOLUTE
    return replace(state, value_mode=mode)


def set_zoom(state: ViewerState, zoom: int) -> ViewerState:
    if not MIN_ZOOM <= zoom <= MAX_ZOOM:
        raise ValueError(f"zoom {zoom} is outside {MIN_ZOOM}..{MAX_ZOOM}")
    return replace(state, zoom=zoom)


def step_zoom(state: ViewerState, delta: int) -> ViewerState:
    return set_zoom(state, min(max(state.zoom + delta, MIN_ZOOM), MAX_ZOOM))


def _image(state: ViewerState) -> LoadedImage:
    if state.image is None:
        raise ValueError("no image open")
    return state.image


def _choice(image: LoadedImage, plane: str, bit: int) -> BitChoice:
    try:
        sample_plane = image.plane(plane)
    except KeyError as exc:
        raise ValueError(f"unknown plane: {plane}") from exc
    if not 0 <= bit < sample_plane.bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{sample_plane.bit_depth - 1}")
    return BitChoice(plane, bit)

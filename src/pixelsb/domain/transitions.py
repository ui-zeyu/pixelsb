"""Pure transitions for viewer state."""

from dataclasses import replace

from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    BitChoice,
    BitOrder,
    DisplayFormat,
    ExtractEncoding,
    ExtractOrder,
    LoadedImage,
    PixelCoord,
    SamplePlane,
    ScanOrder,
    ViewAdjust,
    ViewerState,
    ensure_inside,
    plane_named,
)
from pixelsb.domain.selection import (
    all_bits,
    effective_selection,
    lsb_bits,
    plane_ladder,
    stored_selection,
)

_FORMATS = (DisplayFormat.DECIMAL, DisplayFormat.HEX, DisplayFormat.BINARY)


def open_image(
    state: ViewerState,
    image: LoadedImage,
    *,
    zoom: float | None = None,
) -> ViewerState:
    chosen = state.zoom if zoom is None else zoom
    if not MIN_ZOOM <= chosen <= MAX_ZOOM:
        raise ValueError(f"zoom {chosen} is outside {MIN_ZOOM}..{MAX_ZOOM}")
    if not image.planes:
        raise ValueError("image has no planes")
    return ViewerState(
        image=image,
        cursor=None,
        selection=None,
        focus=BitChoice(image.planes[0].name, 0),
        filter_expr=state.filter_expr,
        value_format=state.value_format,
        extract_encoding=state.extract_encoding,
        extract_order=state.extract_order,
        adjust=state.adjust,
        zoom=chosen,
        only_matched=state.only_matched,
    )


def set_frame(state: ViewerState, image: LoadedImage) -> ViewerState:
    """Show another frame of the open file, keeping what carries over.

    The selection survives pruned to the bits this frame actually offers —
    GIF frames can decode into shifting modes — and a selection that carries
    nothing falls back to the whole image, as a fresh open would.
    """
    previous = state.image
    if previous is None:
        return open_image(state, image)
    chosen = frozenset(
        choice
        for choice in effective_selection(previous, state.selection)
        if _offers(image, choice)
    )
    cursor = state.cursor
    inside = cursor is not None and 0 <= cursor.x < image.width and 0 <= cursor.y < image.height
    focus = state.focus if state.focus is not None and _offers(image, state.focus) else None
    return replace(
        state,
        image=image,
        cursor=cursor if inside else None,
        selection=stored_selection(image, chosen) if chosen else None,
        focus=focus or BitChoice(image.planes[0].name, 0),
    )


def _offers(image: LoadedImage, choice: BitChoice) -> bool:
    """Whether the image has that plane, wide enough for the chosen bit."""
    plane = next((plane for plane in image.planes if plane.name == choice.plane), None)
    return plane is not None and choice.bit < plane.bit_depth


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


def select_lsbs(state: ViewerState) -> ViewerState:
    image = state.image
    if image is None:
        return state
    chosen = lsb_bits(image)
    focus = state.focus if state.focus is not None else next(iter(chosen), None)
    return replace(state, selection=stored_selection(image, chosen), focus=focus)


def select_all_bits(state: ViewerState) -> ViewerState:
    if state.image is None:
        return state
    return replace(state, selection=None)


def set_channel(state: ViewerState, name: str, *, on: bool) -> ViewerState:
    """Switch every bit of one channel in the canvas layer on or off."""
    image = _image(state)
    updated = _set_members(
        effective_selection(image, state.selection), _channel_members(image, name), on=on
    )
    return replace(state, selection=stored_selection(image, updated))


def set_column(state: ViewerState, bit: int, *, on: bool) -> ViewerState:
    """Switch one bit column across every plane in the canvas layer on or off."""
    image = _image(state)
    updated = _set_members(effective_selection(image, state.selection), _column(image, bit), on=on)
    return replace(state, selection=stored_selection(image, updated))


def set_filter_expr(state: ViewerState, expression: str) -> ViewerState:
    return replace(state, filter_expr=expression)


def set_only_matched(state: ViewerState, *, on: bool) -> ViewerState:
    """Show only the pixels that pass the filter; the canvas compacts to them."""
    if state.only_matched is on:
        return state
    return replace(state, only_matched=on)


def _set_members(
    current: frozenset[BitChoice],
    members: frozenset[BitChoice],
    *,
    on: bool,
) -> frozenset[BitChoice]:
    return current | members if on else current - members


def _channel_members(image: LoadedImage, name: str) -> frozenset[BitChoice]:
    """Every bit of one plane, validating the name."""
    plane = _required_plane(image, name)
    return frozenset(BitChoice(name, bit) for bit in range(plane.bit_depth))


def step_focus_bit(state: ViewerState, delta: int) -> ViewerState:
    image = state.image
    if image is None:
        return state
    focus = state.focus or BitChoice(image.planes[0].name, 0)
    plane = image.plane(focus.plane)
    bit = min(max(focus.bit + delta, 0), plane.bit_depth - 1)
    return select_only(state, plane.name, bit)


def step_plane(state: ViewerState, delta: int) -> ViewerState:
    """Walk single bit planes in display order: R7 … R0, G7 … G0, B7 … B0.

    A step from a wider view enters the ladder at the end the arrow points at,
    so the first press lands on the first plane; a step from a single plane
    moves one plane on, wrapping around.
    """
    image = state.image
    if image is None or delta == 0:
        return state
    ladder = plane_ladder(image)
    choice = _single_bit(state)
    if choice is None:
        choice = ladder[0] if delta > 0 else ladder[-1]
    else:
        choice = ladder[(ladder.index(choice) + delta) % len(ladder)]
    return select_only(state, choice.plane, choice.bit)


def step_channel(state: ViewerState, delta: int) -> ViewerState:
    """Walk whole channels: every bit of R, then G, then B, wrapping around."""
    image = state.image
    if image is None or delta == 0:
        return state
    names = [plane.name for plane in image.planes]
    # A view that is not already one whole channel starts just off the ladder,
    # so the first step lands on the channel the arrow points at.
    position = next(
        (
            index
            for index, name in enumerate(names)
            if state.selection == _channel_members(image, name)
        ),
        -1 if delta > 0 else len(names),
    )
    name = names[(position + delta) % len(names)]
    return replace(
        state,
        selection=stored_selection(image, _channel_members(image, name)),
        focus=BitChoice(name, 0),
    )


def _single_bit(state: ViewerState) -> BitChoice | None:
    """The one plane the view shows, when it shows exactly one."""
    selection = state.selection
    if selection is None or len(selection) != 1:
        return None
    return next(iter(selection))


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


def set_extract_encoding(state: ViewerState, encoding: ExtractEncoding) -> ViewerState:
    """Switch how the extract panel reads the byte stream as text."""
    if state.extract_encoding is encoding:
        return state
    return replace(state, extract_encoding=encoding)


def set_channel_order(state: ViewerState, names: tuple[str, ...]) -> ViewerState:
    """Which channel's bits come first in the extracted stream."""
    return _with_order(state, replace(state.extract_order, planes=names))


def set_bit_order(state: ViewerState, bit_order: BitOrder) -> ViewerState:
    """Which end of each byte the first bit of the stream lands in."""
    return _with_order(state, replace(state.extract_order, bit_order=bit_order))


def set_scan_order(state: ViewerState, scan: ScanOrder) -> ViewerState:
    """Whether the stream walks rows first (XY) or columns first (YZ)."""
    return _with_order(state, replace(state.extract_order, scan=scan))


def toggle_invert(state: ViewerState) -> ViewerState:
    """Flip the view's invert post-processing."""
    return _with_adjust(state, replace(state.adjust, invert=not state.adjust.invert))


def toggle_grayscale(state: ViewerState) -> ViewerState:
    """Flip the view's grayscale post-processing."""
    return _with_adjust(state, replace(state.adjust, grayscale=not state.adjust.grayscale))


def toggle_threshold(state: ViewerState) -> ViewerState:
    """Flip the view's threshold post-processing."""
    return _with_adjust(state, replace(state.adjust, threshold=not state.adjust.threshold))


def set_threshold_level(state: ViewerState, level: int) -> ViewerState:
    """The gray value the threshold post-processing splits the channels at."""
    return _with_adjust(state, replace(state.adjust, level=level))


def _with_adjust(state: ViewerState, adjust: ViewAdjust) -> ViewerState:
    if adjust == state.adjust:
        return state
    return replace(state, adjust=adjust)


def _with_order(state: ViewerState, order: ExtractOrder) -> ViewerState:
    if order == state.extract_order:
        return state
    return replace(state, extract_order=order)


def cycle_format(state: ViewerState) -> ViewerState:
    index = _FORMATS.index(state.value_format)
    return replace(state, value_format=_FORMATS[(index + 1) % len(_FORMATS)])


def set_zoom(state: ViewerState, zoom: float) -> ViewerState:
    if not MIN_ZOOM <= zoom <= MAX_ZOOM:
        raise ValueError(f"zoom {zoom} is outside {MIN_ZOOM}..{MAX_ZOOM}")
    return replace(state, zoom=zoom)


def step_zoom(state: ViewerState, delta: float) -> ViewerState:
    return set_zoom(state, min(max(state.zoom + delta, MIN_ZOOM), MAX_ZOOM))


def _image(state: ViewerState) -> LoadedImage:
    if state.image is None:
        raise ValueError("no image open")
    return state.image


def _required_plane(image: LoadedImage, name: str) -> SamplePlane:
    """The plane of that name, or the error a transition raises for a bad one."""
    try:
        return plane_named(image.planes, name)
    except KeyError as exc:
        raise ValueError(f"unknown plane: {name}") from exc


def _choice(image: LoadedImage, plane: str, bit: int) -> BitChoice:
    sample_plane = _required_plane(image, plane)
    if not 0 <= bit < sample_plane.bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{sample_plane.bit_depth - 1}")
    return BitChoice(plane, bit)


def _column(image: LoadedImage, bit: int) -> frozenset[BitChoice]:
    return frozenset(BitChoice(plane.name, bit) for plane in image.planes if bit < plane.bit_depth)

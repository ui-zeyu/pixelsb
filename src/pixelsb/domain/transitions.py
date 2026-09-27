"""Pure transitions for viewer state."""

from collections.abc import Callable
from dataclasses import replace

from pixelsb.domain import commands
from pixelsb.domain.models import (
    MAX_ZOOM,
    MIN_ZOOM,
    ArnoldMask,
    BitChoice,
    BitOrder,
    BitsMask,
    DisplayFormat,
    ExtractEncoding,
    ExtractOrder,
    Layer,
    LoadedImage,
    Mask,
    PixelCoord,
    SamplePlane,
    ScanOrder,
    ViewerState,
    ensure_inside,
    plane_named,
    plane_or_none,
)
from pixelsb.domain.selection import (
    all_bits,
    channel_members,
    column_members,
    grid_planes,
    lsb_bits,
    plane_ladder,
)

_FORMATS = (DisplayFormat.DECIMAL, DisplayFormat.HEX, DisplayFormat.BINARY)


def open_image(
    state: ViewerState,
    image: LoadedImage,
    *,
    zoom: float | None = None,
) -> ViewerState:
    """Show an image. The mask stack carries over, as the view settings do."""
    chosen = state.zoom if zoom is None else zoom
    if not MIN_ZOOM <= chosen <= MAX_ZOOM:
        raise ValueError(f"zoom {chosen} is outside {MIN_ZOOM}..{MAX_ZOOM}")
    if not image.planes:
        raise ValueError("image has no planes")
    return ViewerState(
        image=image,
        cursor=None,
        focus=BitChoice(image.planes[0].name, 0),
        layers=state.layers,
        value_format=state.value_format,
        extract_encoding=state.extract_encoding,
        extract_order=state.extract_order,
        zoom=chosen,
    )


def set_frame(state: ViewerState, image: LoadedImage) -> ViewerState:
    """Show another frame of the open file, keeping what carries over.

    The stack stays where it is: a mask naming a channel this frame lacks keeps
    its place, and simply has nothing of its own to select there.
    """
    previous = state.image
    if previous is None:
        return open_image(state, image)
    cursor = state.cursor
    inside = cursor is not None and 0 <= cursor.x < image.width and 0 <= cursor.y < image.height
    focus = state.focus if state.focus is not None and _offers(image, state.focus) else None
    return replace(
        state,
        image=image,
        cursor=cursor if inside else None,
        focus=focus or BitChoice(image.planes[0].name, 0),
    )


def _offers(image: LoadedImage, choice: BitChoice) -> bool:
    """Whether the image has that plane, wide enough for the chosen bit."""
    plane = plane_or_none(image.planes, choice.plane)
    return plane is not None and choice.bit < plane.bit_depth


# --- the layer stack -------------------------------------------------------


def add_layer(state: ViewerState, mask: Mask) -> ViewerState:
    """Put a mask on top of the stack, where it sees everything below it."""
    _image(state)
    return replace(state, layers=(*state.layers, Layer(mask)))


def remove_layer(state: ViewerState, layer: int) -> ViewerState:
    """Take one mask out of the stack."""
    _layer_at(state.layers, layer)
    return replace(state, layers=state.layers[:layer] + state.layers[layer + 1 :])


def move_layer(state: ViewerState, layer: int, step: int) -> ViewerState:
    """Move one mask along the stack: ``step`` -1 or +1, up or down its order."""
    _layer_at(state.layers, layer)
    target = layer + step
    if not 0 <= target < len(state.layers):
        return state
    layers = list(state.layers)
    layers[layer], layers[target] = layers[target], layers[layer]
    return replace(state, layers=tuple(layers))


def set_layer_enabled(state: ViewerState, layer: int, *, on: bool) -> ViewerState:
    """Switch one mask on or off, keeping its place in the stack."""
    current = _layer_at(state.layers, layer)
    if current.enabled is on:
        return state
    return replace(state, layers=_replaced(state.layers, layer, replace(current, enabled=on)))


def clear_layers(state: ViewerState) -> ViewerState:
    """Drop every mask, leaving the image as it was decoded."""
    return state if not state.layers else replace(state, layers=())


def set_mask_text(state: ViewerState, text: str, *, layer: int | None = None) -> ViewerState:
    """Write the command input's text as the one thing it names.

    A line of operations lands on the operation the box is editing — the layer
    in hand — and rewrites it, kind and all: that is what picking a row and
    changing its command means. With no operation in hand the line becomes a new
    one on top. A line of bits names the projection: it takes over the topmost
    one, or becomes it, whatever the box was editing. A line still being typed
    applies nothing. An empty text drops the layer the box is bound to.
    """
    if not text.strip():
        bound = filter_position(state.layers, layer)
        return state if bound is None else remove_layer(state, bound)
    parsed = _parsed_line(state, text)
    if parsed is None:  # not finished: the box keeps the line, the stack keeps its layers
        return state
    if isinstance(parsed, BitsMask):
        return set_bits(state, parsed.selection)
    position = filter_position(state.layers, layer)
    if position is None:
        return _appended(state, parsed)
    return _remasked(state, position, parsed)


def add_mask_text(state: ViewerState, text: str) -> ViewerState:
    """Write the command input's text as a new operation on top, stacking it.

    Shift+Enter's line: the operation in hand keeps its command and its place,
    so a second threshold can be added without picking another operation first.
    A line of bits still names the projection, which replaces rather than
    stacks — the view is one thing. A line still being typed, and an empty
    one, change nothing.
    """
    parsed = _parsed_line(state, text)
    if parsed is None:
        return state
    if isinstance(parsed, BitsMask):
        return set_bits(state, parsed.selection)
    return _appended(state, parsed)


def _parsed_line(state: ViewerState, text: str) -> Mask | None:
    """What the line names, or ``None`` while it names nothing yet.

    A mask this image cannot carry — the cat map wants a square — is refused
    here rather than landed as a layer that could only report itself broken.
    """
    parsed = commands.parse(text, _image(state).planes)
    if parsed is None:
        return None
    if isinstance(parsed, ArnoldMask) and (image := _image(state)).width != image.height:
        raise commands.CommandError(f"猫脸变换要方图：这张是 {image.width}×{image.height}")
    return parsed


def apply_arnold(state: ViewerState, times: int, a: int, b: int) -> ViewerState:
    """The brute gallery's landing: rewrite the cat-map layer it last wrote.

    One layer carries the search's current answer, so clicking through
    candidates never stacks a tower of half-tried transforms; with no cat-map
    layer in the stack, the pick becomes a new one on top.
    """
    position = _topmost(state.layers, ArnoldMask)
    mask = ArnoldMask(times, a, b)
    return _remasked(state, position, mask) if position is not None else _appended(state, mask)


def set_bits(state: ViewerState, choices: frozenset[BitChoice]) -> ViewerState:
    """Make these bits the view: the topmost projection takes them.

    One projection layer carries the view, so a new answer replaces it and
    never stacks a tower; with no projection in the stack, the bits become a
    new layer on top, where they read everything below. The focus rides to the
    highest bit shown, where the bit keys start.
    """
    focus = min(choices, key=lambda choice: (-choice.bit, choice.plane)) if choices else None
    return _write_bits(state, choices, focus=focus)


def _appended(state: ViewerState, mask: Mask) -> ViewerState:
    """Put one more mask at the top of the stack."""
    return replace(state, layers=(*state.layers, Layer(mask)))


def toggle_bit(state: ViewerState, plane: str, bit: int) -> ViewerState:
    """Switch one bit of one channel on or off in the projection."""
    choice = _choice(_image(state), plane, bit)
    return _edit_bits(
        state,
        lambda current: current - {choice} if choice in current else current | {choice},
    )


def select_only(state: ViewerState, plane: str, bit: int) -> ViewerState:
    """Show that one bit alone."""
    choice = _choice(_image(state), plane, bit)
    return _write_bits(state, frozenset({choice}), focus=choice)


def set_channel(state: ViewerState, name: str, *, on: bool) -> ViewerState:
    """Switch every bit of one channel in the projection on or off."""
    members = _channel_members(_image(state), name)
    return _edit_bits(state, lambda current: _set_members(current, members, on=on))


def set_column(state: ViewerState, bit: int, *, on: bool) -> ViewerState:
    """Switch one bit column across every plane in the projection on or off."""
    members = column_members(_image(state).planes, bit)
    return _edit_bits(state, lambda current: _set_members(current, members, on=on))


def select_lsbs(state: ViewerState) -> ViewerState:
    """Show every channel's lowest bit, the way StegSolve opens."""
    image = state.image
    if image is None:
        return state
    chosen = lsb_bits(image.planes)
    focus = state.focus if state.focus is not None else next(iter(chosen), None)
    return _write_bits(state, chosen, focus=focus)


def select_all_bits(state: ViewerState) -> ViewerState:
    """Drop the projection: every bit of every channel again, the decoded image."""
    position = _topmost(state.layers, BitsMask)
    if position is None:
        return state
    return replace(state, layers=state.layers[:position] + state.layers[position + 1 :])


def step_focus_bit(state: ViewerState, delta: int) -> ViewerState:
    """Move the one shown bit up or down its channel, stopping at either end."""
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
    ladder = plane_ladder(grid_planes(image.planes))
    choice = _single_bit(state)
    # A projection carried over from another image can show a bit this one has
    # no plane for; that is no single bit of *this* image, so the step enters at
    # the end the arrow points at, as a step from a wider view does.
    if choice is None or choice not in ladder:
        choice = ladder[0] if delta > 0 else ladder[-1]
    else:
        choice = ladder[(ladder.index(choice) + delta) % len(ladder)]
    return select_only(state, choice.plane, choice.bit)


def step_channel(state: ViewerState, delta: int) -> ViewerState:
    """Walk whole channels: every bit of R, then G, then B, wrapping around."""
    image = state.image
    if image is None or delta == 0:
        return state
    names = [plane.name for plane in grid_planes(image.planes)]
    current = current_bits(state)
    # A view that is not already one whole channel starts just off the ladder,
    # so the first step lands on the channel the arrow points at.
    position = next(
        (index for index, name in enumerate(names) if current == _channel_members(image, name)),
        -1 if delta > 0 else len(names),
    )
    name = names[(position + delta) % len(names)]
    members = _channel_members(image, name)
    return replace(
        _edit_bits(state, lambda _current: members),
        focus=BitChoice(name, 0),
    )


def select_lsb(state: ViewerState, name: str) -> ViewerState:
    """Show one named channel's lowest bit."""
    image = state.image
    if image is None or plane_or_none(image.planes, name) is None:
        return state
    return select_only(state, name, 0)


def select_lsb_at(state: ViewerState, position: int) -> ViewerState:
    """Show the lowest bit of the plane sitting at ``position`` in the grid order."""
    image = state.image
    if image is None:
        return state
    shown = grid_planes(image.planes)
    if not 0 <= position < len(shown):
        return state
    return select_only(state, shown[position].name, 0)


def _edit_bits(
    state: ViewerState,
    update: Callable[[frozenset[BitChoice]], frozenset[BitChoice]],
) -> ViewerState:
    """Rewrite the projection's bits; the focus stays where the keyboard is."""
    return _write_bits(state, update(current_bits(state)), focus=None)


def _write_bits(
    state: ViewerState,
    choices: frozenset[BitChoice],
    *,
    focus: BitChoice | None,
) -> ViewerState:
    """Land one projection: the topmost one re-taken, or a new layer on top.

    The rewritten layer switches on as well — the bits are where the user is
    looking, and a view command that leaves the view switched off says nothing.
    """
    position = _topmost(state.layers, BitsMask)
    layer = Layer(BitsMask(choices), enabled=True)
    if position is None:
        state = replace(state, layers=(*state.layers, layer))
    else:
        state = replace(state, layers=_replaced(state.layers, position, layer))
    return replace(state, focus=focus) if focus is not None else state


def _topmost(layers: tuple[Layer, ...], kind: type[Mask]) -> int | None:
    """The topmost layer whose mask is of that kind, or ``None`` when there is none."""
    return next(
        (index for index in reversed(range(len(layers))) if isinstance(layers[index].mask, kind)),
        None,
    )


def current_bits(state: ViewerState) -> frozenset[BitChoice]:
    """What the topmost projection shows; with none, every bit of the image.

    The bit grid, the shortcuts, and the extract panel all read the same view,
    so they share this.
    """
    projected = projection_bits(state)
    if projected is not None:
        return projected
    return all_bits(() if state.image is None else state.image.planes)


def projection_bits(state: ViewerState) -> frozenset[BitChoice] | None:
    """The bits the topmost projection carries, or ``None`` with no projection."""
    layer = _bits_layer(state)
    return None if layer is None else layer.selection


def _single_bit(state: ViewerState) -> BitChoice | None:
    """The one bit the projection shows, when it shows exactly one."""
    layer = _bits_layer(state)
    if layer is None or len(layer.selection) != 1:
        return None
    return next(iter(layer.selection))


def _bits_layer(state: ViewerState) -> BitsMask | None:
    """The topmost projection's mask, or ``None`` when the stack carries none."""
    for layer in reversed(state.layers):
        if isinstance(layer.mask, BitsMask):
            return layer.mask
    return None


def filter_position(layers: tuple[Layer, ...], layer: int | None = None) -> int | None:
    """Where the mask the filter box edits sits: the layer in hand, or nowhere.

    Every mask has a command line of its own now, so an index the stack has is
    the box's edit target, and one it no longer has names nothing.
    """
    return layer if layer is not None and 0 <= layer < len(layers) else None


def _layer_at(layers: tuple[Layer, ...], layer: int) -> Layer:
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer} is outside 0..{len(layers) - 1}")
    return layers[layer]


def _remasked(state: ViewerState, layer: int, mask: Mask) -> ViewerState:
    """Swap one layer's mask, keeping its place in the stack and its switch."""
    current = _layer_at(state.layers, layer)
    return replace(state, layers=_replaced(state.layers, layer, replace(current, mask=mask)))


def _replaced(layers: tuple[Layer, ...], layer: int, updated: Layer) -> tuple[Layer, ...]:
    return (*layers[:layer], updated, *layers[layer + 1 :])


def _set_members(
    current: frozenset[BitChoice],
    members: frozenset[BitChoice],
    *,
    on: bool,
) -> frozenset[BitChoice]:
    return current | members if on else current - members


def _channel_members(image: LoadedImage, name: str) -> frozenset[BitChoice]:
    """Every bit of one channel, with the error a transition raises for a bad name."""
    return channel_members(image.planes, _required_plane(image, name).name)


# --- the view --------------------------------------------------------------


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


def _choice(image: LoadedImage, plane: str, bit: int) -> BitChoice:
    sample_plane = _required_plane(image, plane)
    if not 0 <= bit < sample_plane.bit_depth:
        raise ValueError(f"bit {bit} is outside 0..{sample_plane.bit_depth - 1}")
    return BitChoice(plane, bit)


def _required_plane(image: LoadedImage, name: str) -> SamplePlane:
    """The plane of that name, or the error a transition raises for a bad one."""
    try:
        return plane_named(image.planes, name)
    except KeyError as exc:
        raise ValueError(f"unknown plane: {name}") from exc

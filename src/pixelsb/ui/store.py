"""Owns the current viewer state and notifies listeners after each transition."""

from collections.abc import Callable

from pixelsb.domain.models import ViewerState

type Transition = Callable[[ViewerState], ViewerState]
type Listener = Callable[[ViewerState], None]


class Store:
    def __init__(self, state: ViewerState | None = None) -> None:
        self._state = ViewerState() if state is None else state
        self._listeners: list[Listener] = []

    @property
    def state(self) -> ViewerState:
        return self._state

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def update(self, transition: Transition) -> None:
        self._state = transition(self._state)
        for listener in tuple(self._listeners):
            listener(self._state)

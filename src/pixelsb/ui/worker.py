"""The sweep pages' worker thread: one stopping rule, one shutdown.

Both sweeps — the scan candidates and the cat-map gallery — run a
``QThread`` that answers a stop flag between units of work and is dropped
the same way when the page closes or the picture changes.
"""

from PySide6.QtCore import QObject, QThread


class Stoppable(QThread):
    """A worker whose run loop checks :meth:`stop` between units of work."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stopping = False

    def stop(self) -> None:
        self._stopping = True


def shutdown(worker: Stoppable | None) -> None:
    """Stop a worker and free it; the wait bounds the join if a unit stalls.

    ``None`` is the no-worker state every caller carries, so it is nothing to
    stop.
    """
    if worker is None:
        return
    worker.stop()
    worker.wait(2000)
    worker.deleteLater()

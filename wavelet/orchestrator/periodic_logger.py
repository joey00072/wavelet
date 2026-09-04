from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Self

logger = logging.getLogger(__name__)


def pipeline_status(
    *,
    policy_step: int | None,
    published: int,
    target: int,
    submitted: int,
    inflight: int,
    ready: int,
    policy_loading: bool,
) -> str:
    policy = "none" if policy_step is None else str(policy_step)
    return (
        f"Pipeline | policy={policy} | published={published}/{target} | "
        f"submitted={submitted} | inflight={inflight} | ready={ready} | "
        f"policy_load={'pending' if policy_loading else 'idle'}"
    )


class PeriodicLogger:
    """Emit a cheap status snapshot from a daemon thread at a fixed interval."""

    def __init__(self, collect: Callable[[], str], *, interval_seconds: float) -> None:
        self.collect = collect
        self.interval_seconds = interval_seconds
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def emit(self) -> None:
        logger.info(self.collect())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="wavelet-pipeline-status",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped.wait(self.interval_seconds):
            self.emit()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval_seconds, 5.0))
            self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

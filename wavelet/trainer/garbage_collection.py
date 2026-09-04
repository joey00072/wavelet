from __future__ import annotations

import gc
import logging
import time

logger = logging.getLogger(__name__)


class DeterministicGarbageCollector:
    """Run Python collections at shared training-step boundaries."""

    def __init__(self, interval: int) -> None:
        if interval < 1:
            raise ValueError("Garbage-collection interval must be positive.")
        self.interval = interval
        self._automatic_gc_was_enabled = gc.isenabled()
        self._closed = False
        gc.disable()
        self._collect()

    def run(self, step: int) -> None:
        if step > 0 and step % self.interval == 0:
            self._collect()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._automatic_gc_was_enabled:
            gc.enable()

    @staticmethod
    def _collect() -> None:
        started_at = time.monotonic()
        gc.collect(1)
        logger.debug(
            "Collected generation-1 garbage in %.3f seconds",
            time.monotonic() - started_at,
        )

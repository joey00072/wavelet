"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    logger as logger,
    append_event as append_event,
    append_event_best_effort as append_event_best_effort,
    tail_events as tail_events,
)

"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    policy_lag as policy_lag,
    event_rate as event_rate,
    publish_rate as publish_rate,
    consume_rate as consume_rate,
    _parse_datetime as _parse_datetime,
)

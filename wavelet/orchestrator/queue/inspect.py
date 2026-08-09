"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    scan_queue_dir as scan_queue_dir,
    scan_policy_dir as scan_policy_dir,
    build_queue_report as build_queue_report,
    _scan_queue_item as _scan_queue_item,
    _read_optional as _read_optional,
    _oldest_age_seconds as _oldest_age_seconds,
    _age_seconds as _age_seconds,
    _timestamp_age_seconds as _timestamp_age_seconds,
)

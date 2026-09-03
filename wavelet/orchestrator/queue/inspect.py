"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    _age_seconds as _age_seconds,
)
from wavelet.transport.queue import (
    _oldest_age_seconds as _oldest_age_seconds,
)
from wavelet.transport.queue import (
    _read_optional as _read_optional,
)
from wavelet.transport.queue import (
    _scan_queue_item as _scan_queue_item,
)
from wavelet.transport.queue import (
    _timestamp_age_seconds as _timestamp_age_seconds,
)
from wavelet.transport.queue import (
    build_queue_report as build_queue_report,
)
from wavelet.transport.queue import (
    scan_policy_dir as scan_policy_dir,
)
from wavelet.transport.queue import (
    scan_queue_dir as scan_queue_dir,
)

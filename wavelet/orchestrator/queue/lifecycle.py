"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    RecordT as RecordT,
)
from wavelet.transport.queue import (
    _read_record as _read_record,
)
from wavelet.transport.queue import (
    _trace_output_dir as _trace_output_dir,
)
from wavelet.transport.queue import (
    _write_record as _write_record,
)
from wavelet.transport.queue import (
    logger as logger,
)
from wavelet.transport.queue import (
    process_identity as process_identity,
)
from wavelet.transport.queue import (
    read_claim as read_claim,
)
from wavelet.transport.queue import (
    read_consumed as read_consumed,
)
from wavelet.transport.queue import (
    read_manifest as read_manifest,
)
from wavelet.transport.queue import (
    record_rollout_claim as record_rollout_claim,
)
from wavelet.transport.queue import (
    record_rollout_consumed as record_rollout_consumed,
)
from wavelet.transport.queue import (
    utc_now as utc_now,
)
from wavelet.transport.queue import (
    write_claim as write_claim,
)
from wavelet.transport.queue import (
    write_consumed as write_consumed,
)
from wavelet.transport.queue import (
    write_manifest as write_manifest,
)

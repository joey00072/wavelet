"""Compatibility wrapper for :mod:`wavelet.transport.queue`."""

from wavelet.transport.queue import (
    logger as logger,
    RecordT as RecordT,
    utc_now as utc_now,
    process_identity as process_identity,
    write_manifest as write_manifest,
    read_manifest as read_manifest,
    write_claim as write_claim,
    read_claim as read_claim,
    write_consumed as write_consumed,
    read_consumed as read_consumed,
    record_rollout_claim as record_rollout_claim,
    record_rollout_consumed as record_rollout_consumed,
    _trace_output_dir as _trace_output_dir,
    _write_record as _write_record,
    _read_record as _read_record,
)

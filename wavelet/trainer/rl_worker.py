"""Compatibility exports for the consolidated RL trainer module."""

from wavelet.trainer.rl import (
    RLTrainer,
    _dummy_rollout_row,
    _run_streaming_rollout_training,
    _StreamingChunkAccumulator,
    _use_streaming_rollout_chunks,
    _validate_rollout_batch,
    _validate_streaming_rollout_batch,
    main,
)

__all__ = [
    "RLTrainer",
    "_StreamingChunkAccumulator",
    "_dummy_rollout_row",
    "_run_streaming_rollout_training",
    "_use_streaming_rollout_chunks",
    "_validate_rollout_batch",
    "_validate_streaming_rollout_batch",
    "main",
]

"""Shared filesystem-rollout training loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from wavelet.contracts.queue_records import RolloutBatch
from wavelet.data.rl import count_rollout_rows
from wavelet.monitor import emit_perf
from wavelet.trainer.base import TrainerBackend


@dataclass(slots=True)
class RolloutStepTimings:
    """Durations for one consumed rollout batch."""

    wait_batch: float = 0.0
    load_rollout: float = 0.0
    train: float = 0.0
    export_policy: float = 0.0


def load_and_train_received_batch(
    trainer: TrainerBackend,
    received: RolloutBatch,
) -> RolloutStepTimings:
    """Validate, train, consume, and export one already-received batch."""
    result = RolloutStepTimings()
    trainer_step_before = trainer.step
    row_count = count_rollout_rows(
        received.training_path,
        description="Rollout batch",
    )
    trainer.validate_rollout_batch(received, row_count=row_count)
    trainer.record_rollout_claim(
        received,
        trainer_step_before=trainer_step_before,
    )

    started_at = perf_counter()
    trainer.load_rollout_path(received.path)
    result.load_rollout = perf_counter() - started_at

    started_at = perf_counter()
    trainer.prepare_for_training()
    trainer.train_until(trainer.step + 1)
    result.train = perf_counter() - started_at
    trainer.record_rollout_consumed(
        received,
        trainer_step_before=trainer_step_before,
        optimizer_step_completed=True,
    )

    started_at = perf_counter()
    trainer.export_policy(step=trainer.step)
    trainer.offload_after_refit()
    result.export_policy = perf_counter() - started_at

    return result


def run_rl_training(
    trainer: TrainerBackend,
    receiver: object,
    *,
    target_step: int,
) -> None:
    """Run the non-streaming RL loop used by process-mode trainers."""
    wait = receiver.wait
    while trainer.step < target_step:
        started_at = perf_counter()
        wait_started_at = perf_counter()
        batch = wait()
        wait_seconds = perf_counter() - wait_started_at
        timings = load_and_train_received_batch(trainer, batch)
        timings.wait_batch = wait_seconds
        emit_perf(
            "trainer_step",
            step=trainer.step,
            wait_batch=timings.wait_batch,
            load_rollout=timings.load_rollout,
            train=timings.train,
            export_policy=timings.export_policy,
            total=perf_counter() - started_at,
        )


__all__ = [
    "RolloutStepTimings",
    "load_and_train_received_batch",
    "run_rl_training",
]

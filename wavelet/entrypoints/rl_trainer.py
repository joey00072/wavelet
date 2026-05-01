from __future__ import annotations

import os
import sys
from math import ceil
from time import perf_counter

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import FileSystemRolloutReceiver
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.utils.config import load_config


def _perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    trainer = RLTrainer(config)
    try:
        trainer.setup()
        if config.orchestrator.enabled:
            trainer.export_policy(step=trainer.step)
            trainer.offload_after_refit()
            receiver = FileSystemRolloutReceiver(
                config.output_dir,
                config.transport,
                start_step=trainer.step,
            )
            try:
                target_step = config.max_steps or 1
                if _use_streaming_rollout_chunks(config):
                    _run_streaming_rollout_training(
                        config,
                        trainer,
                        receiver,
                        target_step=target_step,
                    )
                    trainer.finalize(status="completed")
                    return 0
                while trainer.step < target_step:
                    loop_started_at = perf_counter()
                    wait_started_at = perf_counter()
                    batch = receiver.wait()
                    wait_seconds = perf_counter() - wait_started_at
                    load_started_at = perf_counter()
                    trainer.load_rollout_path(batch.path)
                    load_seconds = perf_counter() - load_started_at
                    train_started_at = perf_counter()
                    trainer.prepare_for_training()
                    trainer.train_until(trainer.step + 1)
                    train_seconds = perf_counter() - train_started_at
                    export_started_at = perf_counter()
                    trainer.export_policy(step=trainer.step)
                    trainer.offload_after_refit()
                    export_seconds = perf_counter() - export_started_at
                    total_seconds = perf_counter() - loop_started_at
                    if _perf_enabled():
                        print(
                            "WAVELET_PERF trainer_step "
                            f"step={trainer.step} wait_batch={wait_seconds:.3f} "
                            f"load_rollout={load_seconds:.3f} "
                            f"train={train_seconds:.3f} "
                            f"export_policy={export_seconds:.3f} "
                            f"total={total_seconds:.3f}",
                            flush=True,
                        )
            except Exception:
                trainer.finalize(status="failed")
                raise
            trainer.finalize(status="completed")
        else:
            trainer.train()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def _use_streaming_rollout_chunks(config: RLConfig) -> bool:
    return (
        config.launcher.mode == "process"
        and (
            config.orchestrator.custom_rollout_function is None
            or config.orchestrator.custom_rollout_function
            == "wavelet.orchestrator.verifiers:generate_rollouts"
        )
        and config.orchestrator.max_async_level > 0
        and config.orchestrator.examples_per_step is not None
    )


def _run_streaming_rollout_training(
    config: RLConfig,
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    *,
    target_step: int,
) -> None:
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        raise ValueError("orchestrator.examples_per_step is required.")
    rollouts_per_example = config.orchestrator.rollouts_per_example or 1
    target_rollout_rows = examples_per_step * rollouts_per_example
    chunk_examples = _rollout_chunk_examples(config)
    chunks_per_step = max(ceil(examples_per_step / chunk_examples), 1)
    accumulated_rows = 0
    chunk_index = 0

    while trainer.step < target_step:
        loop_started_at = perf_counter()
        wait_started_at = perf_counter()
        batch = receiver.wait()
        wait_seconds = perf_counter() - wait_started_at
        load_started_at = perf_counter()
        row_count = _count_rollout_rows(batch.path)
        trainer.load_rollout_path(batch.path)
        accumulated_rows += row_count
        chunk_index += 1
        chunks_into_step = (chunk_index - 1) % chunks_per_step
        remaining_chunks = max(chunks_per_step - chunks_into_step - 1, 0)
        should_step = (
            accumulated_rows >= target_rollout_rows
            or chunk_index % chunks_per_step == 0
        )
        loaded_micro_batches = trainer._loaded_micro_batch_count
        if should_step:
            trainer.accumulation_steps = (
                trainer._accumulated_micro_batches + loaded_micro_batches
            )
        else:
            trainer.accumulation_steps = (
                trainer._accumulated_micro_batches
                + loaded_micro_batches * (remaining_chunks + 1)
            )
        trainer._optimizer_batch_loss_scale = None
        load_seconds = perf_counter() - load_started_at
        train_started_at = perf_counter()
        trainer.prepare_for_training()
        metrics = trainer.train_loaded_rollouts_once()
        train_seconds = perf_counter() - train_started_at

        export_seconds = 0.0
        if metrics is not None:
            _log_step_perf_metrics(
                trainer,
                metrics,
                train_seconds=train_seconds,
                loop_seconds=perf_counter() - loop_started_at,
            )
            accumulated_rows = 0
            export_started_at = perf_counter()
            trainer.export_policy(step=trainer.step)
            trainer.offload_after_refit()
            export_seconds = perf_counter() - export_started_at
        total_seconds = perf_counter() - loop_started_at
        if _perf_enabled():
            print(
                "WAVELET_PERF trainer_chunk "
                f"queue_step={batch.step} trainer_step={trainer.step} "
                f"rows={row_count} accumulated_rows={accumulated_rows} "
                f"wait_batch={wait_seconds:.3f} "
                f"load_rollout={load_seconds:.3f} "
                f"train={train_seconds:.3f} "
                f"export_policy={export_seconds:.3f} "
                f"optimizer_step={int(metrics is not None)} "
                f"total={total_seconds:.3f}",
                flush=True,
            )


def _log_step_perf_metrics(
    trainer: RLTrainer,
    metrics: dict[str, float],
    *,
    train_seconds: float,
    loop_seconds: float,
) -> None:
    if trainer.monitor is None:
        return
    tokens = metrics.get("tokens/train")
    if tokens is None or tokens <= 0:
        return
    perf_metrics = {
        "perf/train_seconds": train_seconds,
        "perf/step_seconds": loop_seconds,
        "perf/train_tokens_per_second": tokens / max(train_seconds, 1e-9),
        "perf/step_tokens_per_second": tokens / max(loop_seconds, 1e-9),
    }
    trainer.monitor.log(perf_metrics, step=trainer.step)


def _rollout_chunk_examples(config: RLConfig) -> int:
    configured = config.orchestrator.rollout_chunk_examples
    if configured is not None:
        return configured
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        return 1
    async_level = max(config.orchestrator.max_async_level, 1)
    return max(1, ceil(examples_per_step / async_level))


def _count_rollout_rows(path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    if rows == 0:
        raise ValueError(f"Rollout chunk '{path}' contains no rows.")
    return rows


if __name__ == "__main__":
    sys.exit(main())

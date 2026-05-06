from __future__ import annotations

import json
import os
import sys
from math import ceil
from pathlib import Path
from time import perf_counter

import torch

from wavelet.configs.rl_config import RLConfig
from wavelet.distributed.world import barrier
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
            try:
                target_step = config.max_steps or 1
                if _use_streaming_rollout_chunks(config):
                    receiver = FileSystemRolloutReceiver(
                        config.output_dir,
                        config.transport,
                        start_step=trainer.step * _chunks_per_step(config),
                    )
                    _run_streaming_rollout_training(
                        config,
                        trainer,
                        receiver,
                        target_step=target_step,
                    )
                    trainer.finalize(status="completed")
                    return 0
                receiver = FileSystemRolloutReceiver(
                    config.output_dir,
                    config.transport,
                    start_step=trainer.step,
                )
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
    chunks_per_step = _chunks_per_step(config)
    min_loadable_rows = _min_loadable_rollout_rows(config, trainer)
    accumulated_rows = 0
    accumulated_chunks = 0
    accumulated_loss_scale = 0.0
    chunk_index = 0
    pending_paths: list[Path] = []
    pending_rows = 0

    while trainer.step < target_step:
        loop_started_at = perf_counter()
        wait_started_at = perf_counter()
        batch = receiver.wait_available()
        wait_seconds = perf_counter() - wait_started_at
        row_count = _count_rollout_rows(batch.path)
        chunk_index += 1
        pending_paths.append(batch.path)
        pending_rows += row_count
        force_partial_load = accumulated_rows > 0 and pending_rows > 0
        if pending_rows < min_loadable_rows and not force_partial_load:
            if _perf_enabled():
                print(
                    "WAVELET_PERF trainer_chunk_buffered "
                    f"queue_step={batch.step} trainer_step={trainer.step} "
                    f"rows={row_count} pending_rows={pending_rows} "
                    f"min_loadable_rows={min_loadable_rows} "
                    f"wait_batch={wait_seconds:.3f}",
                    flush=True,
                )
            continue

        load_started_at = perf_counter()
        rollout_path = _combined_rollout_path(
            config,
            trainer=trainer,
            paths=pending_paths,
            chunk_index=chunk_index,
            min_rows=min_loadable_rows,
        )
        row_count = _count_rollout_rows(rollout_path)
        loaded_chunks = len(pending_paths)
        pending_paths = []
        pending_rows = 0
        trainer.load_rollout_path(rollout_path)
        chunk_loss_scale = trainer._optimizer_batch_loss_scale
        accumulated_rows += row_count
        accumulated_chunks += loaded_chunks
        if chunk_loss_scale is not None:
            accumulated_loss_scale += float(chunk_loss_scale)
        should_step = _should_step_streaming_rollouts(
            accumulated_rows=accumulated_rows,
            accumulated_chunks=accumulated_chunks,
            target_rollout_rows=target_rollout_rows,
            chunks_per_step=chunks_per_step,
        )
        remaining_chunks = max(chunks_per_step - accumulated_chunks, 0)
        loaded_micro_batches = trainer._loaded_micro_batch_count
        if should_step:
            trainer.accumulation_steps = (
                trainer._accumulated_micro_batches + loaded_micro_batches
            )
        else:
            trainer.accumulation_steps = (
                trainer._accumulated_micro_batches
                + loaded_micro_batches * max(remaining_chunks + 1, 2)
            )
        if accumulated_loss_scale > 0.0:
            # Normalize the whole optimizer step by the exact local unmasked
            # token count. Streaming chunks arrive one at a time, so backprop
            # raw chunk losses and divide accumulated gradients once before the
            # optimizer step, when the exact denominator is known.
            trainer._optimizer_batch_loss_scale = 1.0
            trainer._gradient_accumulation_loss_scale = accumulated_loss_scale
        else:
            trainer._optimizer_batch_loss_scale = None
            trainer._gradient_accumulation_loss_scale = None
        load_seconds = perf_counter() - load_started_at
        train_started_at = perf_counter()
        trainer.prepare_for_training()
        metrics = trainer.train_loaded_rollouts_once()
        _validate_distributed_step_sync(trainer, metrics is not None)
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
            accumulated_chunks = 0
            accumulated_loss_scale = 0.0
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
                f"accumulated_chunks={accumulated_chunks} "
                f"wait_batch={wait_seconds:.3f} "
                f"load_rollout={load_seconds:.3f} "
                f"train={train_seconds:.3f} "
                f"export_policy={export_seconds:.3f} "
                f"optimizer_step={int(metrics is not None)} "
                f"total={total_seconds:.3f}",
                flush=True,
            )


def _should_step_streaming_rollouts(
    *,
    accumulated_rows: int,
    accumulated_chunks: int,
    target_rollout_rows: int,
    chunks_per_step: int,
) -> bool:
    _ = accumulated_rows, target_rollout_rows
    return accumulated_chunks >= chunks_per_step


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


def _validate_distributed_step_sync(trainer: RLTrainer, stepped: bool) -> None:
    if not torch.distributed.is_initialized():
        return
    if trainer.world is None:
        raise RuntimeError("World must be set up before distributed step sync.")
    flag = torch.tensor(int(stepped), device=trainer.world.device)
    min_flag = flag.clone()
    max_flag = flag.clone()
    torch.distributed.all_reduce(min_flag, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(max_flag, op=torch.distributed.ReduceOp.MAX)
    if int(min_flag.item()) != int(max_flag.item()):
        raise RuntimeError(
            "Distributed trainer ranks disagreed on optimizer-step completion for "
            "the current rollout chunk. Increase orchestrator.rollout_chunk_examples "
            "or disable sequence packing for smaller chunks."
        )


def _min_loadable_rollout_rows(config: RLConfig, trainer: RLTrainer) -> int:
    if trainer.world is None or config.data.pack_sequences:
        return 1
    return config.data.micro_batch_size * trainer.world.world_size


def _combined_rollout_path(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    paths: list[Path],
    chunk_index: int,
    min_rows: int,
) -> Path:
    row_count = sum(_count_rollout_rows(path) for path in paths)
    target_rows = _padded_row_count(row_count, multiple=min_rows)
    if len(paths) == 1 and row_count == target_rows:
        return paths[0]

    output_dir = config.output_dir / "rollouts" / "combined"
    path = output_dir / f"trainer-step-{trainer.step:06d}-chunk-{chunk_index:06d}.jsonl"
    world = trainer.world
    if world is None or world.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".jsonl.tmp")
        first_row: dict[str, object] | None = None
        written = 0
        with tmp_path.open("w", encoding="utf-8") as output:
            for source_path in paths:
                with source_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        if first_row is None:
                            first_row = json.loads(line)
                        output.write(line)
                        written += 1
            if first_row is None:
                raise ValueError("Cannot pad an empty rollout chunk.")
            for _ in range(target_rows - written):
                output.write(json.dumps(_dummy_rollout_row(config, first_row)) + "\n")
        tmp_path.replace(path)
    if world is not None and world.world_size > 1:
        barrier(world)
    return path


def _padded_row_count(row_count: int, *, multiple: int) -> int:
    if multiple <= 1:
        return row_count
    return ((row_count + multiple - 1) // multiple) * multiple


def _dummy_rollout_row(config: RLConfig, source: dict[str, object]) -> dict[str, object]:
    row = dict(source)
    loss_mask = row.get("loss_mask")
    if not isinstance(loss_mask, list):
        raise ValueError("Cannot create a dummy rollout row without a loss_mask.")
    row["loss_mask"] = [False] * len(loss_mask)
    row[config.data.advantage_column] = 0.0
    row[config.data.reward_column] = None
    row[config.data.temperature_column] = []
    if config.data.inference_logprobs_column in source:
        row[config.data.inference_logprobs_column] = []
    if config.data.teacher_logprobs_column in source:
        row[config.data.teacher_logprobs_column] = []
    metadata = dict(row.get(config.data.metadata_column) or {})
    metadata["_wavelet_dummy_rollout"] = True
    row[config.data.metadata_column] = metadata
    return row


def _rollout_chunk_examples(config: RLConfig) -> int:
    configured = config.orchestrator.rollout_chunk_examples
    if configured is not None:
        return configured
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        return 1
    async_level = max(config.orchestrator.max_async_level, 1)
    return max(1, ceil(examples_per_step / async_level))


def _chunks_per_step(config: RLConfig) -> int:
    examples_per_step = config.orchestrator.examples_per_step
    if examples_per_step is None:
        raise ValueError("orchestrator.examples_per_step is required.")
    return max(ceil(examples_per_step / _rollout_chunk_examples(config)), 1)


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

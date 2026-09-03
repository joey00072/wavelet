from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from wavelet.configs.config import RLConfig, WandbConfig
from wavelet.orchestrator.rollout_metadata import (
    metadata_harness_name,
    metadata_task_name,
)
from wavelet.orchestrator.trace import append_trace_event_best_effort, make_trace_event
from wavelet.trainer.distributed import World, get_world

_SECRET_KEYS = {
    "api_key",
    "access_token",
    "auth_token",
    "credentials",
    "password",
    "secret",
    "token",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_auth_token",
    "_password",
    "_secret",
    "_token",
)
logger = logging.getLogger(__name__)
_TAIL_READ_BLOCK_BYTES = 64 * 1024


def _state_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read dictionary records from a JSONL file.

    Invalid JSON remains an error for full-file readers. This matches the metrics
    ingestion contract, where a corrupt rollout artifact must not be hidden.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(
    path: Path,
    row: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=sort_keys) + "\n")


def _replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def tail_jsonl(
    path: Path,
    *,
    limit: int,
    ignore_errors: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Return the last dictionary records and the number of malformed rows."""
    if limit <= 0 or not path.exists():
        return [], 0
    lines = _tail_nonempty_lines(path, limit=limit)

    rows: list[dict[str, Any]] = []
    parse_errors = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if not ignore_errors:
                raise
            parse_errors += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            parse_errors += 1
    return rows, parse_errors


def _tail_nonempty_lines(path: Path, *, limit: int) -> list[str]:
    selected: list[bytes] = []
    buffered = b""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and len(selected) < limit:
            read_size = min(position, _TAIL_READ_BLOCK_BYTES)
            position -= read_size
            handle.seek(position)
            buffered = handle.read(read_size) + buffered
            parts = buffered.split(b"\n")
            if position > 0:
                buffered = parts[0]
                parts = parts[1:]
            else:
                buffered = b""
            for line in reversed(parts):
                if not line.strip():
                    continue
                selected.append(line)
                if len(selected) == limit:
                    break
    return [line.decode("utf-8") for line in reversed(selected)]


def tail_jsonl_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Return only valid tail records for state and UI endpoints."""
    return tail_jsonl(path, limit=limit)[0]


def redact(value: Any) -> Any:
    """Recursively redact values whose keys look credential-bearing."""
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if _is_secret_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _is_secret_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_KEY_SUFFIXES)


def series_stats(
    prefix: str,
    values: Iterable[float],
    *,
    include_min: bool = True,
) -> dict[str, float]:
    """Summarize a numeric series using the project metric naming convention."""
    items = list(values)
    if not items:
        return {}
    metrics = {
        f"{prefix}/mean": float(mean(items)),
        f"{prefix}/max": float(max(items)),
        f"{prefix}/std": float(pstdev(items)) if len(items) > 1 else 0.0,
    }
    if include_min:
        metrics[f"{prefix}/min"] = float(min(items))
    return metrics


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize values using the diagnostic JSON response shape."""
    items = list(values)
    if not items:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(items),
        "min": float(min(items)),
        "max": float(max(items)),
        "mean": float(mean(items)),
        "std": float(pstdev(items)) if len(items) > 1 else 0.0,
    }


def unavailable_rollout_inspection(
    reason: str,
    *,
    queue_step: int | None,
    path: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable unavailable response used by rollout inspection."""
    return {
        "available": False,
        "reason": reason,
        "queue_step": queue_step,
        "path": None if path is None else str(path),
        "manifest": manifest,
        "scanned_rows": 0,
        "truncated": False,
        "stats": {"reward": summary_stats([]), "advantage": summary_stats([])},
        "samples": {
            "random": [],
            "min_reward": None,
            "max_reward": None,
            "near_mean_reward": None,
        },
    }


def latest_stable_step(
    directory: Path,
    *,
    prefix: str,
    marker: str,
    filename: str,
) -> int | None:
    """Find the latest numerically named stable artifact directory."""
    if not directory.exists():
        return None
    steps: list[int] = []
    for candidate in directory.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue
        if not (candidate / marker).exists() or not (candidate / filename).exists():
            continue
        try:
            steps.append(int(candidate.name.removeprefix(prefix)))
        except ValueError:
            continue
    return max(steps) if steps else None


def perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


def emit_perf(event: str, *, force: bool = False, **fields: object) -> None:
    if not force and not perf_enabled():
        return
    parts = [f"WAVELET_PERF {event}"]
    parts.extend(f"{key}={_format_perf_value(value)}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def _format_perf_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


class RunMonitor:
    def __init__(
        self,
        output_dir: Path,
        *,
        checkpoint_dir: Path | None = None,
        enabled: bool = True,
        write_events: bool = True,
        write_metrics_jsonl: bool = True,
        write_metrics_csv: bool = True,
        write_run_metadata: bool = True,
        write_heartbeat: bool = True,
        log_cuda_memory: bool = True,
        log_disk_usage: bool = True,
        sample_history_size: int = 256,
        wandb: WandbConfig | None = None,
    ) -> None:
        if sample_history_size < 1:
            raise ValueError("sample_history_size must be at least 1.")
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.enabled = enabled
        self.write_events = write_events
        self.write_metrics_jsonl = write_metrics_jsonl
        self.write_metrics_csv = write_metrics_csv
        self.write_run_metadata = write_run_metadata
        self.write_heartbeat = write_heartbeat
        self.log_cuda_memory = log_cuda_memory
        self.log_disk_usage = log_disk_usage
        self.sample_history_size = sample_history_size
        self.wandb = wandb or WandbConfig()
        self.metrics_file = output_dir / "metrics.jsonl"
        self.csv_file = output_dir / "metrics.csv"
        self.samples_file = output_dir / "samples.jsonl"
        self.events_file = output_dir / "events.jsonl"
        self.heartbeat_file = output_dir / "heartbeat.json"
        self.run_metadata_file = output_dir / "run_metadata.json"
        self._wandb_run: Any | None = None
        self._wandb_samples_table: Any | None = None
        self._wandb_samples_columns: list[str] = []
        self._sample_rows_since_compaction: int | None = None

    def start_run(
        self,
        *,
        run_config: dict[str, Any] | None = None,
        world: World | None = None,
        resumed_from: str | None = None,
    ) -> None:
        if not self._should_write():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        world = world or get_world()
        safe_run_config = redact(run_config)

        if self.write_run_metadata:
            metadata = {
                "started_at": self._timestamp(),
                "pid": os.getpid(),
                "output_dir": str(self.output_dir),
                "world": {
                    "rank": world.rank,
                    "local_rank": world.local_rank,
                    "world_size": world.world_size,
                    "local_world_size": world.local_world_size,
                    "device": str(world.device),
                },
                "resumed_from": resumed_from,
                "config": safe_run_config,
            }
            self.run_metadata_file.write_text(json.dumps(metadata, indent=2))

        self._init_wandb(safe_run_config, resumed_from)
        event = "run_resumed" if resumed_from is not None else "run_started"
        self.log_event(event, payload={"resumed_from": resumed_from})
        self._write_heartbeat(status="running", step=None)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self._should_write():
            return

        row = dict(metrics)
        row["step"] = step
        row["timestamp"] = self._timestamp()
        row.update(self._resource_metrics())

        if self.write_metrics_jsonl:
            append_jsonl(self.metrics_file, row)

        if self.write_metrics_csv:
            _append_csv(self.csv_file, row)

        if self._wandb_run is not None:
            wandb_metrics = {k: v for k, v in row.items() if k != "timestamp"}
            wandb_metrics.update(self._wandb_alias_metrics(row))
            self._wandb_run.log(wandb_metrics, step=step)

        self._write_heartbeat(status="running", step=step, metrics=row)

    def log_event(
        self,
        name: str,
        *,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._should_write() or not self.write_events:
            return
        record = {
            "timestamp": self._timestamp(),
            "event": name,
            "step": step,
            "payload": payload or {},
        }
        append_jsonl(self.events_file, record)

    def log_samples(self, samples: list[dict[str, Any]], step: int) -> None:
        if not self._should_write() or not samples:
            return

        timestamp = self._timestamp()
        rows = [{"timestamp": timestamp, "step": step, **sample} for sample in samples]
        self._append_sample_history(rows)

        if self._wandb_run is None:
            return

        columns = ["step", *samples[0].keys()]
        if self._wandb_samples_table is None:
            import wandb

            self._wandb_samples_columns = columns
            self._wandb_samples_table = wandb.Table(
                columns=columns,
                log_mode="INCREMENTAL",
            )
        if columns != self._wandb_samples_columns:
            raise ValueError("Sample log columns changed during the run.")

        for sample in samples:
            self._wandb_samples_table.add_data(
                *[step if column == "step" else sample[column] for column in columns]
            )
        self._wandb_run.log({"samples": self._wandb_samples_table}, step=step)

    def _append_sample_history(self, rows: list[dict[str, Any]]) -> None:
        if self._sample_rows_since_compaction is None:
            retained = tail_jsonl_rows(
                self.samples_file,
                limit=self.sample_history_size,
            )
            _replace_jsonl(
                self.samples_file,
                [*retained, *rows][-self.sample_history_size :],
            )
            self._sample_rows_since_compaction = 0
            return

        for row in rows:
            append_jsonl(self.samples_file, row)
        self._sample_rows_since_compaction += len(rows)
        if self._sample_rows_since_compaction < self.sample_history_size:
            return
        _replace_jsonl(
            self.samples_file,
            tail_jsonl_rows(self.samples_file, limit=self.sample_history_size),
        )
        self._sample_rows_since_compaction = 0

    def finish(
        self,
        *,
        status: str,
        step: int | None,
    ) -> None:
        if not self._should_write():
            return
        self.log_event("run_finished", step=step, payload={"status": status})
        self._write_heartbeat(status=status, step=step)
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _should_write(self) -> bool:
        if not self.enabled:
            return False
        world = get_world()
        return world.is_main

    def _resource_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if self.log_cuda_memory:
            metrics.update(self._cuda_metrics())
        if self.log_disk_usage:
            metrics.update(self._disk_metrics())
        return metrics

    def _cuda_metrics(self) -> dict[str, Any]:
        metric_names = (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        )
        available = torch.cuda.is_available()
        return {
            f"cuda_{name}_bytes": getattr(torch.cuda, name)() if available else None
            for name in metric_names
        }

    def _disk_metrics(self) -> dict[str, Any]:
        usage = shutil.disk_usage(_existing_path(self.output_dir))
        metrics: dict[str, Any] = {
            "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used,
            "disk_free_bytes": usage.free,
            "disk_free_ratio": usage.free / usage.total if usage.total else 0.0,
        }
        if self.checkpoint_dir is not None:
            checkpoint_usage = shutil.disk_usage(_existing_path(self.checkpoint_dir))
            metrics.update(
                {
                    "checkpoint_disk_total_bytes": checkpoint_usage.total,
                    "checkpoint_disk_used_bytes": checkpoint_usage.used,
                    "checkpoint_disk_free_bytes": checkpoint_usage.free,
                    "checkpoint_disk_free_ratio": (
                        checkpoint_usage.free / checkpoint_usage.total
                        if checkpoint_usage.total
                        else 0.0
                    ),
                }
            )
        return metrics

    def _write_heartbeat(
        self,
        *,
        status: str,
        step: int | None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.write_heartbeat:
            return
        payload = {
            "timestamp": self._timestamp(),
            "pid": os.getpid(),
            "status": status,
            "step": step,
        }
        if metrics is not None:
            payload["metrics"] = metrics
        self.heartbeat_file.write_text(json.dumps(payload, indent=2))

    def _init_wandb(
        self,
        run_config: dict[str, Any] | None,
        resumed_from: str | None,
    ) -> None:
        if not self.wandb.enabled or self.wandb.mode == "disabled":
            return
        import wandb

        init_kwargs = {
            "project": self.wandb.project or "wavelet",
            "entity": self.wandb.entity,
            "name": self.wandb.name,
            "group": self.wandb.group,
            "tags": self.wandb.tags,
            "dir": str(self.output_dir),
            "config": run_config,
            "resume": "allow" if resumed_from is not None else None,
        }
        settings_factory = getattr(wandb, "Settings", None)
        if callable(settings_factory):
            init_kwargs["settings"] = settings_factory(
                init_timeout=self.wandb.init_timeout_seconds
            )
        try:
            self._wandb_run = wandb.init(mode=self.wandb.mode, **init_kwargs)
        except Exception:
            if self.wandb.mode != "online" or not self.wandb.offline_fallback:
                raise
            logger.warning(
                "W&B online initialization failed; falling back to offline W&B "
                "logging in %s.",
                self.output_dir,
                exc_info=True,
            )
            self._wandb_run = wandb.init(mode="offline", **init_kwargs)
        self._wandb_run.define_metric("step")
        self._wandb_run.define_metric("*", step_metric="step")

    @staticmethod
    def _wandb_alias_metrics(row: dict[str, Any]) -> dict[str, Any]:
        aliases: dict[str, Any] = {}
        if "reward_mean" in row:
            aliases["reward"] = row["reward_mean"]
            aliases["train/reward_mean"] = row["reward_mean"]
        if "reward/all/mean" in row:
            aliases["rollout/reward_mean"] = row["reward/all/mean"]
        if "rollout/count" in row:
            aliases["rollout/count"] = row["rollout/count"]
        if "loss" in row:
            aliases["train/loss"] = row["loss"]
        if "lr" in row:
            aliases["train/lr"] = row["lr"]
            aliases["scheduler/lr"] = row["lr"]
        if "optim/lr" in row:
            aliases.setdefault("train/lr", row["optim/lr"])
            aliases.setdefault("scheduler/lr", row["optim/lr"])
        return aliases

    def _timestamp(self) -> str:
        return datetime.now(UTC).isoformat()


def _existing_path(path: Path) -> Path:
    """Return the nearest existing path whose filesystem contains ``path``."""
    cursor = path
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    return cursor


def setup_logger(name: str, level: str = "info") -> Any:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


_WANDB_RUN = None


@dataclass(frozen=True)
class RolloutMetricInputs:
    rows: list[dict[str, Any]]
    rollouts_per_example: int
    step: int
    policy_step: int | None = None
    queue_step: int | None = None
    optimizer_step: int | None = None
    chunk_index: int | None = None
    timings: dict[str, float] | None = None
    extra_metrics: dict[str, float] | None = None


def log_rollout_metrics(
    config: RLConfig,
    path: Path,
    *,
    step: int,
    policy_step: int | None = None,
    queue_step: int | None = None,
    optimizer_step: int | None = None,
    chunk_index: int | None = None,
    timings: dict[str, float] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    rows = read_jsonl(path)
    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=rows,
            rollouts_per_example=config.orchestrator.rollouts_per_example or 1,
            step=step,
            policy_step=policy_step,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            timings=timings,
            extra_metrics=extra_metrics,
        )
    )
    _append_metrics(config.output_dir, metrics, step=step)
    _append_rollout_trace(
        config,
        rows,
        metrics,
        step=step,
        policy_step=policy_step,
        queue_step=queue_step,
        optimizer_step=optimizer_step,
    )
    _wandb_log(config, metrics, step=step)
    return metrics


def log_eval_metrics(
    config: RLConfig,
    metrics: dict[str, float],
    *,
    step: int,
    policy_step: int,
) -> None:
    """Publish fixed-policy evaluation metrics to the orchestrator monitor."""
    _wandb_log(config, metrics, step=step)
    append_trace_event_best_effort(
        config.output_dir,
        make_trace_event(
            subsystem="orchestrator",
            event="eval_metrics_logged",
            step=step,
            optimizer_step=step,
            policy_step=policy_step,
            details={
                key: value for key, value in metrics.items() if key.startswith("eval/")
            },
        ),
    )


def _append_rollout_trace(
    config: RLConfig,
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
    *,
    step: int,
    policy_step: int | None,
    queue_step: int | None,
    optimizer_step: int | None,
) -> None:
    task_names = sorted(
        {
            name
            for row in rows
            if (name := metadata_task_name(_metadata(row))) is not None
        }
    )
    harness_names = sorted(
        {
            name
            for row in rows
            if (name := metadata_harness_name(_metadata(row))) is not None
        }
    )
    append_trace_event_best_effort(
        config.output_dir,
        make_trace_event(
            subsystem="orchestrator",
            event="rollout_metrics_logged",
            step=step,
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            policy_step=policy_step,
            task=",".join(task_names) if task_names else None,
            harness=",".join(harness_names) if harness_names else None,
            details={
                "samples": metrics["progress/samples"],
                "tokens": metrics["progress/tokens"],
                "trainable": metrics["fate/all/trainable"],
                "filtered": metrics["fate/all/filtered"],
                "errored": metrics["fate/all/errored"],
            },
        ),
    )


def rollout_metrics(inputs: RolloutMetricInputs) -> dict[str, float]:
    rows = inputs.rows
    grouped = _group_by_example(rows)
    seq_lens = [_seq_len(row) for row in rows]
    decode_lens = [_decode_len(row) for row in rows]
    prefill_lens = [
        max(seq_len - decode_len, 0)
        for seq_len, decode_len in zip(seq_lens, decode_lens, strict=True)
    ]
    advantages = [_float_or_none(row.get("advantage")) for row in rows]
    metrics: dict[str, float] = {
        "progress/tokens": float(sum(seq_lens)),
        "progress/prefill_tokens": float(sum(prefill_lens)),
        "progress/decode_tokens": float(sum(decode_lens)),
        "progress/samples": float(len(rows)),
        "progress/problems": float(len(grouped)),
        "progress/ckpt_step": float(
            inputs.policy_step if inputs.policy_step is not None else inputs.step
        ),
        "progress/queue_step": float(
            inputs.queue_step if inputs.queue_step is not None else inputs.step
        ),
        "progress/optimizer_step": float(
            inputs.optimizer_step if inputs.optimizer_step is not None else inputs.step
        ),
        "filters/all/is_filtered": float(mean(_filtered_flags(rows))) if rows else 0.0,
        "step": float(inputs.step),
    }
    if inputs.policy_step is not None:
        metrics["policy/step"] = float(inputs.policy_step)
        metrics["policy/lag"] = float(inputs.step - inputs.policy_step)
    if inputs.chunk_index is not None:
        metrics["progress/chunk_index"] = float(inputs.chunk_index)

    for name, value_fn, include_min in (
        (
            "prefill_len",
            lambda row: max(_seq_len(row) - _decode_len(row), 0),
            True,
        ),
        (
            "is_truncated",
            lambda row: _bool_metric(_metadata(row).get("is_truncated")),
            False,
        ),
        ("samples_per_rollout", _sample_count, True),
        ("num_turns", _turn_count, True),
    ):
        metrics.update(
            series_stats(
                f"{name}/all",
                _grouped_means(grouped, value_fn),
                include_min=include_min,
            )
        )
    advantage_values = [value for value in advantages if value is not None]
    metrics.update(series_stats("advantage/all", advantage_values))
    _add_group_metrics(
        metrics,
        rows,
        grouped,
        suffix="all",
        rollouts_per_example=inputs.rollouts_per_example,
    )
    _add_environment_metrics(
        metrics,
        rows,
        rollouts_per_example=inputs.rollouts_per_example,
    )

    if inputs.timings:
        for key, value in inputs.timings.items():
            metrics[f"time/{key}"] = float(value)

    if inputs.extra_metrics:
        duplicate_keys = metrics.keys() & inputs.extra_metrics.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys))
            raise ValueError(
                f"Extra rollout metrics replace core metrics: {duplicates}"
            )
        metrics.update(
            {key: float(value) for key, value in inputs.extra_metrics.items()}
        )

    return metrics


def _add_environment_metrics(
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
    *,
    rollouts_per_example: int,
) -> None:
    for env_name, env_rows in _group_by_env(rows).items():
        grouped = _group_by_example(env_rows)
        metrics[f"batch/{env_name}"] = len(env_rows) / max(len(rows), 1)
        _add_group_metrics(
            metrics,
            env_rows,
            grouped,
            suffix=env_name,
            rollouts_per_example=rollouts_per_example,
        )


def _add_group_metrics(
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    suffix: str,
    rollouts_per_example: int,
) -> None:
    metrics.update(_fate_metrics(f"fate/{suffix}", _fate_counts(rows)))
    for name, value_fn in (
        ("seq_len", _seq_len),
        ("decode_len", _decode_len),
    ):
        metrics.update(
            series_stats(f"{name}/{suffix}", _grouped_means(grouped, value_fn))
        )
    metrics.update(
        series_stats(
            f"reward/{suffix}",
            _grouped_rollout_means(
                grouped,
                lambda row: _float_or_none(row.get("reward")),
            ),
        )
    )
    solve_none, solve_all, effective = _solve_rates(grouped, rollouts_per_example)
    metrics.update(
        {
            f"solve_none/{suffix}": solve_none,
            f"solve_all/{suffix}": solve_all,
            f"effective_batch_size/{suffix}": effective,
        }
    )
    _add_stop_condition_metrics(metrics, rows, prefix=f"stop_condition/{suffix}")


def _add_stop_condition_metrics(
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
    *,
    prefix: str,
) -> None:
    stop_conditions = [_metadata(row).get("stop_condition") for row in rows]
    truncated = [_bool_metric(_metadata(row).get("is_truncated")) for row in rows]
    generation_truncated = [
        flag and stop_condition != "prompt_too_long"
        for flag, stop_condition in zip(truncated, stop_conditions, strict=True)
    ]
    metrics[f"{prefix}/generation_truncated"] = (
        float(mean(generation_truncated)) if generation_truncated else 0.0
    )
    rates = _category_rates(value for value in stop_conditions if value is not None)
    for condition, rate in rates.items():
        metrics[f"{prefix}/{condition}"] = rate


def _append_metrics(output_dir: Path, metrics: dict[str, float], *, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": step,
        **metrics,
    }
    jsonl_path = output_dir / "orchestrator_metrics.jsonl"
    append_jsonl(jsonl_path, row, sort_keys=True)

    csv_path = output_dir / "orchestrator_metrics.csv"
    _append_csv(csv_path, row)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    headers = list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_headers = list(reader.fieldnames or [])
            existing_rows = list(reader)
        new_headers = [key for key in headers if key not in existing_headers]
        if not new_headers:
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=existing_headers).writerow(row)
            return
        headers = [*existing_headers, *new_headers]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def _wandb_log(config: RLConfig, metrics: dict[str, float], *, step: int) -> None:
    global _WANDB_RUN
    wandb_config = config.monitor.wandb
    if not wandb_config.enabled or wandb_config.mode == "disabled":
        return
    try:
        import wandb
    except ImportError:
        return
    try:
        if _WANDB_RUN is None:
            run_name = wandb_config.name
            _WANDB_RUN = wandb.init(
                project=wandb_config.project or "wavelet",
                entity=wandb_config.entity,
                name=f"{run_name}-orchestrator" if run_name else None,
                group=wandb_config.group or run_name,
                tags=wandb_config.tags,
                mode=wandb_config.mode,
                dir=str(config.output_dir),
                config=redact(config.model_dump(mode="json")),
            )
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        _WANDB_RUN.log({**metrics, "step": step}, step=step)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        logger.warning("Failed to log orchestrator metrics to W&B: %s", exc)


def _group_by_example(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        env_name = str(row.get("env_name") or row.get("source") or "all")
        example_id = str(
            _metadata(row).get("group_key") or row.get("example_id") or len(grouped)
        )
        grouped[(env_name, example_id)].append(row)
    return dict(grouped)


def _group_by_env(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("env_name") or row.get("source") or "all")].append(row)
    return dict(grouped)


def _grouped_means(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    value_fn,
) -> list[float]:
    values: list[float] = []
    for rows in grouped.values():
        row_values = [_float_or_none(value_fn(row)) for row in rows]
        numeric = [value for value in row_values if value is not None]
        if numeric:
            values.append(float(mean(numeric)))
    return values


def _grouped_rollout_means(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    value_fn,
) -> list[float]:
    values: list[float] = []
    for rows in grouped.values():
        weighted_values = [
            (value, max(_sample_count(row), 0))
            for row in rows
            if (value := _float_or_none(value_fn(row))) is not None
        ]
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight > 0:
            values.append(
                sum(value * weight for value, weight in weighted_values) / total_weight
            )
    return values


def _solve_rates(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    rollouts_per_example: int,
) -> tuple[float, float, float]:
    if not grouped:
        return 0.0, 0.0, 0.0
    reward_sums = []
    for rows in grouped.values():
        reward_sums.append(
            sum(
                (_float_or_none(row.get("reward")) or 0.0) * max(_sample_count(row), 0)
                for row in rows
            )
        )
    solve_none = sum(value == 0.0 for value in reward_sums) / len(reward_sums)
    solve_all = sum(value >= rollouts_per_example for value in reward_sums) / len(
        reward_sums
    )
    return solve_none, solve_all, 1.0 - solve_none - solve_all


def _category_rates(values) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for value in values:
        counts[str(value)] += 1
        total += 1
    if total == 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _seq_len(row: dict[str, Any]) -> int:
    input_ids = row.get("input_ids")
    if isinstance(input_ids, list):
        return len(input_ids)
    return _decode_len(row)


def _decode_len(row: dict[str, Any]) -> int:
    loss_mask = row.get("loss_mask")
    if isinstance(loss_mask, list):
        return sum(bool(item) for item in loss_mask)
    inference_logprobs = row.get("inference_logprobs")
    if isinstance(inference_logprobs, list):
        return len(inference_logprobs)
    metadata = _metadata(row)
    value = metadata.get("completion_token_count")
    return int(value) if isinstance(value, int | float) else 0


def _sample_count(row: dict[str, Any]) -> int:
    metadata = _metadata(row)
    value = metadata.get("_wavelet_rollout_count", 1)
    return int(value) if isinstance(value, int | float) else 1


def _turn_count(row: dict[str, Any]) -> int:
    metadata = _metadata(row)
    value = metadata.get("turn_count")
    if isinstance(value, int | float):
        return int(value)
    completion = row.get("completion")
    if isinstance(completion, list):
        return len(completion)
    return 1


def _filtered_flags(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(bool(_metadata(row).get("_wavelet_filtered_rollout"))) for row in rows
    ]


def _fate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    names = (
        "trainable",
        "zero_loss",
        "filtered",
        "dummy",
        "errored",
        "truncated",
        "with_inference_logprobs",
        "with_teacher_logprobs",
    )
    counts = {"produced": len(rows), **dict.fromkeys(names, 0)}
    for row in rows:
        metadata = _metadata(row)
        trainable_tokens = _decode_len(row)
        flags = (
            trainable_tokens > 0,
            trainable_tokens == 0,
            metadata.get("_wavelet_filtered_rollout"),
            metadata.get("_wavelet_dummy_rollout"),
            _has_error(row),
            metadata.get("is_truncated"),
            isinstance(row.get("inference_logprobs"), list),
            isinstance(row.get("teacher_logprobs"), list),
        )
        for name, flag in zip(names, flags, strict=True):
            counts[name] += int(bool(flag))
    return counts


def _fate_metrics(prefix: str, counts: dict[str, int]) -> dict[str, float]:
    total = max(counts["produced"], 1)
    metrics: dict[str, float] = {}
    for name, count in counts.items():
        metrics[f"{prefix}/{name}"] = float(count)
        if name != "produced":
            metrics[f"{prefix}/{name}_rate"] = float(count / total)
    return metrics


def _has_error(row: dict[str, Any]) -> bool:
    if row.get("error") is not None:
        return True
    return _metadata(row).get("error") is not None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _bool_metric(value: object) -> float:
    return float(bool(value))


# Shared run-state JSON and rollout inspection readers.
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_jsonl_dicts(
    path: Path, *, limit: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    if limit <= 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row_index, row


def _numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _message_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(f"{role}: {content}" if role else content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts) if parts else None
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _compact_rollout_row(
    row: dict[str, Any],
    *,
    row_index: int,
    max_text_chars: int,
) -> dict[str, Any]:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    sample = {
        "row_index": row_index,
        "reward": _numeric(row, "reward"),
        "advantage": _numeric(row, "advantage"),
        "source": row.get("source"),
        "env_name": row.get("env_name"),
        "task": row.get("task"),
        "example_id": row.get("example_id"),
        "group_key": metadata.get("group_key"),
        "rollout_key": metadata.get("rollout_key"),
        "stop_condition": metadata.get("stop_condition"),
        "is_truncated": metadata.get("is_truncated"),
        "completion_token_count": metadata.get("completion_token_count"),
        "turn_count": metadata.get("turn_count"),
        "prompt": _message_text(row.get("prompt")),
        "completion": _message_text(row.get("completion")),
        "target_completion": _message_text(row.get("target_completion")),
    }
    for key in ("prompt", "completion", "target_completion"):
        value = sample.get(key)
        if isinstance(value, str) and len(value) > max_text_chars:
            sample[key] = value[:max_text_chars] + "...<truncated>"
    return {key: value for key, value in sample.items() if value is not None}


@dataclass
class _RolloutScan:
    reward_values: list[float]
    advantage_values: list[float]
    random_rows: list[dict[str, Any]]
    min_reward_row: dict[str, Any] | None
    max_reward_row: dict[str, Any] | None
    scanned_rows: int


def _scan_rollouts(
    path: Path,
    *,
    random_count: int,
    seed: int | None,
    max_scan_rows: int,
    max_text_chars: int,
) -> _RolloutScan:
    rng = random.Random(seed)
    rewards: list[float] = []
    advantages: list[float] = []
    random_rows: list[dict[str, Any]] = []
    min_pair: tuple[float, dict[str, Any]] | None = None
    max_pair: tuple[float, dict[str, Any]] | None = None
    scanned_rows = 0

    for scanned_rows, (row_index, row) in enumerate(
        _iter_jsonl_dicts(path, limit=max_scan_rows),
        start=1,
    ):
        compact = _compact_rollout_row(
            row,
            row_index=row_index,
            max_text_chars=max_text_chars,
        )
        reward = _numeric(row, "reward")
        if reward is not None:
            rewards.append(reward)
            if min_pair is None or reward < min_pair[0]:
                min_pair = (reward, compact)
            if max_pair is None or reward > max_pair[0]:
                max_pair = (reward, compact)

        advantage = _numeric(row, "advantage")
        if advantage is not None:
            advantages.append(advantage)

        if len(random_rows) < random_count:
            random_rows.append(compact)
        elif random_count > 0:
            replacement = rng.randint(0, scanned_rows - 1)
            if replacement < random_count:
                random_rows[replacement] = compact

    return _RolloutScan(
        reward_values=rewards,
        advantage_values=advantages,
        random_rows=random_rows,
        min_reward_row=None if min_pair is None else min_pair[1],
        max_reward_row=None if max_pair is None else max_pair[1],
        scanned_rows=scanned_rows,
    )


def _nearest_reward_row(
    path: Path,
    *,
    target: float | None,
    max_scan_rows: int,
    max_text_chars: int,
) -> dict[str, Any] | None:
    if target is None:
        return None
    nearest: tuple[float, dict[str, Any]] | None = None
    for row_index, row in _iter_jsonl_dicts(path, limit=max_scan_rows):
        reward = _numeric(row, "reward")
        if reward is None:
            continue
        distance = abs(reward - target)
        if nearest is None or distance < nearest[0]:
            nearest = (
                distance,
                _compact_rollout_row(
                    row,
                    row_index=row_index,
                    max_text_chars=max_text_chars,
                ),
            )
    return None if nearest is None else nearest[1]


def _file_exceeds_rows(path: Path, *, limit: int) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return any(index >= limit for index, _ in enumerate(handle))
    except OSError:
        return False


class RolloutStateEventsMixin:
    """Shared bounded rollout event-tail updates for run-state services."""

    def mark_submitted(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        pending_count: int,
    ) -> None:
        self._mark_rollout(
            "submitted",
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            pending_count=pending_count,
        )

    def mark_completed(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        pending_count: int,
        completed_count: int,
    ) -> None:
        self._mark_rollout(
            "completed",
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            pending_count=pending_count,
            completed_count=completed_count,
        )

    def mark_published(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        path: str,
        next_queue_step_to_publish: int,
        completed_count: int,
    ) -> None:
        self._mark_rollout(
            "published",
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            chunk_index=chunk_index,
            path=path,
            next_queue_step_to_publish=next_queue_step_to_publish,
            completed_count=completed_count,
        )

    def _mark_rollout(
        self,
        event_type: str,
        *,
        queue_step: int,
        optimizer_step: int | None,
        chunk_index: int | None,
        path: str | None = None,
        **rollout_patch: Any,
    ) -> None:
        item = {
            "type": event_type,
            "queue_step": queue_step,
            "optimizer_step": optimizer_step,
            "chunk_index": chunk_index,
            "timestamp": _state_timestamp(),
        }
        if path is not None:
            item["path"] = path
        self._append_tail(f"{event_type}_tail", item, **rollout_patch)

    def _append_tail(
        self, key: str, item: dict[str, Any], **rollout_patch: Any
    ) -> None:
        with self._lock:
            tail = list(self._state["rollouts"].get(key, []))
            tail.append(item)
            self._state["rollouts"][key] = tail[-20:]
            events = list(self._state.get("events", []))
            events.append(item)
            self._state["events"] = events[
                -self._config.orchestrator.state_server.max_events :
            ]
            self._state["rollouts"].update(rollout_patch)
            self._state["updated_at"] = _state_timestamp()

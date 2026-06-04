from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from wavelet.configs.rl_config import RLConfig

logger = logging.getLogger(__name__)

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
) -> dict[str, float]:
    rows = _read_jsonl(path)
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
        )
    )
    _append_metrics(config.output_dir, metrics, step=step)
    _wandb_log(config, metrics, step=step)
    return metrics


def rollout_metrics(inputs: RolloutMetricInputs) -> dict[str, float]:
    rows = inputs.rows
    grouped = _group_by_example(rows)
    seq_lens = [_seq_len(row) for row in rows]
    decode_lens = [_decode_len(row) for row in rows]
    prefill_lens = [max(seq_len - decode_len, 0) for seq_len, decode_len in zip(seq_lens, decode_lens, strict=True)]
    rewards = [_float_or_none(row.get("reward")) for row in rows]
    advantages = [_float_or_none(row.get("advantage")) for row in rows]
    is_truncated = [_bool_metric(_metadata(row).get("is_truncated")) for row in rows]
    sample_counts = [_sample_count(row) for row in rows]
    turn_counts = [_turn_count(row) for row in rows]
    fate_counts = _fate_counts(rows)

    metrics: dict[str, float] = {
        "progress/tokens": float(sum(seq_lens)),
        "progress/prefill_tokens": float(sum(prefill_lens)),
        "progress/decode_tokens": float(sum(decode_lens)),
        "progress/samples": float(len(rows)),
        "progress/problems": float(len(grouped)),
        "progress/ckpt_step": float(inputs.policy_step if inputs.policy_step is not None else inputs.step),
        "progress/queue_step": float(inputs.queue_step if inputs.queue_step is not None else inputs.step),
        "progress/optimizer_step": float(inputs.optimizer_step if inputs.optimizer_step is not None else inputs.step),
        "filters/all/is_filtered": float(mean(_filtered_flags(rows))) if rows else 0.0,
        "step": float(inputs.step),
    }
    metrics.update(_fate_metrics("fate/all", fate_counts))
    if inputs.policy_step is not None:
        metrics["policy/step"] = float(inputs.policy_step)
        metrics["policy/lag"] = float(inputs.step - inputs.policy_step)
    if inputs.chunk_index is not None:
        metrics["progress/chunk_index"] = float(inputs.chunk_index)

    metrics.update(_series_stats("seq_len/all", _grouped_means(grouped, _seq_len)))
    metrics.update(_series_stats("prefill_len/all", _grouped_means_from_values(grouped, prefill_lens)))
    metrics.update(_series_stats("decode_len/all", _grouped_means_from_values(grouped, decode_lens)))
    metrics.update(_series_stats("is_truncated/all", _grouped_means_from_values(grouped, is_truncated), include_min=False))
    metrics.update(_series_stats("samples_per_rollout/all", _grouped_means_from_values(grouped, sample_counts)))
    metrics.update(_series_stats("num_turns/all", _grouped_means_from_values(grouped, turn_counts)))

    reward_by_example = _grouped_means_from_values(grouped, rewards)
    metrics.update(_series_stats("reward/all", reward_by_example))
    advantage_values = [value for value in advantages if value is not None]
    metrics.update(_series_stats("advantage/all", advantage_values))

    solve_none, solve_all, effective = _solve_rates(grouped, inputs.rollouts_per_example)
    metrics["solve_none/all"] = solve_none
    metrics["solve_all/all"] = solve_all
    metrics["effective_batch_size/all"] = effective

    stop_conditions = [_metadata(row).get("stop_condition") for row in rows]
    generation_truncated = [
        truncated and stop_condition != "prompt_too_long"
        for truncated, stop_condition in zip(is_truncated, stop_conditions, strict=True)
    ]
    metrics["stop_condition/all/generation_truncated"] = float(mean(generation_truncated)) if generation_truncated else 0.0
    for condition, rate in _category_rates(value for value in stop_conditions if value is not None).items():
        metrics[f"stop_condition/all/{condition}"] = rate

    for env_name, env_rows in _group_by_env(rows).items():
        env_grouped = _group_by_example(env_rows)
        metrics[f"batch/{env_name}"] = len(env_rows) / max(len(rows), 1)
        metrics.update(_fate_metrics(f"fate/{env_name}", _fate_counts(env_rows)))
        metrics.update(_series_stats(f"seq_len/{env_name}", _grouped_means(env_grouped, _seq_len)))
        metrics.update(_series_stats(f"decode_len/{env_name}", _grouped_means(env_grouped, _decode_len)))
        metrics.update(_series_stats(f"reward/{env_name}", _grouped_means(env_grouped, lambda row: _float_or_none(row.get("reward")))))
        env_solve_none, env_solve_all, env_effective = _solve_rates(env_grouped, inputs.rollouts_per_example)
        metrics[f"solve_none/{env_name}"] = env_solve_none
        metrics[f"solve_all/{env_name}"] = env_solve_all
        metrics[f"effective_batch_size/{env_name}"] = env_effective
        env_truncated = [_bool_metric(_metadata(row).get("is_truncated")) for row in env_rows]
        env_stop = [_metadata(row).get("stop_condition") for row in env_rows]
        metrics[f"stop_condition/{env_name}/generation_truncated"] = (
            float(mean([flag and sc != "prompt_too_long" for flag, sc in zip(env_truncated, env_stop, strict=True)]))
            if env_rows
            else 0.0
        )
        for condition, rate in _category_rates(value for value in env_stop if value is not None).items():
            metrics[f"stop_condition/{env_name}/{condition}"] = rate

    if inputs.timings:
        for key, value in inputs.timings.items():
            metrics[f"time/{key}"] = float(value)

    return metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_metrics(output_dir: Path, metrics: dict[str, float], *, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        **metrics,
    }
    jsonl_path = output_dir / "orchestrator_metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

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
                config=config.model_dump(mode="json"),
            )
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        _WANDB_RUN.log({**metrics, "step": step}, step=step)
    except Exception as exc:  # pragma: no cover - diagnostics must not kill training
        logger.warning("Failed to log orchestrator metrics to W&B: %s", exc)


def _group_by_example(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        env_name = str(row.get("env_name") or row.get("source") or "all")
        example_id = str(row.get("example_id") or _metadata(row).get("group_key") or len(grouped))
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


def _grouped_means_from_values(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    values: list[float | int | bool | None],
) -> list[float]:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    row_index = 0
    for key, rows in grouped.items():
        for _ in rows:
            value = _float_or_none(values[row_index])
            if value is not None:
                by_key[key].append(value)
            row_index += 1
    return [float(mean(items)) for items in by_key.values() if items]


def _series_stats(
    prefix: str,
    values: list[float],
    *,
    include_min: bool = True,
) -> dict[str, float]:
    if not values:
        return {}
    metrics = {
        f"{prefix}/mean": float(mean(values)),
        f"{prefix}/max": float(max(values)),
    }
    if include_min:
        metrics[f"{prefix}/min"] = float(min(values))
    if len(values) > 1:
        metrics[f"{prefix}/std"] = float(pstdev(values))
    else:
        metrics[f"{prefix}/std"] = 0.0
    return metrics


def _solve_rates(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    rollouts_per_example: int,
) -> tuple[float, float, float]:
    if not grouped:
        return 0.0, 0.0, 0.0
    reward_sums = []
    for rows in grouped.values():
        reward_sums.append(sum(_float_or_none(row.get("reward")) or 0.0 for row in rows))
    solve_none = sum(value == 0.0 for value in reward_sums) / len(reward_sums)
    solve_all = sum(value >= rollouts_per_example for value in reward_sums) / len(reward_sums)
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
    return [float(bool(_metadata(row).get("_wavelet_filtered_rollout"))) for row in rows]


def _fate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "produced": len(rows),
        "trainable": 0,
        "zero_loss": 0,
        "filtered": 0,
        "dummy": 0,
        "errored": 0,
        "truncated": 0,
        "with_inference_logprobs": 0,
        "with_teacher_logprobs": 0,
    }
    for row in rows:
        metadata = _metadata(row)
        trainable_tokens = _decode_len(row)
        counts["trainable"] += int(trainable_tokens > 0)
        counts["zero_loss"] += int(trainable_tokens == 0)
        counts["filtered"] += int(bool(metadata.get("_wavelet_filtered_rollout")))
        counts["dummy"] += int(bool(metadata.get("_wavelet_dummy_rollout")))
        counts["errored"] += int(_has_error(row))
        counts["truncated"] += int(bool(metadata.get("is_truncated")))
        counts["with_inference_logprobs"] += int(
            isinstance(row.get("inference_logprobs"), list)
        )
        counts["with_teacher_logprobs"] += int(
            isinstance(row.get("teacher_logprobs"), list)
        )
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

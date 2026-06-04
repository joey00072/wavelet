from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import get_step_dir, resolve_queue_dir


def inspect_rollout_batch(
    config: RLConfig,
    *,
    rollout_path: Path | None = None,
    queue_step: int | None = None,
    max_rows: int | None = None,
    sample_limit: int = 3,
) -> dict[str, Any]:
    path = _resolve_rollout_path(config, rollout_path=rollout_path, queue_step=queue_step)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows_scanned = 0
    rows_with_trainable_tokens = 0
    total_tokens = 0
    trainable_tokens = 0
    rows_with_inference_logprobs = 0
    rows_with_teacher_logprobs = 0

    for row_index, payload in _iter_jsonl(path):
        if max_rows is not None and rows_scanned >= max_rows:
            break
        rows_scanned += 1
        row_errors, row_warnings, stats = _inspect_row(
            payload,
            row_index=row_index,
            inference_logprobs_column=config.data.inference_logprobs_column,
            teacher_logprobs_column=config.data.teacher_logprobs_column,
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        total_tokens += stats["tokens"]
        trainable_tokens += stats["trainable_tokens"]
        rows_with_trainable_tokens += int(stats["trainable_tokens"] > 0)
        rows_with_inference_logprobs += int(stats["has_inference_logprobs"])
        rows_with_teacher_logprobs += int(stats["has_teacher_logprobs"])
        if len(rows) < sample_limit:
            rows.append(_sample_row(payload, row_index=row_index, stats=stats))

    return {
        "path": str(path),
        "queue_step": queue_step,
        "rows_scanned": rows_scanned,
        "truncated": max_rows is not None and rows_scanned >= max_rows,
        "summary": {
            "total_tokens": total_tokens,
            "trainable_tokens": trainable_tokens,
            "rows_with_trainable_tokens": rows_with_trainable_tokens,
            "rows_with_inference_logprobs": rows_with_inference_logprobs,
            "rows_with_teacher_logprobs": rows_with_teacher_logprobs,
        },
        "errors": errors,
        "warnings": warnings,
        "samples": rows,
        "ok": not errors,
    }


def build_runtime_parity_report(
    config: RLConfig,
    *,
    rollout_path: Path | None = None,
    queue_step: int | None = None,
    trainer_logprobs_column: str = "trainer_logprobs",
    max_rows: int | None = None,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    path = _resolve_rollout_path(config, rollout_path=rollout_path, queue_step=queue_step)
    errors: list[dict[str, Any]] = []
    rows_checked = 0
    token_count = 0
    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    rows_missing_trainer_logprobs = 0
    rows_missing_inference_logprobs = 0

    for row_index, payload in _iter_jsonl(path):
        if max_rows is not None and rows_checked >= max_rows:
            break
        if "_wavelet_parse_error" in payload:
            errors.append(
                {
                    "row": row_index,
                    "field": "jsonl",
                    "message": str(payload["_wavelet_parse_error"]),
                }
            )
            continue
        rows_checked += 1
        loss_mask = _sequence(payload.get("loss_mask"))
        trainable_tokens = sum(bool(value) for value in loss_mask or [])
        inference_logprobs = _float_sequence(
            payload.get(config.data.inference_logprobs_column)
        )
        trainer_logprobs = _float_sequence(payload.get(trainer_logprobs_column))
        if inference_logprobs is None:
            rows_missing_inference_logprobs += 1
            continue
        if trainer_logprobs is None:
            rows_missing_trainer_logprobs += 1
            continue
        if len(inference_logprobs) != trainable_tokens:
            errors.append(
                {
                    "row": row_index,
                    "field": config.data.inference_logprobs_column,
                    "message": "logprob count must match trainable token count",
                }
            )
            continue
        if len(trainer_logprobs) != trainable_tokens:
            errors.append(
                {
                    "row": row_index,
                    "field": trainer_logprobs_column,
                    "message": "trainer logprob count must match trainable token count",
                }
            )
            continue
        for inference_logprob, trainer_logprob in zip(
            inference_logprobs,
            trainer_logprobs,
            strict=True,
        ):
            diff = abs(trainer_logprob - inference_logprob)
            max_abs_diff = max(max_abs_diff, diff)
            abs_diff_sum += diff
            token_count += 1

    skipped = token_count == 0 and not errors
    return {
        "path": str(path),
        "queue_step": queue_step,
        "threshold": threshold,
        "rows_checked": rows_checked,
        "token_count": token_count,
        "max_abs_diff": max_abs_diff if token_count else None,
        "mean_abs_diff": abs_diff_sum / token_count if token_count else None,
        "passed": bool(token_count and max_abs_diff <= threshold and not errors),
        "skipped": skipped,
        "skip_reason": _parity_skip_reason(
            rows_checked=rows_checked,
            rows_missing_inference_logprobs=rows_missing_inference_logprobs,
            rows_missing_trainer_logprobs=rows_missing_trainer_logprobs,
            token_count=token_count,
            errors=errors,
        ),
        "coverage": {
            "rows_missing_inference_logprobs": rows_missing_inference_logprobs,
            "rows_missing_trainer_logprobs": rows_missing_trainer_logprobs,
        },
        "errors": errors,
    }


def export_rollout_token_debug(
    config: RLConfig,
    *,
    write_path: Path,
    rollout_path: Path | None = None,
    queue_step: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    path = _resolve_rollout_path(config, rollout_path=rollout_path, queue_step=queue_step)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows_scanned = 0
    rows_exported = 0
    total_tokens = 0
    trainable_tokens = 0
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as handle:
        for row_index, payload in _iter_jsonl(path):
            if max_rows is not None and rows_scanned >= max_rows:
                break
            rows_scanned += 1
            row_errors, row_warnings, stats = _inspect_row(
                payload,
                row_index=row_index,
                inference_logprobs_column=config.data.inference_logprobs_column,
                teacher_logprobs_column=config.data.teacher_logprobs_column,
            )
            errors.extend(row_errors)
            warnings.extend(row_warnings)
            total_tokens += stats["tokens"]
            trainable_tokens += stats["trainable_tokens"]
            if row_errors:
                continue
            token_row = _token_debug_row(
                payload,
                row_index=row_index,
                inference_logprobs_column=config.data.inference_logprobs_column,
                teacher_logprobs_column=config.data.teacher_logprobs_column,
            )
            handle.write(json.dumps(token_row, sort_keys=True) + "\n")
            rows_exported += 1

    return {
        "path": str(path),
        "write_path": str(write_path),
        "queue_step": queue_step,
        "rows_scanned": rows_scanned,
        "rows_exported": rows_exported,
        "truncated": max_rows is not None and rows_scanned >= max_rows,
        "summary": {
            "total_tokens": total_tokens,
            "trainable_tokens": trainable_tokens,
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def _resolve_rollout_path(
    config: RLConfig,
    *,
    rollout_path: Path | None,
    queue_step: int | None,
) -> Path:
    if rollout_path is not None:
        return Path(rollout_path)
    if queue_step is None:
        raise ValueError("Either rollout_path or queue_step is required.")
    queue_dir = resolve_queue_dir(config.output_dir, config.transport)
    return get_step_dir(queue_dir, queue_step) / config.transport.rollout_filename


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                yield row_index, {"_wavelet_parse_error": str(exc)}
                continue
            if not isinstance(payload, dict):
                yield row_index, {"_wavelet_parse_error": "row is not a JSON object"}
                continue
            yield row_index, payload


def _inspect_row(
    payload: dict[str, Any],
    *,
    row_index: int,
    inference_logprobs_column: str,
    teacher_logprobs_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if "_wavelet_parse_error" in payload:
        errors.append(
            {
                "row": row_index,
                "field": "jsonl",
                "message": str(payload["_wavelet_parse_error"]),
            }
        )
        return errors, warnings, _empty_stats()

    input_ids = _sequence(payload.get("input_ids"))
    target_ids = _sequence(payload.get("target_ids"))
    loss_mask = _sequence(payload.get("loss_mask"))
    if input_ids is None or target_ids is None or loss_mask is None:
        errors.append(
            {
                "row": row_index,
                "field": "input_ids/target_ids/loss_mask",
                "message": "pretokenized rows must include input_ids, target_ids, and loss_mask",
            }
        )
        return errors, warnings, _empty_stats()

    if not (len(input_ids) == len(target_ids) == len(loss_mask)):
        errors.append(
            {
                "row": row_index,
                "field": "input_ids/target_ids/loss_mask",
                "message": "input_ids, target_ids, and loss_mask lengths differ",
                "lengths": {
                    "input_ids": len(input_ids),
                    "target_ids": len(target_ids),
                    "loss_mask": len(loss_mask),
                },
            }
        )
    trainable_tokens = sum(bool(value) for value in loss_mask)
    if trainable_tokens == 0:
        warnings.append(
            {
                "row": row_index,
                "field": "loss_mask",
                "message": "row has no trainable tokens",
            }
        )

    inference_logprobs = _sequence(payload.get(inference_logprobs_column))
    teacher_logprobs = _sequence(payload.get(teacher_logprobs_column))
    _check_logprob_alignment(
        errors,
        row_index=row_index,
        field=inference_logprobs_column,
        values=inference_logprobs,
        trainable_tokens=trainable_tokens,
    )
    _check_logprob_alignment(
        errors,
        row_index=row_index,
        field=teacher_logprobs_column,
        values=teacher_logprobs,
        trainable_tokens=trainable_tokens,
    )
    return errors, warnings, {
        "tokens": len(input_ids),
        "trainable_tokens": trainable_tokens,
        "has_inference_logprobs": inference_logprobs is not None,
        "has_teacher_logprobs": teacher_logprobs is not None,
    }


def _check_logprob_alignment(
    errors: list[dict[str, Any]],
    *,
    row_index: int,
    field: str,
    values: list[Any] | None,
    trainable_tokens: int,
) -> None:
    if values is None:
        return
    if len(values) != trainable_tokens:
        errors.append(
            {
                "row": row_index,
                "field": field,
                "message": "logprob count must match trainable token count",
                "lengths": {
                    field: len(values),
                    "trainable_tokens": trainable_tokens,
                },
            }
        )


def _sample_row(
    payload: dict[str, Any],
    *,
    row_index: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return {
        "row": row_index,
        "example_id": _metadata_value(metadata, "example_id"),
        "trajectory_id": _metadata_value(metadata, "trajectory_id"),
        "tokens": stats["tokens"],
        "trainable_tokens": stats["trainable_tokens"],
        "reward": payload.get("reward"),
    }


def _token_debug_row(
    payload: dict[str, Any],
    *,
    row_index: int,
    inference_logprobs_column: str,
    teacher_logprobs_column: str,
) -> dict[str, Any]:
    input_ids = _sequence(payload.get("input_ids")) or []
    target_ids = _sequence(payload.get("target_ids")) or []
    loss_mask = [bool(value) for value in _sequence(payload.get("loss_mask")) or []]
    trainable_indexes = [
        index for index, trainable in enumerate(loss_mask) if bool(trainable)
    ]
    metadata = payload.get("metadata")
    return {
        "row": row_index,
        "example_id": _metadata_value(metadata, "example_id"),
        "trajectory_id": _metadata_value(metadata, "trajectory_id"),
        "rollout_key": _metadata_value(metadata, "rollout_key"),
        "input_ids": [int(value) for value in input_ids],
        "target_ids": [int(value) for value in target_ids],
        "loss_mask": loss_mask,
        "trainable_indexes": trainable_indexes,
        "trainable_target_ids": [int(target_ids[index]) for index in trainable_indexes],
        "reward": payload.get("reward"),
        "advantage": payload.get("advantage"),
        "inference_logprobs": _float_sequence(
            payload.get(inference_logprobs_column)
        ),
        "teacher_logprobs": _float_sequence(payload.get(teacher_logprobs_column)),
        "temperatures": _float_sequence(payload.get("temperatures"))
        or _float_sequence(payload.get("temperature")),
    }


def _metadata_value(metadata: object, key: str) -> object | None:
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _sequence(value: object) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _float_sequence(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def _parity_skip_reason(
    *,
    rows_checked: int,
    rows_missing_inference_logprobs: int,
    rows_missing_trainer_logprobs: int,
    token_count: int,
    errors: list[dict[str, Any]],
) -> str | None:
    if errors or token_count:
        return None
    if rows_checked == 0:
        return "no rows checked"
    if rows_missing_inference_logprobs == rows_checked:
        return "all checked rows are missing inference logprobs"
    if rows_missing_trainer_logprobs == rows_checked:
        return "all checked rows are missing trainer logprobs"
    return "no comparable logprob pairs"


def _empty_stats() -> dict[str, Any]:
    return {
        "tokens": 0,
        "trainable_tokens": 0,
        "has_inference_logprobs": False,
        "has_teacher_logprobs": False,
    }

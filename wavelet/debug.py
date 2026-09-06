from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from wavelet.configs.config import CustomAlgorithmConfig, RLAlgorithmConfig, RLConfig
from wavelet.data.rl import RLExample, load_rl_records
from wavelet.monitor import RolloutMetricInputs, _existing_path, rollout_metrics
from wavelet.orchestrator.algorithms import build_algorithm
from wavelet.orchestrator.placement import (
    device_group_conflict_error,
    device_group_size,
    device_groups,
    http_ports,
    required_inference_devices,
    rollout_reward_mode_error,
    trainer_device_group,
)
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import (
    chunks_per_step,
    required_policy_step,
    rollout_chunk_examples,
    target_steps,
)
from wavelet.transport.queue import (
    get_step_dir,
    resolve_policy_dir,
    resolve_queue_dir,
)

DEBUG_COMMANDS = {
    "preflight": (
        "_preflight_main",
        "Validate cheap RL launch prerequisites",
    ),
    "inference": (
        "_inference_main",
        "Inspect and probe RL inference",
    ),
    "orchestrator": (
        "_orchestrator_main",
        "Inspect and benchmark RL orchestration",
    ),
    "trainer": (
        "_trainer_main",
        "Inspect trainer inputs and replay artifacts",
    ),
}

CheckStatus = Literal["ok", "warning", "error"]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: wavelet debug <subcommand> [args]")
        print("Subcommands:")
        for command, (_, description) in DEBUG_COMMANDS.items():
            print(f"  {command:<14} {description}")
        return 1

    command = argv[0]
    if command not in DEBUG_COMMANDS:
        print(f"Unknown debug subcommand: {command}")
        return 1

    entrypoint_name, _ = DEBUG_COMMANDS[command]
    return globals()[entrypoint_name](argv[1:])


def _preflight_main(argv: list[str]) -> int:
    from wavelet.utils.config import load_config

    parser = argparse.ArgumentParser(
        prog="wavelet debug preflight",
        description="Validate cheap RL launch prerequisites without starting workers.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    report = build_preflight_report(load_config(RLConfig, config_args))
    _print_report(report, json_output=args.json_output, text=_print_preflight_text)
    return 0 if report["ok"] else 1


def _inference_main(argv: list[str]) -> int:
    from wavelet.utils.config import load_config

    parser = _inference_parser()
    args, config_args = parser.parse_known_args(argv)
    _validate_inference_args(args)

    config = load_config(RLConfig, config_args)
    if args.action == "inspect":
        _print_report(inference_debug_state(config), json_output=args.json_output)
        return 0
    if args.action == "health":
        _print_report({"health": http_health(config)}, json_output=args.json_output)
        return 0
    if args.action == "continuous-batch":
        return _run_continuous_batch_probe(config, args)
    return _run_inference_engine_probe(config, args)


def _inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavelet debug inference",
        description="Inspect and probe RL inference without trainer/orchestrator.",
    )
    parser.add_argument(
        "action",
        choices=["inspect", "health", "smoke", "benchmark", "continuous-batch"],
        help="Diagnostic action to run.",
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Answer with the single word ok.",
        help="Prompt text used by smoke and benchmark.",
    )
    parser.add_argument("--policy-dir", default=None)
    parser.add_argument("--policy-step", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--stagger-ms", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


_INFERENCE_ARG_MINIMUMS = (
    ("count", 1),
    ("warmup", 0),
    ("repeats", 1),
    ("concurrency", 1),
    ("stagger_ms", 0),
    ("max_completion_tokens", 1),
)


def _validate_inference_args(args: argparse.Namespace) -> None:
    for name, minimum in _INFERENCE_ARG_MINIMUMS:
        if getattr(args, name) < minimum:
            flag = name.replace("_", "-")
            raise SystemExit(f"--{flag} must be >= {minimum}")
    if args.data_parallel_size is not None and args.data_parallel_size < 1:
        raise SystemExit("--data-parallel-size must be >= 1")


def _run_continuous_batch_probe(config: Any, args: argparse.Namespace) -> int:
    requests, metrics = continuous_batch_probe(
        config,
        count=args.count,
        concurrency=args.concurrency,
        prompt=args.prompt,
        max_completion_tokens=args.max_completion_tokens,
        stagger_seconds=args.stagger_ms / 1000.0,
        data_parallel_size=args.data_parallel_size,
    )
    _print_report(
        {
            "metrics": metrics.to_dict(),
            "sample": [request.to_dict() for request in requests[:5]],
            "errors": [
                request.to_dict() for request in requests if request.error is not None
            ][:5],
        },
        json_output=args.json_output,
    )
    return 0


def _run_inference_engine_probe(config: Any, args: argparse.Namespace) -> int:
    from wavelet.inference.policy import create_policy_inference_engine

    records = make_probe_examples(count=args.count, prompt=args.prompt)
    engine = create_policy_inference_engine(config)
    try:
        engine.setup()
        if args.policy_dir is not None:
            engine.load_policy(Path(args.policy_dir), step=args.policy_step)
        annotated, metrics = probe_engine(
            engine,
            records,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    finally:
        engine.close()
    _print_report(
        {
            "metrics": metrics.to_dict(),
            "sample": _sample_records(annotated, limit=3),
        },
        json_output=args.json_output,
    )
    return 0


def _orchestrator_main(argv: list[str]) -> int:
    from wavelet.inference.policy import create_policy_inference_engine
    from wavelet.utils.config import load_config

    parser = argparse.ArgumentParser(
        prog="wavelet debug orchestrator",
        description="Inspect and benchmark RL orchestration without trainer.",
    )
    parser.add_argument(
        "action",
        choices=["inspect", "sample", "benchmark", "materialize"],
        help="Diagnostic action to run.",
    )
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Do not create a policy inference engine for benchmark/materialize.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    if args.examples is not None and args.examples < 1:
        raise SystemExit("--examples must be >= 1")
    if args.rollouts is not None and args.rollouts < 1:
        raise SystemExit("--rollouts must be >= 1")
    if args.retry < 0:
        raise SystemExit("--retry must be >= 0")

    config = with_orchestrator_limits(
        load_config(RLConfig, config_args),
        examples=args.examples,
        rollouts=args.rollouts,
    )
    if args.action == "inspect":
        _print_report(orchestrator_debug_state(config), json_output=args.json_output)
        return 0
    if args.action == "sample":
        _print_report(
            sample_orchestrator_records(config, step=args.step, retry=args.retry),
            json_output=args.json_output,
        )
        return 0

    engine = None
    if not args.no_inference:
        engine = create_policy_inference_engine(config)
        engine.setup()
    try:
        probe = probe_orchestrator(
            config,
            step=args.step,
            retry=args.retry,
            inference_engine=engine,
            write=args.action == "materialize",
        )
    finally:
        if engine is not None:
            engine.close()
    _print_report(probe.to_dict(), json_output=args.json_output)
    return 0


def _trainer_main(argv: list[str]) -> int:
    from wavelet.utils.config import load_config

    parser = argparse.ArgumentParser(
        prog="wavelet debug trainer",
        description="Inspect trainer inputs without launching training.",
    )
    parser.add_argument(
        "action",
        choices=["inspect", "parity", "tokens"],
        help="Diagnostic action to run.",
    )
    parser.add_argument("--rollout-path", type=Path, default=None)
    parser.add_argument("--queue-step", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--trainer-logprobs-column", default="trainer_logprobs")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--write-report", type=Path, default=None)
    parser.add_argument("--write-tokens", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1")
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit must be >= 0")
    if args.threshold < 0.0:
        raise SystemExit("--threshold must be >= 0")
    if args.action == "tokens" and args.write_tokens is None:
        raise SystemExit("tokens action requires --write-tokens")

    config = load_config(RLConfig, config_args)
    if args.action == "inspect":
        report = inspect_rollout_batch(
            config,
            rollout_path=args.rollout_path,
            queue_step=args.queue_step,
            max_rows=args.max_rows,
            sample_limit=args.sample_limit,
        )
    elif args.action == "parity":
        report = build_runtime_parity_report(
            config,
            rollout_path=args.rollout_path,
            queue_step=args.queue_step,
            trainer_logprobs_column=args.trainer_logprobs_column,
            max_rows=args.max_rows,
            threshold=args.threshold,
        )
    else:
        report = export_rollout_token_debug(
            config,
            write_path=args.write_tokens,
            rollout_path=args.rollout_path,
            queue_step=args.queue_step,
            max_rows=args.max_rows,
        )
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _print_report(report, json_output=args.json_output, text=_print_trainer_text)
    return _trainer_exit_code(report, action=args.action)


def _sample_records(records: list[Any], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "completion": (
                record.completion[0].get("content", "") if record.completion else ""
            ),
            "trainable_tokens": sum(record.loss_mask or []),
            "has_inference_logprobs": record.inference_logprobs is not None,
        }
        for record in records[:limit]
    ]


def _print_report(
    report: dict[str, Any],
    *,
    json_output: bool,
    text: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif text is not None:
        text(report)
    else:
        _print_nested_text(report)


def _print_nested_text(report: dict[str, Any]) -> None:
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for child_key, child_value in value.items():
                print(f"  {child_key}: {child_value}")
        else:
            print(f"{key}: {value}")


def _print_preflight_text(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "failed"
    print(f"preflight: {status}")
    print("summary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print("checks:")
    for check in report["checks"]:
        print(f"  [{check['status']}] {check['name']}: {check['message']}")
    print("commands:")
    for command in report["commands"]:
        role = command["role"]
        command_text = command["command"]
        print(f"  {role}: {command_text}")
        for key, value in command.items():
            if key in {"role", "command"}:
                continue
            print(f"    {key}: {value}")


def _print_trainer_text(report: dict[str, Any]) -> None:
    print(f"path: {report['path']}")
    _print_present_fields(report, ("ok", "passed", "skipped"))
    for key, value in report.get("summary", {}).items():
        print(f"{key}: {value}")
    _print_present_fields(
        report,
        (
            "rows_exported",
            "write_path",
            "token_count",
            "max_abs_diff",
            "mean_abs_diff",
            "skip_reason",
        ),
    )
    _print_report_issues("errors", report["errors"])
    _print_report_issues("warnings", report["warnings"])


def _print_present_fields(report: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in report:
            print(f"{key}: {report[key]}")


def _print_report_issues(label: str, issues: list[dict[str, Any]]) -> None:
    if not issues:
        return
    print(f"{label}:")
    for issue in issues:
        print(f"  row {issue.get('row')}: {issue.get('field')}: {issue.get('message')}")


def _trainer_exit_code(report: dict[str, Any], *, action: str) -> int:
    if action in {"inspect", "tokens"}:
        return 0 if report["ok"] else 1
    if report["errors"]:
        return 1
    if report["skipped"]:
        return 0
    return 0 if report["passed"] else 1


# Consolidated from wavelet/trainer/diagnostics.py.
def inspect_rollout_batch(
    config: RLConfig,
    *,
    rollout_path: Path | None = None,
    queue_step: int | None = None,
    max_rows: int | None = None,
    sample_limit: int = 3,
) -> dict[str, Any]:
    path = _resolve_rollout_path(
        config, rollout_path=rollout_path, queue_step=queue_step
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows_scanned = 0
    rows_with_trainable_tokens = 0
    total_tokens = 0
    trainable_tokens = 0
    rows_with_inference_logprobs = 0
    rows_with_teacher_logprobs = 0

    for row_index, payload in _iter_jsonl_or_errors(path):
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
    path = _resolve_rollout_path(
        config, rollout_path=rollout_path, queue_step=queue_step
    )
    errors: list[dict[str, Any]] = []
    rows_checked = 0
    token_count = 0
    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    rows_missing_trainer_logprobs = 0
    rows_missing_inference_logprobs = 0

    for row_index, payload in _iter_jsonl_or_errors(path):
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
    path = _resolve_rollout_path(
        config, rollout_path=rollout_path, queue_step=queue_step
    )
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows_scanned = 0
    rows_exported = 0
    total_tokens = 0
    trainable_tokens = 0
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("w", encoding="utf-8") as handle:
        for row_index, payload in _iter_jsonl_or_errors(path):
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


def _iter_jsonl_or_errors(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
    return (
        errors,
        warnings,
        {
            "tokens": len(input_ids),
            "trainable_tokens": trainable_tokens,
            "has_inference_logprobs": inference_logprobs is not None,
            "has_teacher_logprobs": teacher_logprobs is not None,
        },
    )


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
    metadata = _metadata(payload)
    return {
        "row": row_index,
        "example_id": metadata.get("example_id"),
        "trajectory_id": metadata.get("trajectory_id"),
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
    metadata = _metadata(payload)
    return {
        "row": row_index,
        "example_id": metadata.get("example_id"),
        "trajectory_id": metadata.get("trajectory_id"),
        "rollout_key": metadata.get("rollout_key"),
        "input_ids": [int(value) for value in input_ids],
        "target_ids": [int(value) for value in target_ids],
        "loss_mask": loss_mask,
        "trainable_indexes": trainable_indexes,
        "trainable_target_ids": [int(target_ids[index]) for index in trainable_indexes],
        "reward": payload.get("reward"),
        "advantage": payload.get("advantage"),
        "inference_logprobs": _float_sequence(payload.get(inference_logprobs_column)),
        "teacher_logprobs": _float_sequence(payload.get(teacher_logprobs_column)),
        "temperatures": _float_sequence(payload.get("temperatures"))
        or _float_sequence(payload.get("temperature")),
    }


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


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


# Consolidated from wavelet/inference/diagnostics.py.
class _AsDictMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceProbeMetrics(_AsDictMixin):
    records: int
    wall_seconds: float
    records_per_second: float
    model_input_tokens: int
    completion_tokens: int
    trainable_tokens: int
    model_input_tokens_per_second: float
    completion_tokens_per_second: float
    trainable_tokens_per_second: float
    records_with_completion: int
    records_with_inference_logprobs: int
    records_with_loss_mask: int
    min_completion_tokens: int
    max_completion_tokens: int
    mean_completion_tokens: float


@dataclass(frozen=True)
class ContinuousBatchRequest(_AsDictMixin):
    index: int
    base_url: str
    data_parallel_rank: int | None
    ok: bool
    latency_seconds: float
    start_offset_seconds: float
    end_offset_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class ContinuousBatchMetrics(_AsDictMixin):
    requests: int
    succeeded: int
    failed: int
    concurrency: int
    stagger_seconds: float
    wall_seconds: float
    requests_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_per_second: float
    total_tokens_per_second: float
    latency_p50_seconds: float
    latency_p90_seconds: float
    latency_max_seconds: float
    max_observed_concurrency: int
    overlapped: bool
    per_rank_requests: dict[str, int]


def base_urls(config: RLConfig) -> list[str]:
    ports = config.inference.http.ports or [config.inference.http.port]
    return [f"http://{config.inference.http.host}:{port}" for port in ports]


def inference_debug_state(config: RLConfig) -> dict[str, Any]:
    return {
        "model": {
            "name": config.model.name,
            "torch_dtype": config.model.torch_dtype,
            "adapter_path": str(config.model.adapter_path)
            if config.model.adapter_path is not None
            else None,
            "load_in_4bit": config.model.load_in_4bit,
        },
        "inference": {
            "mode": config.inference.mode,
            "enabled": config.inference.enabled,
            "http_base_urls": base_urls(config)
            if config.inference.mode == "vllm_http"
            else [],
            "server_backend": config.inference.vllm.server_backend,
            "tensor_parallel_size": config.inference.vllm.tensor_parallel_size,
            "data_parallel_size": config.inference.vllm.data_parallel_size,
            "data_parallel_size_local": config.inference.vllm.data_parallel_size_local,
            "max_model_len": config.inference.vllm.max_model_len,
            "gpu_memory_utilization": config.inference.vllm.gpu_memory_utilization,
            "quantization": config.inference.vllm.quantization,
            "load_format": config.inference.vllm.load_format,
            "use_generation_logprobs": config.inference.vllm.use_generation_logprobs,
        },
        "sampling": config.inference.sampling.model_dump(mode="json"),
        "lora": None
        if config.lora is None
        else config.lora.model_dump(mode="json", exclude_none=True),
        "policy_transfer": config.policy_transfer.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "output_dir": str(config.output_dir),
    }


def make_probe_examples(*, count: int, prompt: str) -> list[RLExample]:
    return [
        RLExample(
            prompt=[{"role": "user", "content": f"{prompt} #{index}"}],
            completion=[{"role": "assistant", "content": ""}],
            advantage=None,
            reward=None,
            metadata={"probe_index": index},
            source="inference_probe",
        )
        for index in range(count)
    ]


def probe_engine(
    engine: Any,
    records: list[RLExample],
    *,
    warmup: int = 0,
    repeats: int = 1,
) -> tuple[list[RLExample], InferenceProbeMetrics]:
    if warmup > 0:
        engine.annotate(records[:warmup])
    started_at = time.perf_counter()
    annotated: list[RLExample] = []
    for _ in range(repeats):
        annotated.extend(engine.annotate(records))
    wall_seconds = time.perf_counter() - started_at
    return annotated, summarize_records(annotated, wall_seconds=wall_seconds)


def summarize_records(
    records: list[RLExample],
    *,
    wall_seconds: float,
) -> InferenceProbeMetrics:
    model_input_tokens = sum(len(record.input_ids or []) for record in records)
    trainable_tokens = sum(sum(record.loss_mask or []) for record in records)
    completion_lengths = [
        sum(record.loss_mask or [])
        for record in records
        if record.completion and record.completion[0].get("content")
    ]
    completion_tokens = sum(completion_lengths)
    records_with_completion = len(completion_lengths)
    records_with_inference_logprobs = sum(
        1 for record in records if record.inference_logprobs is not None
    )
    records_with_loss_mask = sum(
        1 for record in records if record.loss_mask is not None
    )
    wall = max(wall_seconds, 1e-9)
    return InferenceProbeMetrics(
        records=len(records),
        wall_seconds=wall_seconds,
        records_per_second=len(records) / wall,
        model_input_tokens=model_input_tokens,
        completion_tokens=completion_tokens,
        trainable_tokens=trainable_tokens,
        model_input_tokens_per_second=model_input_tokens / wall,
        completion_tokens_per_second=completion_tokens / wall,
        trainable_tokens_per_second=trainable_tokens / wall,
        records_with_completion=records_with_completion,
        records_with_inference_logprobs=records_with_inference_logprobs,
        records_with_loss_mask=records_with_loss_mask,
        min_completion_tokens=min(completion_lengths, default=0),
        max_completion_tokens=max(completion_lengths, default=0),
        mean_completion_tokens=(
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else 0.0
        ),
    )


def http_health(config: RLConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for base_url in base_urls(config):
        started_at = time.perf_counter()
        try:
            payload = _http_json(base_url, "GET", "/health")
            try:
                debug_state = _http_json(base_url, "GET", "/debug/state")
            except (OSError, RuntimeError):
                debug_state = None
            results.append(
                {
                    "base_url": base_url,
                    "ok": True,
                    "latency_seconds": time.perf_counter() - started_at,
                    "response": payload,
                    "debug_state": debug_state,
                }
            )
        except (OSError, RuntimeError) as exc:
            results.append(
                {
                    "base_url": base_url,
                    "ok": False,
                    "latency_seconds": time.perf_counter() - started_at,
                    "error": str(exc),
                }
            )
    return results


def continuous_batch_probe(
    config: RLConfig,
    *,
    count: int,
    concurrency: int,
    prompt: str,
    max_completion_tokens: int,
    stagger_seconds: float = 0.0,
    data_parallel_size: int | None = None,
) -> tuple[list[ContinuousBatchRequest], ContinuousBatchMetrics]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be >= 1")
    if stagger_seconds < 0.0:
        raise ValueError("stagger_seconds must be >= 0")

    routes = _continuous_batch_routes(config, data_parallel_size=data_parallel_size)
    model_name = config.orchestrator.verifier_model or config.model.name
    sampling = config.inference.sampling
    started_at = time.perf_counter()

    def run_request(index: int) -> ContinuousBatchRequest:
        scheduled_at = started_at + index * stagger_seconds
        sleep_seconds = scheduled_at - time.perf_counter()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        base_url, headers = routes[index % len(routes)]
        dp_rank = headers.get("X-data-parallel-rank")
        request_started_at = time.perf_counter()
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt} #{index}",
                }
            ],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }
        try:
            response = _http_json(
                base_url,
                "POST",
                _chat_completions_path(base_url),
                payload,
                headers=headers,
                timeout=config.inference.http.request_timeout_seconds,
            )
            usage = response.get("usage") or {}
            ok = True
            error = None
        except (OSError, RuntimeError) as exc:
            usage = {}
            ok = False
            error = str(exc)
        request_ended_at = time.perf_counter()
        return ContinuousBatchRequest(
            index=index,
            base_url=base_url,
            data_parallel_rank=int(dp_rank) if dp_rank is not None else None,
            ok=ok,
            latency_seconds=request_ended_at - request_started_at,
            start_offset_seconds=request_started_at - started_at,
            end_offset_seconds=request_ended_at - started_at,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            error=error,
        )

    max_workers = min(count, concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_request, index) for index in range(count)]
        requests = [future.result() for future in as_completed(futures)]
    requests.sort(key=lambda item: item.index)
    wall_seconds = time.perf_counter() - started_at
    metrics = _continuous_batch_metrics(
        requests,
        concurrency=concurrency,
        stagger_seconds=stagger_seconds,
        wall_seconds=wall_seconds,
    )
    return requests, metrics


def _http_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if headers is not None:
        request_headers.update(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _continuous_batch_routes(
    config: RLConfig,
    *,
    data_parallel_size: int | None,
) -> list[tuple[str, dict[str, str]]]:
    base_url_value = config.orchestrator.verifier_base_url
    if base_url_value is None:
        urls = [f"{url}/v1" for url in base_urls(config)]
    elif isinstance(base_url_value, str):
        urls = [base_url_value]
    else:
        urls = list(base_url_value)
    if not urls:
        raise ValueError("at least one inference base URL is required")

    dp_size = data_parallel_size or config.inference.vllm.data_parallel_size
    routes: list[tuple[str, dict[str, str]]] = []
    for base_url in urls:
        for dp_rank in range(dp_size):
            headers = {"X-data-parallel-rank": str(dp_rank)} if dp_size > 1 else {}
            routes.append((base_url.rstrip("/"), headers))
    return routes


def _chat_completions_path(base_url: str) -> str:
    return "/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions"


def _continuous_batch_metrics(
    requests: list[ContinuousBatchRequest],
    *,
    concurrency: int,
    stagger_seconds: float,
    wall_seconds: float,
) -> ContinuousBatchMetrics:
    succeeded = sum(1 for request in requests if request.ok)
    prompt_tokens = sum(request.prompt_tokens for request in requests)
    completion_tokens = sum(request.completion_tokens for request in requests)
    total_tokens = sum(request.total_tokens for request in requests)
    latencies = [request.latency_seconds for request in requests if request.ok]
    per_rank_requests = dict(
        Counter(
            "none"
            if request.data_parallel_rank is None
            else str(request.data_parallel_rank)
            for request in requests
        )
    )
    wall = max(wall_seconds, 1e-9)
    max_observed_concurrency = _max_observed_concurrency(requests)
    return ContinuousBatchMetrics(
        requests=len(requests),
        succeeded=succeeded,
        failed=len(requests) - succeeded,
        concurrency=concurrency,
        stagger_seconds=stagger_seconds,
        wall_seconds=wall_seconds,
        requests_per_second=len(requests) / wall,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_per_second=completion_tokens / wall,
        total_tokens_per_second=total_tokens / wall,
        latency_p50_seconds=_quantile(latencies, 0.50),
        latency_p90_seconds=_quantile(latencies, 0.90),
        latency_max_seconds=max(latencies, default=0.0),
        max_observed_concurrency=max_observed_concurrency,
        overlapped=max_observed_concurrency > 1,
        per_rank_requests=per_rank_requests,
    )


def _max_observed_concurrency(requests: list[ContinuousBatchRequest]) -> int:
    events: list[tuple[float, int]] = []
    for request in requests:
        events.append((request.start_offset_seconds, 1))
        events.append((request.end_offset_seconds, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        active += delta
        max_active = max(max_active, active)
    return max_active


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


# Consolidated from wavelet/orchestrator/diagnostics.py.
@dataclass(frozen=True)
class OrchestratorProbe(_AsDictMixin):
    timings: dict[str, float]
    records_available: int
    records_selected: int
    records_scored: int
    records_trainable: int
    rollouts_per_example: int
    examples_per_step: int | None
    token_batch_size: int | None
    metrics: dict[str, float]
    output_path: str | None


def orchestrator_debug_state(config: RLConfig) -> dict[str, Any]:
    schedule: dict[str, Any] = {
        "target_steps": target_steps(config),
        "examples_per_step": config.orchestrator.examples_per_step,
        "token_batch_size": config.orchestrator.token_batch_size,
        "rollouts_per_example": config.orchestrator.rollouts_per_example,
        "max_async_level": config.orchestrator.max_async_level,
        "max_off_policy_steps": config.orchestrator.max_off_policy_steps,
        "required_policy_step_at_rollout_0": required_policy_step(config, 0),
        "required_policy_step_at_rollout_1": required_policy_step(config, 1),
    }
    if config.orchestrator.concurrency is not None:
        schedule["concurrency"] = config.orchestrator.concurrency.model_dump(
            mode="json",
            exclude_none=True,
        )
    if config.orchestrator.examples_per_step is not None:
        schedule["rollout_chunk_examples"] = rollout_chunk_examples(config)
        schedule["chunks_per_step"] = chunks_per_step(config)
        schedule["rollouts_per_optimizer_step"] = (
            config.orchestrator.examples_per_step
            * (config.orchestrator.rollouts_per_example or 1)
        )
    elif config.orchestrator.token_batch_size is not None:
        schedule["chunks_per_step"] = chunks_per_step(config)
    return {
        "algo": config.algo.model_dump(mode="json", exclude_none=True),
        "data": {
            "source": config.data.source,
            "path": str(config.data.path),
            "seed": config.data.seed,
            "seq_len": config.data.seq_len,
            "batch_size": config.data.batch_size,
            "micro_batch_size": config.data.micro_batch_size,
        },
        "orchestrator": {
            "enabled": config.orchestrator.enabled,
            "custom_rollout_function": config.orchestrator.custom_rollout_function,
            "verifier_env_id": config.orchestrator.verifier_env_id,
            "envs": [
                env.model_dump(mode="json", exclude_none=True)
                for env in config.orchestrator.envs
            ],
            "curriculum": (
                None
                if config.orchestrator.curriculum is None
                else config.orchestrator.curriculum.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "verifier_model": config.orchestrator.verifier_model,
            "verifier_client_type": config.orchestrator.verifier_client_type,
            "filter_zero_advantage": config.orchestrator.filter_zero_advantage,
            "zero_advantage_max_retries": config.orchestrator.zero_advantage_max_retries,
            "oversampling_factor": config.orchestrator.oversampling_factor,
        },
        "reward": config.reward.model_dump(mode="json"),
        "schedule": schedule,
        "transport": config.transport.model_dump(mode="json", exclude_none=True),
        "output_dir": str(config.output_dir),
    }


def sample_orchestrator_records(
    config: RLConfig,
    *,
    step: int | None,
    retry: int = 0,
) -> dict[str, Any]:
    orchestrator = RLOrchestrator(config)
    started_at = time.perf_counter()
    all_records = load_rl_records(config.data)
    load_seconds = time.perf_counter() - started_at
    started_at = time.perf_counter()
    records = orchestrator._select_step_records(
        all_records,
        seed=orchestrator._step_seed(step=step, retry=retry),
    )
    select_seconds = time.perf_counter() - started_at
    seconds = load_seconds + select_seconds
    return {
        "records_available": len(all_records),
        "records": len(records),
        "timings": {
            "load_records": load_seconds,
            "select_records": select_seconds,
            "total": seconds,
        },
        "records_per_second": len(records) / max(seconds, 1e-9),
        "sample": [_record_summary(record) for record in records[:5]],
    }


def probe_orchestrator(
    config: RLConfig,
    *,
    step: int | None,
    retry: int = 0,
    inference_engine: Any = None,
    write: bool = False,
) -> OrchestratorProbe:
    orchestrator = RLOrchestrator(config)
    timings: dict[str, float] = {}

    started_at = time.perf_counter()
    all_records = load_rl_records(config.data)
    timings["load_records"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    selected_records = orchestrator._select_step_records(
        all_records,
        seed=orchestrator._step_seed(step=step, retry=retry),
    )
    timings["select_records"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    scored_records = orchestrator._generate_and_score(
        selected_records,
        inference_engine=inference_engine,
    )
    timings["generate_score"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    trainable_records = orchestrator._filter_zero_advantage_records(scored_records)
    timings["filter_zero_advantage"] = time.perf_counter() - started_at

    output_path = None
    if write:
        started_at = time.perf_counter()
        output_path = str(orchestrator._write_records(trainable_records, step=step))
        timings["write"] = time.perf_counter() - started_at

    timings["total"] = sum(timings.values())
    rows = [orchestrator._serialize_record(record) for record in trainable_records]
    metrics = rollout_metrics(
        RolloutMetricInputs(
            rows=rows,
            rollouts_per_example=config.orchestrator.rollouts_per_example,
            step=step or 0,
            timings=timings,
        )
    )
    return OrchestratorProbe(
        timings=timings,
        records_available=len(all_records),
        records_selected=len(selected_records),
        records_scored=len(scored_records),
        records_trainable=len(trainable_records),
        rollouts_per_example=config.orchestrator.rollouts_per_example,
        examples_per_step=config.orchestrator.examples_per_step,
        token_batch_size=config.orchestrator.token_batch_size,
        metrics=metrics,
        output_path=output_path,
    )


def with_orchestrator_limits(
    config: RLConfig,
    *,
    examples: int | None,
    rollouts: int | None,
) -> RLConfig:
    updates: dict[str, Any] = {}
    if examples is not None:
        updates["examples_per_step"] = examples
        updates["token_batch_size"] = None
    if rollouts is not None:
        updates["rollouts_per_example"] = rollouts
    if not updates:
        return config
    return config.model_copy(
        update={
            "orchestrator": config.orchestrator.model_copy(update=updates),
        }
    )


def _record_summary(record: RLExample) -> dict[str, Any]:
    return {
        "source": record.source,
        "prompt_turns": len(record.prompt),
        "completion_turns": len(record.completion),
        "reward": record.reward,
        "advantage": record.advantage,
        "has_input_ids": record.input_ids is not None,
        "trainable_tokens": sum(record.loss_mask or []),
        "metadata": record.metadata or {},
    }


# Consolidated from wavelet/orchestrator/preflight.py.
@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] | None = None


def build_preflight_report(config: RLConfig) -> dict[str, Any]:
    """Build cheap launch diagnostics without starting trainer or inference."""
    checks = [
        *_path_checks(config),
        *_launcher_checks(config),
        *_port_checks(config),
        *_schedule_checks(config),
        *_algorithm_checks(config),
        *_attention_backend_checks(config),
        *_low_precision_checks(config),
    ]
    commands: list[dict[str, Any]] = []
    try:
        commands = _resolved_commands(config)
    except ValueError as exc:
        checks.append(
            PreflightCheck(
                name="resolved_commands",
                status="error",
                message=str(exc),
            )
        )
    return {
        "ok": not any(check.status == "error" for check in checks),
        "summary": _summary(config),
        "paths": _paths(config),
        "commands": commands,
        "checks": [asdict(check) for check in checks],
    }


def _summary(config: RLConfig) -> dict[str, Any]:
    return {
        "model": config.model.name,
        "output_dir": str(config.output_dir),
        "launcher_mode": config.launcher.mode,
        "orchestrator_enabled": config.orchestrator.enabled,
        "inference_mode": config.inference.mode,
        "inference_backend": config.inference.vllm.server_backend,
        "trainer_attention": config.model.attn_implementation,
        "policy_transfer": config.policy_transfer.type,
        "algo": config.algo.model_dump(mode="json", exclude_none=True),
        "training_envs": [
            env.model_dump(mode="json", exclude_none=True)
            for env in config.orchestrator.envs
        ],
        "curriculum": (
            None
            if config.orchestrator.curriculum is None
            else config.orchestrator.curriculum.model_dump(
                mode="json",
                exclude_none=True,
            )
        ),
        "target_steps": target_steps(config),
        "low_precision": _low_precision_summary(config),
    }


def _attention_backend_checks(config: RLConfig) -> list[PreflightCheck]:
    attention = config.model.attn_implementation
    if attention != "flash_attention_2":
        return [
            PreflightCheck(
                name="trainer_attention",
                status="ok",
                message=f"Trainer attention implementation is {attention!r}.",
                details={"attn_implementation": attention},
            )
        ]

    available = importlib.util.find_spec("flash_attn") is not None
    return [
        PreflightCheck(
            name="flash_attention_available",
            status="ok" if available else "error",
            message=(
                "FlashAttention 2 is available for the trainer."
                if available
                else "model.attn_implementation='flash_attention_2' requires "
                "flash-attn. Install it with `uv sync --extra flash-attn`."
            ),
            details={"attn_implementation": attention},
        )
    ]


def _low_precision_summary(config: RLConfig) -> dict[str, Any]:
    return {
        "trainer_load_in_4bit": config.model.load_in_4bit,
        "lora_enabled": config.lora is not None,
        "fsdp_enabled": config.fsdp.enabled,
        "launcher_mode": config.launcher.mode,
        "inference_quantization": config.inference.vllm.quantization,
        "inference_load_format": config.inference.vllm.load_format,
    }


def _paths(config: RLConfig) -> dict[str, str]:
    return {
        "output_dir": str(config.output_dir),
        "checkpoint_dir": str(config.checkpoint_output_dir),
        "queue_dir": str(resolve_queue_dir(config.output_dir, config.transport)),
        "policy_dir": str(
            resolve_policy_dir(config.output_dir, config.policy_transfer)
        ),
        "events_dir": str(config.output_dir / "events"),
    }


def _path_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.extend(_data_path_checks(config))
    checks.extend(_adapter_path_checks(config))
    checks.append(_output_dir_check(config.output_dir, clean=config.clean_output_dir))
    checks.append(
        _parent_writable_check(
            config.checkpoint_output_dir,
            name="checkpoint_parent_writable",
        )
    )
    checks.append(
        _parent_writable_check(
            resolve_queue_dir(config.output_dir, config.transport),
            name="queue_parent_writable",
        )
    )
    checks.append(
        _parent_writable_check(
            resolve_policy_dir(config.output_dir, config.policy_transfer),
            name="policy_parent_writable",
        )
    )
    return checks


def _adapter_path_checks(config: RLConfig) -> list[PreflightCheck]:
    adapter_path = config.model.adapter_path
    if adapter_path is None:
        return []

    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_files = [
        filename
        for filename in required_files
        if not (adapter_path / filename).is_file()
    ]
    removed_by_clean = (
        config.clean_output_dir
        and adapter_path.absolute().is_relative_to(config.output_dir.absolute())
    )
    valid = adapter_path.is_dir() and not missing_files and not removed_by_clean
    return [
        PreflightCheck(
            name="model_adapter_path",
            status="ok" if valid else "error",
            message=(
                f"Model adapter is ready: {adapter_path}"
                if valid
                else (
                    "clean_output_dir=true would remove model.adapter_path before "
                    f"launch: {adapter_path}"
                    if removed_by_clean
                    else "Model adapter is not a loadable Wavelet LoRA snapshot: "
                    f"{adapter_path}"
                )
            ),
            details={
                "path": str(adapter_path),
                "missing_files": missing_files,
                "removed_by_clean_output_dir": removed_by_clean,
            },
        )
    ]


def _data_path_checks(config: RLConfig) -> list[PreflightCheck]:
    if config.orchestrator.envs:
        checks: list[PreflightCheck] = []
        for env_index, env in enumerate(config.orchestrator.envs):
            if env.data_path is not None:
                env_paths = [env.data_path]
            elif config.data.source == "local":
                env_paths = _local_data_paths(config)
            else:
                checks.append(
                    PreflightCheck(
                        name=f"data_source_env_{env_index}",
                        status="ok",
                        message=(
                            f"Training environment {env.resolved_name!r} uses "
                            f"data.source={config.data.source!r}, which does not "
                            "require a local path preflight."
                        ),
                        details={"environment": env.resolved_name},
                    )
                )
                continue
            for path_index, path in enumerate(env_paths):
                exists = Path(path).exists()
                checks.append(
                    PreflightCheck(
                        name=f"data_path_env_{env_index}_{path_index}",
                        status="ok" if exists else "error",
                        message=(
                            f"Local data path for {env.resolved_name!r} "
                            f"{'exists' if exists else 'does not exist'}: {path}"
                        ),
                        details={
                            "environment": env.resolved_name,
                            "path": str(path),
                        },
                    )
                )
        return checks
    if config.data.source != "local":
        return [
            PreflightCheck(
                name="data_source",
                status="ok",
                message=f"data.source={config.data.source!r} does not require a local path preflight.",
            )
        ]
    return [
        PreflightCheck(
            f"data_path_{index}",
            "ok" if Path(path).exists() else "error",
            f"Local data path {'exists' if Path(path).exists() else 'does not exist'}: {path}",
            {"path": str(path)},
        )
        for index, path in enumerate(_local_data_paths(config))
    ]


def _local_data_paths(config: RLConfig) -> list[Path]:
    path = config.data.path
    return path if isinstance(path, list) else [path]


def _output_dir_check(output_dir: Path, *, clean: bool) -> PreflightCheck:
    exists = output_dir.exists() and not clean
    return PreflightCheck(
        "output_dir",
        "warning" if exists else "ok",
        (
            (
                "Output directory already exists; use a clean run directory unless "
                "you are intentionally resuming or inspecting existing state."
            )
            if exists
            else f"Output directory is ready to create: {output_dir}"
        ),
        {"path": str(output_dir)},
    )


def _parent_writable_check(path: Path, *, name: str) -> PreflightCheck:
    parent = _existing_path(path)
    writable = os.access(parent, os.W_OK)
    return PreflightCheck(
        name,
        "ok" if writable else "error",
        (
            f"Parent directory is writable: {parent}"
            if writable
            else f"Parent directory is not writable: {parent}"
        ),
        {"path": str(path), "parent": str(parent)},
    )


def _launcher_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Mirrors the runtime guard: every non-integrated mode spawns its own roles.
    invalid = config.launcher.mode != "integrated" and world_size > 1
    checks.append(
        PreflightCheck(
            "torchrun_launcher",
            "error" if invalid else "ok",
            (
                f"Do not run 'wavelet rl' launcher.mode={config.launcher.mode!r} "
                "under torchrun."
                if invalid
                else "Launcher mode is compatible with the current WORLD_SIZE."
            ),
            {"WORLD_SIZE": world_size},
        )
    )

    checks.append(_rollout_reward_mode_check(config))
    checks.extend(_device_group_checks(config))
    return checks


def _rollout_reward_mode_check(config: RLConfig) -> PreflightCheck:
    error = rollout_reward_mode_error(config)
    return PreflightCheck(
        "rollout_reward_mode",
        "ok" if error is None else "error",
        error or "reward.mode is compatible with the rollout source.",
        {"inference_mode": config.inference.mode, "reward_mode": config.reward.mode},
    )


def _trainer_process_count_check(
    config: RLConfig,
    trainer_group: str | None,
) -> PreflightCheck | None:
    if config.launcher.trainer_cuda_visible_devices is None or trainer_group is None:
        return None
    device_count = len([d for d in trainer_group.split(",") if d.strip()])
    processes = config.launcher.trainer_num_processes
    if device_count == processes:
        return None
    return PreflightCheck(
        "trainer_num_processes",
        "error",
        (
            f"launcher.trainer_num_processes={processes} does not match the "
            f"{device_count} pinned trainer device(s) in "
            f"launcher.trainer_cuda_visible_devices={trainer_group!r}."
        ),
        {"cuda_visible_devices": trainer_group, "trainer_num_processes": processes},
    )


def _device_group_checks(config: RLConfig) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    gpu_indices = _available_gpu_indices()
    replicas = config.launcher.inference_num_replicas
    try:
        inference_groups = device_groups(
            config.launcher.inference_cuda_visible_devices,
            replicas,
        )
    except ValueError as exc:
        return [
            PreflightCheck(
                name="inference_cuda_visible_devices",
                status="error",
                message=str(exc),
            )
        ]

    for index, group in enumerate(inference_groups):
        device_count = device_group_size(config, group)
        required = required_inference_devices(config)
        status: CheckStatus = "ok" if device_count >= required else "warning"
        fallback = (
            f"Inference replica {index} has {device_count} visible "
            f"device(s); configured vLLM needs {required}."
        )
        if config.policy_transfer.type == "nccl" and device_count != required:
            # Only TP x DP vLLM workers join the NCCL group; extra visible
            # devices would leave the trainer waiting for ranks that never join.
            status = "error"
            fallback += (
                " NCCL policy transfer requires exactly tensor_parallel_size x "
                "data_parallel_size visible devices per replica."
            )
        checks.append(
            PreflightCheck(
                name=f"inference_devices_{index}",
                status=_device_status(status, group, gpu_indices),
                message=(
                    _device_message(
                        f"Inference replica {index}",
                        group,
                        gpu_indices,
                        fallback=fallback,
                    )
                ),
                details={"cuda_visible_devices": group, "required_devices": required},
            )
        )

    try:
        trainer_group = trainer_device_group(config, strict=False)
    except ValueError as exc:
        checks.append(
            PreflightCheck(
                name="trainer_devices",
                status="error",
                message=str(exc),
            )
        )
        return checks
    if (
        config.launcher.mode in {"colocate", "colocate_sleep"}
        and config.launcher.inference_num_replicas != 1
        and config.launcher.trainer_cuda_visible_devices is None
    ):
        checks.append(
            PreflightCheck(
                name="trainer_devices",
                status="error",
                message=(
                    f"launcher.mode={config.launcher.mode!r} requires "
                    "launcher.trainer_cuda_visible_devices when using multiple "
                    "inference replicas."
                ),
            )
        )
        return checks
    checks.append(
        PreflightCheck(
            name="trainer_devices",
            status=_device_status("ok", trainer_group, gpu_indices),
            message=_device_message(
                "Trainer",
                trainer_group,
                gpu_indices,
                fallback=(
                    "Trainer CUDA_VISIBLE_DEVICES is resolved."
                    if trainer_group is not None
                    else "Trainer CUDA_VISIBLE_DEVICES is not pinned; current environment will be used."
                ),
            ),
            details={"cuda_visible_devices": trainer_group},
        )
    )
    process_count_check = _trainer_process_count_check(config, trainer_group)
    if process_count_check is not None:
        checks.append(process_count_check)
    conflict = device_group_conflict_error(config)
    if conflict is not None:
        checks.append(
            PreflightCheck(
                name="device_group_overlap",
                status="error",
                message=conflict,
            )
        )
    return checks


def _port_checks(config: RLConfig) -> list[PreflightCheck]:
    if config.inference.mode != "vllm_http":
        return [
            PreflightCheck(
                name="inference_ports",
                status="ok",
                message="Inference mode does not start HTTP vLLM servers.",
            )
        ]
    try:
        ports = http_ports(config, config.launcher.inference_num_replicas)
    except ValueError as exc:
        return [
            PreflightCheck(
                name="inference_ports",
                status="error",
                message=str(exc),
            )
        ]
    host = config.inference.http.host
    return [
        (
            PreflightCheck(
                f"inference_port_{port}",
                "ok" if available else "warning",
                (
                    f"HTTP inference port appears available: {port}"
                    if available
                    else f"HTTP inference port appears in use: {port}"
                ),
                {"host": host, "port": port},
            )
        )
        for port in ports
        for available in (_port_available(host, port),)
    ]


def _schedule_checks(config: RLConfig) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            name="target_steps",
            status="ok",
            message=f"Resolved target steps: {target_steps(config)}",
        )
    ]
    if config.orchestrator.examples_per_step is not None:
        examples = config.orchestrator.examples_per_step
        rollouts = config.orchestrator.rollouts_per_example or 1
        if config.orchestrator.envs:
            environments = [
                {
                    "name": env.resolved_name,
                    "ratio": env.ratio,
                    "rollouts_per_group": env.group_size or rollouts,
                    "algorithm": (env.algo or config.algo).type,
                }
                for env in config.orchestrator.envs
            ]
            checks.append(
                PreflightCheck(
                    name="rollout_chunks",
                    status="ok",
                    message=(
                        f"Resolved optimizer batch: {examples} weighted group(s) "
                        f"across {len(environments)} training environment(s) and "
                        f"{chunks_per_step(config)} chunk(s)."
                    ),
                    details={
                        "groups": examples,
                        "chunks": chunks_per_step(config),
                        "environments": environments,
                    },
                )
            )
            return checks
        checks.append(
            PreflightCheck(
                name="rollout_chunks",
                status="ok",
                message=(
                    f"Resolved optimizer batch: {examples} group(s) x {rollouts} "
                    f"rollout(s) = {examples * rollouts} rollout(s) across "
                    f"{chunks_per_step(config)} chunk(s)."
                ),
                details={
                    "groups": examples,
                    "rollouts_per_group": rollouts,
                    "rollouts": examples * rollouts,
                    "chunks": chunks_per_step(config),
                },
            )
        )
    elif config.orchestrator.token_batch_size is not None:
        checks.append(
            PreflightCheck(
                name="rollout_chunks",
                status="ok",
                message=(
                    "Resolved optimizer batch: at least "
                    f"{config.orchestrator.token_batch_size} serialized rollout "
                    "token(s) in one dynamic chunk."
                ),
                details={
                    "token_batch_size": config.orchestrator.token_batch_size,
                    "chunks": chunks_per_step(config),
                },
            )
        )
    return checks


def _algorithm_checks(config: RLConfig) -> list[PreflightCheck]:
    algorithms = [("algorithm", config.algo)]
    algorithms.extend(
        (f"algorithm_env_{index}", env.algo)
        for index, env in enumerate(config.orchestrator.envs)
        if env.algo is not None and env.algo != config.algo
    )
    return [_algorithm_check(name, algorithm) for name, algorithm in algorithms]


def _algorithm_check(
    name: str,
    algorithm: RLAlgorithmConfig,
) -> PreflightCheck:
    if not isinstance(algorithm, CustomAlgorithmConfig):
        return PreflightCheck(
            name,
            "ok",
            f"Built-in algorithm is available: {algorithm.type}",
        )
    try:
        build_algorithm(algorithm)
    except (ImportError, OSError, SyntaxError, TypeError, ValueError) as exc:
        status: CheckStatus = "error"
        message = (
            f"Could not load custom algorithm {algorithm.algorithm!r} "
            f"from {str(algorithm.file)!r}: {type(exc).__name__}: {exc}"
        )
    else:
        status = "ok"
        message = (
            f"Custom algorithm is loadable: {algorithm.algorithm} from {algorithm.file}"
        )
    return PreflightCheck(
        name,
        status,
        message,
        {
            "file": str(algorithm.file),
            "algorithm": algorithm.algorithm,
            "scope": algorithm.scope,
        },
    )


def _low_precision_checks(config: RLConfig) -> list[PreflightCheck]:
    trainer_4bit = config.model.load_in_4bit
    inference_quantized = bool(
        config.inference.vllm.quantization or config.inference.vllm.load_format
    )
    low_precision = trainer_4bit or inference_quantized
    checks: list[PreflightCheck] = [
        PreflightCheck(
            name="low_precision",
            status="ok",
            message=(
                "Resolved low-precision launch settings."
                if low_precision
                else "No low-precision trainer or inference settings enabled."
            ),
            details=_low_precision_summary(config),
        )
    ]
    if trainer_4bit:
        checks.extend(_trainer_4bit_checks(config))
    if inference_quantized and not trainer_4bit:
        checks.append(
            PreflightCheck(
                name="low_precision_inference_mismatch",
                status="warning",
                message=(
                    "Inference is configured with vLLM low-precision loading, but "
                    "trainer model.load_in_4bit is false. Verify the train/serve "
                    "precision mismatch is intentional."
                ),
                details={
                    "quantization": config.inference.vllm.quantization,
                    "load_format": config.inference.vllm.load_format,
                },
            )
        )
    return checks


def _trainer_4bit_checks(config: RLConfig) -> list[PreflightCheck]:
    bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
    lora_enabled = config.lora is not None
    fsdp_disabled = not config.fsdp.enabled
    movable = config.launcher.mode != "colocate_sleep"
    checks = [
        PreflightCheck(name, "ok" if valid else "error", ok if valid else error)
        for name, valid, ok, error in (
            (
                "bitsandbytes_available",
                bitsandbytes_available,
                "bitsandbytes is importable for QLoRA training.",
                "model.load_in_4bit=true requires bitsandbytes to be installed.",
            ),
            (
                "qlora_adapter",
                lora_enabled,
                "QLoRA adapter training is enabled.",
                (
                    "model.load_in_4bit=true requires a LoRA config; Wavelet does "
                    "not support full-model 4-bit training."
                ),
            ),
            (
                "qlora_fsdp",
                fsdp_disabled,
                "FSDP is disabled for QLoRA training.",
                (
                    "QLoRA training uses replicated DDP in Wavelet. Disable "
                    "fsdp.enabled for model.load_in_4bit=true."
                ),
            ),
            (
                "qlora_colocate_sleep",
                movable,
                "Launcher mode is compatible with QLoRA.",
                (
                    "QLoRA does not support colocate_sleep yet because bitsandbytes "
                    "4-bit modules cannot be moved between CPU and GPU."
                ),
            ),
        )
    ]
    if config.fsdp.tp > 1:
        checks.append(
            PreflightCheck(
                "qlora_tensor_parallel",
                "error",
                (
                    "QLoRA with trainer tensor parallelism is not supported. Set "
                    "fsdp.tp=1 for model.load_in_4bit=true."
                ),
                {"fsdp_tp": config.fsdp.tp},
            )
        )
    return checks


def _resolved_commands(config: RLConfig) -> list[dict[str, Any]]:
    if not config.orchestrator.enabled:
        return [
            {
                "role": "trainer",
                "command": "uv run python -m wavelet rl",
                "config": "<provided config>",
                "cuda_visible_devices": trainer_device_group(config, strict=False),
            }
        ]

    if config.launcher.mode in {"process", "colocate", "colocate_sleep"}:
        commands: list[dict[str, Any]] = []
        ports = http_ports(config, config.launcher.inference_num_replicas)
        inference_groups = device_groups(
            config.launcher.inference_cuda_visible_devices,
            len(ports),
        )
        if config.inference.mode == "vllm_http":
            server_command = (
                "inference-server"
                if config.inference.vllm.server_backend == "openai"
                else "native-inference-server"
            )
            for index, (port, devices) in enumerate(
                zip(ports, inference_groups, strict=True)
            ):
                commands.append(
                    {
                        "role": f"inference_server_{index}",
                        "command": f"uv run python -m wavelet {server_command}",
                        "port": port,
                        "cuda_visible_devices": devices,
                    }
                )
        commands.extend(
            [
                {
                    "role": "trainer",
                    "command": "uv run python -m wavelet rl-trainer",
                    "torchrun_nproc_per_node": config.launcher.trainer_num_processes,
                    "cuda_visible_devices": trainer_device_group(config, strict=False),
                },
                {
                    "role": "inference",
                    "command": "uv run python -m wavelet rl-inference",
                    "cuda_visible_devices": None,
                },
            ]
        )
        return commands

    return [
        {
            "role": "integrated",
            "command": "uv run python -m wavelet rl",
            "cuda_visible_devices": trainer_device_group(config, strict=False),
        }
    ]


def _available_gpu_indices() -> set[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    indices = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return indices or None


def _missing_gpu_indices(
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
) -> list[str]:
    if cuda_visible_devices is None or available_indices is None:
        return []
    requested = [device.strip() for device in cuda_visible_devices.split(",")]
    numeric = [device for device in requested if device.isdigit()]
    return [device for device in numeric if device not in available_indices]


def _device_status(
    fallback: CheckStatus,
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
) -> CheckStatus:
    if _missing_gpu_indices(cuda_visible_devices, available_indices):
        return "error"
    return fallback


def _device_message(
    label: str,
    cuda_visible_devices: str | None,
    available_indices: set[str] | None,
    *,
    fallback: str,
) -> str:
    missing = _missing_gpu_indices(cuda_visible_devices, available_indices)
    if missing:
        available = ", ".join(sorted(available_indices or [])) or "unknown"
        return (
            f"{label} requests CUDA device(s) {', '.join(missing)}, but "
            f"nvidia-smi reports available device index(es): {available}."
        )
    return fallback


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())

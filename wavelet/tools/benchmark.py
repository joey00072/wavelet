"""Run short Wavelet jobs and compare their metrics with saved baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

FORMAT_VERSION = 1
METRIC_DIRECTIONS: dict[str, Literal["higher", "lower"]] = {
    "perf/mfu": "higher",
    "perf/model_tokens_per_second": "higher",
    "perf/step_tokens_per_second": "higher",
    "perf/throughput": "higher",
    "perf/tokens_per_second": "higher",
    "perf/peak_memory_gib": "lower",
    "perf/step_seconds": "lower",
}


@dataclass(frozen=True, slots=True)
class Regression:
    metric: str
    baseline: float
    current: float | None
    change_ratio: float | None
    reason: str


def aggregate_metrics(
    metrics_path: Path,
    *,
    warmup_steps: int = 1,
) -> dict[str, dict[str, float]]:
    """Merge monitor rows per step and summarize known performance metrics."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    per_step: dict[int, dict[str, float]] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            destination = per_step.setdefault(step, {})
            for metric in METRIC_DIRECTIONS:
                value = row.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    number = float(value)
                    if math.isfinite(number):
                        destination[metric] = number
    measured_steps = sorted(per_step)[warmup_steps:]
    summaries: dict[str, dict[str, float]] = {}
    for metric in METRIC_DIRECTIONS:
        values = [
            per_step[step][metric]
            for step in measured_steps
            if metric in per_step[step]
        ]
        if not values:
            continue
        summaries[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
            "samples": float(len(values)),
        }
    if not summaries:
        raise ValueError(
            f"No post-warmup performance metrics found in '{metrics_path}'."
        )
    return summaries


def compare_results(
    current: dict[str, object],
    baseline: dict[str, object],
    *,
    threshold: float,
) -> list[Regression]:
    """Return metrics that regress beyond the configured relative threshold."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0, 1).")
    if current.get("benchmark_key") != baseline.get("benchmark_key"):
        raise ValueError(
            "Benchmark identity differs from the baseline; use the same command, "
            "config contents, and hardware."
        )
    current_metrics = _metrics_mapping(current)
    baseline_metrics = _metrics_mapping(baseline)
    regressions: list[Regression] = []
    for metric, baseline_summary in baseline_metrics.items():
        if metric not in METRIC_DIRECTIONS:
            continue
        baseline_mean = _summary_mean(metric, baseline_summary)
        current_summary = current_metrics.get(metric)
        if current_summary is None:
            regressions.append(
                Regression(
                    metric=metric,
                    baseline=baseline_mean,
                    current=None,
                    change_ratio=None,
                    reason="metric missing from current result",
                )
            )
            continue
        current_mean = _summary_mean(metric, current_summary)
        direction = METRIC_DIRECTIONS[metric]
        regressed = (
            current_mean < baseline_mean * (1 - threshold)
            if direction == "higher"
            else current_mean > baseline_mean * (1 + threshold)
        )
        if baseline_mean == 0:
            regressed = direction == "lower" and current_mean > 0
            change_ratio = None
        else:
            change_ratio = (current_mean - baseline_mean) / baseline_mean
        if regressed:
            regressions.append(
                Regression(
                    metric=metric,
                    baseline=baseline_mean,
                    current=current_mean,
                    change_ratio=change_ratio,
                    reason=f"{direction}-is-better threshold exceeded",
                )
            )
    return regressions


def _metrics_mapping(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("Benchmark result has no metrics mapping.")
    return {str(key): value for key, value in metrics.items()}


def _summary_mean(metric: str, summary: object) -> float:
    if not isinstance(summary, dict):
        raise TypeError(f"Benchmark metric '{metric}' is not a summary mapping.")
    value = summary.get("mean")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Benchmark metric '{metric}' has no numeric mean.")
    return float(value)


def build_run_command(
    command: Literal["sft", "rl"],
    config_path: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "wavelet",
        command,
        "@",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--clean-output-dir",
        "false",
        "--monitor.enabled",
        "true",
        "--monitor.write-metrics-jsonl",
        "true",
        "--monitor.wandb.enabled",
        "false",
    ]


def run_benchmark(
    command: Literal["sft", "rl"],
    config_path: Path,
    run_dir: Path,
    output_path: Path,
    *,
    warmup_steps: int = 1,
    timeout_seconds: float = 3600,
) -> dict[str, object]:
    """Execute a clean benchmark run and persist its aggregate result."""
    config_path = config_path.expanduser()
    run_dir = run_dir.expanduser()
    output_path = output_path.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark config not found at '{config_path}'.")
    if run_dir.exists() and not run_dir.is_dir():
        raise FileExistsError(f"Benchmark run path '{run_dir}' is not a directory.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Benchmark run directory '{run_dir}' is not empty; choose a new path."
        )
    if output_path.exists():
        raise FileExistsError(
            f"Benchmark result '{output_path}' already exists; choose a new path."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    invocation = build_run_command(command, config_path, run_dir)
    log_path = run_dir / "benchmark.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"command: {shlex.join(invocation)}\n\n")
            completed = subprocess.run(
                invocation,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Benchmark exceeded {timeout_seconds:g} seconds; see '{log_path}'."
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed with code {completed.returncode}; see "
            f"'{log_path}'."
        )
    metrics = aggregate_metrics(
        run_dir / "metrics.jsonl",
        warmup_steps=warmup_steps,
    )
    identity = _benchmark_identity(command, config_path)
    result: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_key": _benchmark_key(identity),
        "identity": identity,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _benchmark_identity(command: str, config_path: Path) -> dict[str, object]:
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    device_names = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else ["cpu"]
    )
    return {
        "command": command,
        "config_sha256": config_digest,
        "device_names": device_names,
        "torch_version": torch.__version__,
    }


def _benchmark_key(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Benchmark result '{path}' is not a JSON object.")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported benchmark format in '{path}'.")
    return payload


def _print_regressions(regressions: list[Regression]) -> None:
    for regression in regressions:
        change = (
            "missing"
            if regression.change_ratio is None
            else f"{regression.change_ratio:+.1%}"
        )
        print(
            f"REGRESSION {regression.metric}: baseline={regression.baseline:g} "
            f"current={regression.current!s} change={change}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="run and aggregate a benchmark")
    run_parser.add_argument("command", choices=("sft", "rl"))
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("run_dir", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--warmup-steps", type=int, default=1)
    run_parser.add_argument("--timeout-seconds", type=float, default=3600)
    run_parser.add_argument("--baseline", type=Path)
    run_parser.add_argument("--regression-threshold", type=float, default=0.05)
    compare_parser = subparsers.add_parser("compare", help="compare two results")
    compare_parser.add_argument("current", type=Path)
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("--regression-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    if args.action == "run":
        current = run_benchmark(
            args.command,
            args.config,
            args.run_dir,
            args.output,
            warmup_steps=args.warmup_steps,
            timeout_seconds=args.timeout_seconds,
        )
        if args.baseline is None:
            print(f"Wrote benchmark result to {args.output}")
            return 0
        baseline = _load_result(args.baseline)
    else:
        current = _load_result(args.current)
        baseline = _load_result(args.baseline)
    regressions = compare_results(
        current,
        baseline,
        threshold=args.regression_threshold,
    )
    if regressions:
        _print_regressions(regressions)
        return 2
    print("No benchmark regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.diagnostics import (
    continuous_batch_probe,
    http_health,
    inference_debug_state,
    make_probe_examples,
    probe_engine,
)
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
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
    args, config_args = parser.parse_known_args(argv)

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.stagger_ms < 0:
        raise SystemExit("--stagger-ms must be >= 0")
    if args.max_completion_tokens < 1:
        raise SystemExit("--max-completion-tokens must be >= 1")
    if args.data_parallel_size is not None and args.data_parallel_size < 1:
        raise SystemExit("--data-parallel-size must be >= 1")

    config = load_config(RLConfig, config_args)
    if args.action == "inspect":
        _print_report(inference_debug_state(config), json_output=args.json_output)
        return 0
    if args.action == "health":
        _print_report({"health": http_health(config)}, json_output=args.json_output)
        return 0
    if args.action == "continuous-batch":
        requests, metrics = continuous_batch_probe(
            config,
            count=args.count,
            concurrency=args.concurrency,
            prompt=args.prompt,
            max_completion_tokens=args.max_completion_tokens,
            stagger_seconds=args.stagger_ms / 1000.0,
            data_parallel_size=args.data_parallel_size,
        )
        report = {
            "metrics": metrics.to_dict(),
            "sample": [
                request.to_dict() for request in requests[: min(5, len(requests))]
            ],
            "errors": [
                request.to_dict() for request in requests if request.error is not None
            ][:5],
        }
        _print_report(report, json_output=args.json_output)
        return 0

    records = make_probe_examples(count=args.count, prompt=args.prompt)
    engine = create_policy_inference_engine(config)
    try:
        engine.setup()
        if args.policy_dir is not None:
            from pathlib import Path

            engine.load_policy(Path(args.policy_dir), step=args.policy_step)
        annotated, metrics = probe_engine(
            engine,
            records,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    finally:
        engine.close()
    report = {
        "metrics": metrics.to_dict(),
        "sample": _sample_records(annotated, limit=min(3, len(annotated))),
    }
    _print_report(report, json_output=args.json_output)
    return 0


def _sample_records(records: list[Any], *, limit: int) -> list[dict[str, Any]]:
    sample = []
    for record in records[:limit]:
        completion_text = ""
        if record.completion:
            completion_text = record.completion[0].get("content", "")
        sample.append(
            {
                "completion": completion_text,
                "trainable_tokens": sum(record.loss_mask or []),
                "has_inference_logprobs": record.inference_logprobs is not None,
            }
        )
    return sample


def _print_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for child_key, child_value in value.items():
                print(f"  {child_key}: {child_value}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    sys.exit(main())

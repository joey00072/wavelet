from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
    from wavelet.configs.rl_config import RLConfig
    from wavelet.orchestrator.preflight import build_preflight_report
    from wavelet.utils.config import load_config

    parser = argparse.ArgumentParser(
        prog="wavelet debug preflight",
        description="Validate cheap RL launch prerequisites without starting workers.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    report = build_preflight_report(load_config(RLConfig, config_args))
    _print_preflight_report(report, json_output=args.json_output)
    return 0 if report["ok"] else 1


def _inference_main(argv: list[str]) -> int:
    from wavelet.configs.rl_config import RLConfig
    from wavelet.inference.diagnostics import http_health, inference_debug_state
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


def _validate_inference_args(args: argparse.Namespace) -> None:
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


def _run_continuous_batch_probe(config: Any, args: argparse.Namespace) -> int:
    from wavelet.inference.diagnostics import continuous_batch_probe

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
            "sample": [
                request.to_dict() for request in requests[: min(5, len(requests))]
            ],
            "errors": [
                request.to_dict() for request in requests if request.error is not None
            ][:5],
        },
        json_output=args.json_output,
    )
    return 0


def _run_inference_engine_probe(config: Any, args: argparse.Namespace) -> int:
    from wavelet.inference.diagnostics import make_probe_examples, probe_engine
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
            "sample": _sample_records(annotated, limit=min(3, len(annotated))),
        },
        json_output=args.json_output,
    )
    return 0


def _orchestrator_main(argv: list[str]) -> int:
    from wavelet.configs.rl_config import RLConfig
    from wavelet.inference.policy import create_policy_inference_engine
    from wavelet.orchestrator.diagnostics import (
        orchestrator_debug_state,
        probe_orchestrator,
        sample_orchestrator_records,
        with_orchestrator_limits,
    )
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
    from wavelet.configs.rl_config import RLConfig
    from wavelet.trainer.diagnostics import (
        build_runtime_parity_report,
        export_rollout_token_debug,
        inspect_rollout_batch,
    )
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
    _print_trainer_report(report, json_output=args.json_output)
    return _trainer_exit_code(report, action=args.action)


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


def _print_preflight_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

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


def _print_trainer_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"path: {report['path']}")
    _print_present_fields(report, ("ok", "passed", "skipped"))
    if "summary" in report:
        for key, value in report["summary"].items():
            print(f"{key}: {value}")
    _print_present_fields(report, ("rows_exported", "write_path"))
    _print_present_fields(
        report,
        ("token_count", "max_abs_diff", "mean_abs_diff", "skip_reason"),
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


if __name__ == "__main__":
    sys.exit(main())

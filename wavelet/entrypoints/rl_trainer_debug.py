from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.diagnostics import (
    build_runtime_parity_report,
    export_rollout_token_debug,
    inspect_rollout_batch,
)
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
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
    _print_report(report, json_output=args.json_output)
    return _exit_code(report, action=args.action)


def _print_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"path: {report['path']}")
    for key in ("ok", "passed", "skipped"):
        if key in report:
            print(f"{key}: {report[key]}")
    if "summary" in report:
        for key, value in report["summary"].items():
            print(f"{key}: {value}")
    for key in ("rows_exported", "write_path"):
        if key in report:
            print(f"{key}: {report[key]}")
    for key in ("token_count", "max_abs_diff", "mean_abs_diff", "skip_reason"):
        if key in report:
            print(f"{key}: {report[key]}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  row {error.get('row')}: {error.get('field')}: {error.get('message')}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(
                f"  row {warning.get('row')}: "
                f"{warning.get('field')}: {warning.get('message')}"
            )


def _exit_code(report: dict[str, Any], *, action: str) -> int:
    if action in {"inspect", "tokens"}:
        return 0 if report["ok"] else 1
    if report["errors"]:
        return 1
    if report["skipped"]:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

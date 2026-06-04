from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.trainer.diagnostics import inspect_rollout_batch
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="wavelet debug trainer",
        description="Inspect trainer inputs without launching training.",
    )
    parser.add_argument("action", choices=["inspect"], help="Diagnostic action to run.")
    parser.add_argument("--rollout-path", type=Path, default=None)
    parser.add_argument("--queue-step", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1")
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit must be >= 0")

    config = load_config(RLConfig, config_args)
    report = inspect_rollout_batch(
        config,
        rollout_path=args.rollout_path,
        queue_step=args.queue_step,
        max_rows=args.max_rows,
        sample_limit=args.sample_limit,
    )
    _print_report(report, json_output=args.json_output)
    return 0 if report["ok"] else 1


def _print_report(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"path: {report['path']}")
    print(f"ok: {report['ok']}")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
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


if __name__ == "__main__":
    sys.exit(main())

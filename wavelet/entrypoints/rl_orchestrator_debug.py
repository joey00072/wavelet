from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.inference.policy import create_policy_inference_engine
from wavelet.orchestrator.diagnostics import (
    orchestrator_debug_state,
    probe_orchestrator,
    sample_orchestrator_records,
    with_orchestrator_limits,
)
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="wavelet orchestrator-debug",
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

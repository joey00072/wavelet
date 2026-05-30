from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.preflight import build_preflight_report
from wavelet.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="wavelet debug preflight",
        description="Validate cheap RL launch prerequisites without starting workers.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args, config_args = parser.parse_known_args(argv)

    config = load_config(RLConfig, config_args)
    report = build_preflight_report(config)
    _print_report(report, json_output=args.json_output)
    return 0 if report["ok"] else 1


def _print_report(report: dict[str, Any], *, json_output: bool) -> None:
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


if __name__ == "__main__":
    sys.exit(main())

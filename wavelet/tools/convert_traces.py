"""Convert Wavelet trace JSONL files to a Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

TRACE_FIELDS = (
    "format_version",
    "timestamp",
    "subsystem",
    "event",
    "step",
    "queue_step",
    "optimizer_step",
    "policy_step",
    "task",
    "harness",
    "rollout_id",
)
REQUIRED_TRACE_FIELDS = ("format_version", "timestamp", "subsystem", "event")


def _trace_files(inputs: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            search_root = path / "traces" if (path / "traces").is_dir() else path
            files.update(
                candidate
                for candidate in search_root.rglob("*.jsonl")
                if candidate.is_file()
            )
        else:
            raise FileNotFoundError(f"Trace input not found: '{path}'.")
    return sorted(files)


def load_trace_rows(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Read trace events in deterministic file and line order."""
    rows: list[dict[str, Any]] = []
    for path in _trace_files(inputs):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in '{path}' at line {line_number}: {exc.msg}."
                    ) from exc
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"Trace row in '{path}' at line {line_number} must be an object."
                    )
                missing = [
                    name for name in REQUIRED_TRACE_FIELDS if name not in payload
                ]
                if missing:
                    names = ", ".join(missing)
                    raise ValueError(
                        f"Trace row in '{path}' at line {line_number} is missing: {names}."
                    )
                rows.append(
                    {
                        **{name: payload.get(name) for name in TRACE_FIELDS},
                        "details": json.dumps(
                            payload.get("details"),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "source_file": str(path),
                        "source_line": line_number,
                    }
                )
    return rows


def _register_dataset_file(
    root: Path,
    *,
    subset: str,
    split: str,
    relative_path: str,
) -> None:
    readme = root / "README.md"
    metadata: dict[str, Any] = {}
    body = "# Wavelet Trace Dataset\n"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        end = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if lines and lines[0].strip() == "---" and end is not None:
            metadata = yaml.safe_load("".join(lines[1:end])) or {}
            body = "".join(lines[end + 1 :])
        else:
            body = text

    configs = metadata.setdefault("configs", [])
    config = next(
        (item for item in configs if item.get("config_name") == subset),
        None,
    )
    if config is None:
        config = {"config_name": subset, "data_files": []}
        configs.append(config)
    data_files = config.setdefault("data_files", [])
    entry = next((item for item in data_files if item.get("split") == split), None)
    if entry is None:
        data_files.append({"split": split, "path": relative_path})
    else:
        entry["path"] = relative_path

    rendered = yaml.safe_dump(metadata, sort_keys=False)
    content = f"---\n{rendered}---\n{body}"
    readme.write_text(
        content if content.endswith("\n") else content + "\n",
        encoding="utf-8",
    )


def convert_traces(
    inputs: Sequence[Path],
    *,
    output_dir: Path | None = None,
    repo_id: str | None = None,
    subset: str = "default",
    split: str = "train",
    public: bool = False,
) -> Path | str:
    """Write trace rows locally as parquet or push them to the Hub."""
    if (output_dir is None) == (repo_id is None):
        raise ValueError("Choose exactly one of output_dir or repo_id.")
    if public and repo_id is None:
        raise ValueError("public=true is only valid when pushing to repo_id.")
    rows = load_trace_rows(inputs)
    if not rows:
        raise ValueError("No trace events found in the requested inputs.")
    dataset = Dataset.from_list(rows)
    if repo_id is not None:
        dataset.push_to_hub(
            repo_id,
            config_name=subset,
            split=split,
            private=not public,
        )
        return repo_id

    assert output_dir is not None
    root = output_dir.expanduser()
    relative_path = f"{subset}/{split}.parquet"
    output_path = root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path.as_posix())
    _register_dataset_file(
        root,
        subset=subset,
        split=split,
        relative_path=relative_path,
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Trace JSONL files or directories"
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path, help="Local dataset directory")
    destination.add_argument("--repo-id", help="Hugging Face dataset repository id")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--public", action="store_true", help="Create a public Hub dataset"
    )
    args = parser.parse_args(argv)
    try:
        result = convert_traces(
            args.inputs,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            subset=args.subset,
            split=args.split,
            public=args.public,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Converted Wavelet traces to {result}.")
    return 0

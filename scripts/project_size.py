from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
    "ref",
    "wandb",
}

COUNTED_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}

CATEGORY_PREFIXES = (
    ("wavelet/trainer/", "trainer"),
    ("wavelet/inference/", "inference"),
    ("wavelet/orchestrator/", "orchestrator"),
    ("wavelet/data/", "data"),
    ("wavelet/configs/", "configs"),
    ("wavelet/entrypoints/", "entrypoints"),
    ("wavelet/kernels/", "kernels"),
    ("wavelet/utils/", "utils"),
    ("tests/", "tests"),
    ("examples/", "examples"),
    ("docs/", "docs"),
    ("webui/", "webui"),
    ("scripts/", "tooling"),
)

CORE_CATEGORIES = {
    "configs",
    "data",
    "entrypoints",
    "inference",
    "kernels",
    "orchestrator",
    "trainer",
    "utils",
    "wavelet_core",
}

PY_COMPLEXITY_NODES = (
    ast.Assert,
    ast.BoolOp,
    ast.ExceptHandler,
    ast.For,
    ast.If,
    ast.IfExp,
    ast.Match,
    ast.Try,
    ast.While,
    ast.With,
    ast.comprehension,
)


_ACCUMULATED_FIELDS = (
    "lines",
    "source_lines",
    "python_functions",
    "python_classes",
    "python_complexity_points",
)


@dataclass(frozen=True)
class FileMetrics:
    path: str
    suffix: str
    lines: int
    source_lines: int
    blank_lines: int
    comment_lines: int
    python_functions: int = 0
    python_classes: int = 0
    python_complexity_points: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count Wavelet files, lines, and rough Python complexity."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", type=Path, help="Append one JSON snapshot line.")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override snapshot timestamp. Defaults to current UTC time.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of largest files to include in the snapshot.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = list_project_files(root)
    metrics = [measure_file(root, path) for path in files]
    snapshot = summarize(metrics, timestamp=args.timestamp, top=args.top)
    text = json.dumps(snapshot, sort_keys=True)
    print(text)
    if args.write is not None:
        write_jsonl(root, args.write, text)
    return 0


def list_project_files(root: Path) -> list[Path]:
    git_files = _git_files(root)
    if git_files is None:
        paths = (path for path in root.rglob("*") if path.is_file())
    else:
        paths = (root / path for path in git_files)
    return sorted(
        path
        for path in paths
        if path.is_file()
        if path.suffix in COUNTED_SUFFIXES and not _ignored(root, path)
    )


def measure_file(root: Path, path: Path) -> FileMetrics:
    relative = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    blank_lines = 0
    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif _is_comment_line(path.suffix, stripped):
            comment_lines += 1
    base = {
        "path": relative,
        "suffix": path.suffix,
        "lines": len(lines),
        "source_lines": len(lines) - blank_lines - comment_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
    }
    if path.suffix != ".py":
        return FileMetrics(**base)
    return _measure_python_file(text, base)


def summarize(
    metrics: Iterable[FileMetrics],
    *,
    timestamp: str | None,
    top: int,
) -> dict[str, object]:
    files = list(metrics)
    by_extension: Counter[str] = Counter()
    by_top_level: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    core: Counter[str] = Counter()
    for metric in files:
        by_extension[metric.suffix or "<none>"] += metric.lines
        by_top_level[metric.path.split("/", 1)[0]] += metric.lines
        category = _category_for_path(metric.path)
        _accumulate(by_category.setdefault(category, Counter()), metric)
        if category in CORE_CATEGORIES:
            _accumulate(core, metric)
    return {
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        "files": len(files),
        "lines": sum(metric.lines for metric in files),
        "source_lines": sum(metric.source_lines for metric in files),
        "blank_lines": sum(metric.blank_lines for metric in files),
        "comment_lines": sum(metric.comment_lines for metric in files),
        "python_files": sum(1 for metric in files if metric.suffix == ".py"),
        "python_functions": sum(metric.python_functions for metric in files),
        "python_classes": sum(metric.python_classes for metric in files),
        "python_complexity_points": sum(
            metric.python_complexity_points for metric in files
        ),
        "categories": {
            category: dict(values) for category, values in sorted(by_category.items())
        },
        "core": dict(core),
        "lines_by_extension": dict(sorted(by_extension.items())),
        "lines_by_top_level": dict(sorted(by_top_level.items())),
        "largest_files": [
            {
                "path": metric.path,
                "lines": metric.lines,
                "source_lines": metric.source_lines,
            }
            for metric in sorted(files, key=lambda item: item.lines, reverse=True)[:top]
        ],
    }


def _accumulate(totals: Counter[str], metric: FileMetrics) -> None:
    totals["files"] += 1
    for name in _ACCUMULATED_FIELDS:
        totals[name] += getattr(metric, name)


def _category_for_path(path: str) -> str:
    for prefix, category in CATEGORY_PREFIXES:
        if path.startswith(prefix):
            return category
    if path in {"AGENTS.md", "README.md", "pyproject.toml"}:
        return "project_docs"
    if path.startswith("wavelet/"):
        return "wavelet_core"
    return "other"


def write_jsonl(root: Path, path: Path, text: str) -> None:
    target = path if path.is_absolute() else root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _git_files(root: Path) -> list[Path] | None:
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [Path(line) for line in result.stdout.splitlines() if line]


def _ignored(root: Path, path: Path) -> bool:
    parts = set(path.relative_to(root).parts)
    return bool(parts & IGNORED_PARTS)


def _is_comment_line(suffix: str, stripped: str) -> bool:
    if suffix == ".py":
        return stripped.startswith("#")
    if suffix in {".js", ".ts", ".tsx", ".css"}:
        return stripped.startswith(("//", "/*"))
    if suffix in {".md", ".toml", ".yaml", ".yml"}:
        return stripped.startswith("#")
    return False


def _measure_python_file(text: str, base: dict[str, object]) -> FileMetrics:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return FileMetrics(**base)
    return FileMetrics(
        **base,
        python_functions=sum(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            for node in ast.walk(tree)
        ),
        python_classes=sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
        python_complexity_points=sum(
            isinstance(node, PY_COMPLEXITY_NODES) for node in ast.walk(tree)
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

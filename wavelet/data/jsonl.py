from __future__ import annotations

from pathlib import Path


def count_nonempty_jsonl_rows(
    path: Path,
    *,
    description: str = "JSONL file",
) -> int:
    rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    if rows == 0:
        raise ValueError(f"{description} '{path}' contains no rows.")
    return rows

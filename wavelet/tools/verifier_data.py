"""Turn Verifiers environment datasets into Wavelet RL data rows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any


def import_verifiers() -> ModuleType:
    """Import ``verifiers`` or exit with the install hint."""
    try:
        import verifiers
    except ImportError as exc:
        raise SystemExit(
            "This example uses Verifiers environments. Install them with "
            "`uv sync --extra verifiers --extra envs`."
        ) from exc
    return verifiers


def verifier_rows(
    dataset: Iterable[Any],
    *,
    limit: int | None = None,
    id_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Wrap verifier examples as prompt-only RL rows with the example as metadata.

    ``example_id`` defaults to the first present key in ``id_keys``, then the
    dataset index.
    """
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        payload = dict(example)
        fallback = next((payload[key] for key in id_keys if key in payload), index)
        payload.setdefault("example_id", fallback)
        rows.append(
            {
                "prompt": payload["prompt"],
                "completion": "",
                "metadata": {
                    "verifier_example": payload,
                    "example_id": payload["example_id"],
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSON lines, creating parent directories; return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count

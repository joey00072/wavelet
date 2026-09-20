"""Read bounded live episode snapshots without changing run artifacts."""

from __future__ import annotations

import itertools
import json
import os
import stat
from pathlib import Path
from typing import Any
from uuid import UUID

from wavelet.monitor import redact
from wavelet.orchestrator.live import episode_abandoned

MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_SCAN = 2048


def _root(output_dir: Path) -> Path | None:
    run = output_dir.resolve()
    root = (run / "traces" / "live").resolve()
    return root if root.is_relative_to(run) else None


def _read(root: Path, path: Path, maximum: int) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or path.resolve().parent != root:
            return None
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            raw = handle.read(maximum + 1)
        if len(raw) > maximum:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "running" and episode_abandoned(payload):
            payload["status"] = "abandoned"
        return redact(payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None


def read_episode(output_dir: Path, episode_id: str) -> dict[str, Any] | None:
    """Read a UUID-addressed snapshot confined to its run directory."""
    try:
        canonical_id = str(UUID(episode_id))
    except ValueError:
        return None
    root = _root(output_dir)
    if root is None or canonical_id != episode_id:
        return None
    payload = _read(root, root / f"{canonical_id}.json", MAX_SNAPSHOT_BYTES)
    if payload is None or payload.get("id") != canonical_id:
        return None
    return payload


def list_episodes(output_dir: Path, *, limit: int = 100) -> dict[str, Any]:
    root = _root(output_dir)
    if root is None or not root.is_dir():
        return {"episodes": [], "total": 0, "truncated": False}
    rows = []
    with os.scandir(root) as entries:
        paths = list(
            itertools.islice(
                (
                    Path(entry.path)
                    for entry in entries
                    if entry.name.endswith(".meta.json")
                ),
                MAX_SCAN + 1,
            )
        )
    for path in paths[:MAX_SCAN]:
        payload = _read(root, path, MAX_METADATA_BYTES)
        if payload is None or path.name != f"{payload.get('id')}.meta.json":
            continue
        rows.append(payload)
    rows.sort(
        key=lambda row: (
            row.get("status") == "running",
            str(row.get("updated_at", "")),
        ),
        reverse=True,
    )
    return {
        "episodes": rows[:limit],
        "total": len(rows),
        "truncated": len(paths) > MAX_SCAN,
    }

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _CachedFile:
    size: int = -1
    mtime_ns: int = -1
    offset: int = 0
    rows: deque[dict[str, Any]] = field(default_factory=deque)
    parse_errors: int = 0
    dropped: int = 0


class JsonlCache:
    """Incrementally parsed JSONL files keyed by path.

    Event and sample files are append-only, so only the appended bytes are read
    on refresh. A file that shrank or was rewritten is parsed from the start.
    ``max_rows`` bounds memory: only the newest rows are retained and
    ``dropped`` counts the rest, so a long run cannot grow the server without
    limit. Wide numeric metric files use :class:`~wavelet.dashboard.metrics.MetricStore`.
    """

    def __init__(self, *, max_rows: int | None = None) -> None:
        self._files: dict[Path, _CachedFile] = {}
        self._lock = threading.Lock()
        self._max_rows = max_rows

    def rows(self, path: Path) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._refresh(path).rows)

    def parse_errors(self, path: Path) -> int:
        with self._lock:
            return self._refresh(path).parse_errors

    def dropped(self, path: Path) -> int:
        with self._lock:
            return self._refresh(path).dropped

    def _new_file(self) -> _CachedFile:
        return _CachedFile(rows=deque(maxlen=self._max_rows))

    def _refresh(self, path: Path) -> _CachedFile:
        cached = self._files.get(path)
        if cached is None:
            cached = self._files[path] = self._new_file()
        try:
            stat = path.stat()
        except OSError:
            self._files[path] = self._new_file()
            return self._files[path]
        if stat.st_size == cached.size and stat.st_mtime_ns == cached.mtime_ns:
            return cached
        if stat.st_size < cached.offset:
            cached = self._new_file()
            self._files[path] = cached
        with path.open("rb") as handle:
            handle.seek(cached.offset)
            data = handle.read()
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            cached.size = stat.st_size
            cached.mtime_ns = stat.st_mtime_ns
            return cached
        complete = data[: last_newline + 1]
        for line in complete.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                cached.parse_errors += 1
                continue
            if isinstance(row, dict):
                if (
                    cached.rows.maxlen is not None
                    and len(cached.rows) == cached.rows.maxlen
                ):
                    cached.dropped += 1
                cached.rows.append(row)
        cached.offset += len(complete)
        cached.size = stat.st_size
        cached.mtime_ns = stat.st_mtime_ns
        return cached


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns

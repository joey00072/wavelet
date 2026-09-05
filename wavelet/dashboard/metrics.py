"""Columnar, incrementally loaded metric tables for the dashboard.

Metric JSONL files are append-only and wide (hundreds of numeric keys). Keeping
every row as a ``dict`` costs kilobytes per step; this store keeps one
``array('d')`` per key (8 bytes per value, NaN for missing) plus the step and
timestamp columns, reads only appended bytes on refresh, and answers windowed
or downsampled series queries without materializing rows.
"""

from __future__ import annotations

import json
import math
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SERIES_POINTS = 100_000
_NAN = float("nan")


@dataclass
class MetricTable:
    size: int = -1
    mtime_ns: int = -1
    offset: int = 0
    parse_errors: int = 0
    steps: list[float | None] = field(default_factory=list)
    timestamps: list[str | None] = field(default_factory=list)
    columns: dict[str, array] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    last_row: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, row: dict[str, Any]) -> None:
        index = len(self.steps)
        step = row.get("step")
        self.steps.append(
            float(step)
            if isinstance(step, int | float) and math.isfinite(step)
            else None
        )
        stamp = row.get("timestamp")
        self.timestamps.append(stamp if isinstance(stamp, str) else None)
        for key, value in row.items():
            if key in {"step", "timestamp"} or isinstance(value, bool):
                continue
            if not isinstance(value, int | float) or not math.isfinite(value):
                continue
            column = self.columns.get(key)
            if column is None:
                column = array("d", [_NAN]) * index
                self.columns[key] = column
            column.append(float(value))
            self.counts[key] = self.counts.get(key, 0) + 1
        for column in self.columns.values():
            if len(column) <= index:
                column.append(_NAN)
        self.last_row = row

    def keys(self) -> list[dict[str, Any]]:
        return [
            {"key": key, "count": count} for key, count in sorted(self.counts.items())
        ]

    def value(self, key: str, index: int) -> float | None:
        column = self.columns.get(key)
        if column is None or index >= len(column):
            return None
        value = column[index]
        return None if math.isnan(value) else value

    def row(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step": _int_if_whole(self.steps[index]),
            "timestamp": self.timestamps[index],
        }
        for key, column in self.columns.items():
            value = column[index]
            if not math.isnan(value):
                row[key] = _int_if_whole(value)
        return row

    def indices(
        self,
        *,
        sort_by_step: bool = False,
        start: float | None = None,
        end: float | None = None,
        after: float | None = None,
        limit: int = 0,
    ) -> list[int]:
        indices = list(range(len(self.steps)))
        if sort_by_step:
            indices.sort(
                key=lambda i: self.steps[i] if self.steps[i] is not None else 0
            )
        if start is not None or end is not None or after is not None:
            kept: list[int] = []
            for i in indices:
                step = self.steps[i]
                if step is None:
                    continue
                if start is not None and step < start:
                    continue
                if end is not None and step > end:
                    continue
                if after is not None and step <= after:
                    continue
                kept.append(i)
            indices = kept
        if limit > 0:
            indices = indices[-limit:]
        return indices

    def series(
        self,
        keys: list[str],
        indices: list[int],
        *,
        points: int = 0,
    ) -> dict[str, Any]:
        bucket = 1
        if points > 0 and len(indices) > points:
            bucket = math.ceil(len(indices) / points)
        if bucket == 1:
            return {
                "steps": [_int_if_whole(self.steps[i]) for i in indices],
                "timestamps": [self.timestamps[i] for i in indices],
                "series": {key: [self.value(key, i) for i in indices] for key in keys},
                "rows": len(indices),
                "downsampled": False,
                "bucket": 1,
            }
        steps: list[float | int | None] = []
        timestamps: list[str | None] = []
        series: dict[str, list[float | None]] = {key: [] for key in keys}
        envelope: dict[str, dict[str, list[float | None]]] = {
            key: {"min": [], "max": []} for key in keys
        }
        for start in range(0, len(indices), bucket):
            chunk = indices[start : start + bucket]
            last = chunk[-1]
            steps.append(_int_if_whole(self.steps[last]))
            timestamps.append(self.timestamps[last])
            for key in keys:
                values = [v for i in chunk if (v := self.value(key, i)) is not None]
                if values:
                    series[key].append(sum(values) / len(values))
                    envelope[key]["min"].append(min(values))
                    envelope[key]["max"].append(max(values))
                else:
                    series[key].append(None)
                    envelope[key]["min"].append(None)
                    envelope[key]["max"].append(None)
        return {
            "steps": steps,
            "timestamps": timestamps,
            "series": series,
            "envelope": envelope,
            "rows": len(indices),
            "downsampled": True,
            "bucket": bucket,
        }


class MetricStore:
    """Incrementally refreshed :class:`MetricTable` per file path."""

    def __init__(self) -> None:
        self._tables: dict[Path, MetricTable] = {}
        self._lock = threading.Lock()

    def table(self, path: Path) -> MetricTable:
        with self._lock:
            return self._refresh(path)

    def _refresh(self, path: Path) -> MetricTable:
        table = self._tables.setdefault(path, MetricTable())
        try:
            stat = path.stat()
        except OSError:
            self._tables[path] = MetricTable()
            return self._tables[path]
        if stat.st_size == table.size and stat.st_mtime_ns == table.mtime_ns:
            return table
        if stat.st_size < table.offset:
            table = MetricTable()
            self._tables[path] = table
        with path.open("rb") as handle:
            handle.seek(table.offset)
            data = handle.read()
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            table.size = stat.st_size
            table.mtime_ns = stat.st_mtime_ns
            return table
        complete = data[: last_newline + 1]
        for line in complete.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                table.parse_errors += 1
                continue
            if isinstance(row, dict):
                table.append(row)
        table.offset += len(complete)
        table.size = stat.st_size
        table.mtime_ns = stat.st_mtime_ns
        return table


def _int_if_whole(value: float | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value

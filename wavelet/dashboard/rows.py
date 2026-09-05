"""Compact, sortable views over rollout and evaluation JSONL rows."""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wavelet.dashboard.jsonl import file_signature
from wavelet.monitor import _iter_jsonl_dicts, _message_text, _numeric

ROLLOUT_SORT_KEYS = frozenset(
    {
        "row_index",
        "reward",
        "advantage",
        "completion_token_count",
        "input_token_count",
        "logprob_mean",
        "logprob_min",
        "group_key",
        "env",
        "example_id",
        "policy_step",
        "is_truncated",
        "turn_count",
    }
)
EVAL_SORT_KEYS = frozenset(
    {
        "row_index",
        "reward",
        "example_id",
        "completion_token_count",
        "is_truncated",
        "has_error",
    }
)
MAX_TEXT_CHARS = 3000
MAX_SCAN_ROWS = 50_000


@dataclass(frozen=True, slots=True)
class RowFilters:
    env: str | None = None
    group_key: str | None = None
    example_id: str | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    truncated: bool | None = None
    stop_condition: str | None = None
    advantage_sign: str | None = None
    has_error: bool | None = None
    search: str | None = None

    def matches(self, row: dict[str, Any]) -> bool:
        if self.env is not None and row.get("env") != self.env:
            return False
        if self.group_key is not None and row.get("group_key") != self.group_key:
            return False
        if self.example_id is not None and row.get("example_id") != self.example_id:
            return False
        reward = row.get("reward")
        if self.min_reward is not None and (reward is None or reward < self.min_reward):
            return False
        if self.max_reward is not None and (reward is None or reward > self.max_reward):
            return False
        if (
            self.truncated is not None
            and bool(row.get("is_truncated")) != self.truncated
        ):
            return False
        if (
            self.stop_condition is not None
            and row.get("stop_condition") != self.stop_condition
        ):
            return False
        if self.has_error is not None and bool(row.get("error")) != self.has_error:
            return False
        if self.advantage_sign is not None and not _advantage_matches(
            row.get("advantage"), self.advantage_sign
        ):
            return False
        if self.search:
            needle = self.search.lower()
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("prompt", "completion", "example_id", "group_key")
            ).lower()
            if needle not in haystack:
                return False
        return True


def _advantage_matches(value: float | None, sign: str) -> bool:
    if value is None:
        return False
    if sign == "positive":
        return value > 0
    if sign == "negative":
        return value < 0
    if sign == "zero":
        return value == 0
    if sign == "nonzero":
        return value != 0
    return True


def compact_rollout_row(
    row: dict[str, Any], *, row_index: int, max_text_chars: int = MAX_TEXT_CHARS
) -> dict[str, Any]:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    task = metadata.get("task") if isinstance(metadata.get("task"), dict) else {}
    logprobs = row.get("inference_logprobs")
    logprob_values = (
        [float(v) for v in logprobs if isinstance(v, int | float) and math.isfinite(v)]
        if isinstance(logprobs, list)
        else []
    )
    loss_mask = row.get("loss_mask")
    input_ids = row.get("input_ids")
    env = row.get("env_name") or row.get("source") or task.get("name")
    example_id = row.get("example_id", task.get("example_id"))
    rollout = (
        metadata.get("rollout") if isinstance(metadata.get("rollout"), dict) else {}
    )
    compact = {
        "row_index": row_index,
        "reward": _numeric(row, "reward"),
        "advantage": _numeric(row, "advantage"),
        "env": str(env) if env is not None else None,
        "example_id": str(example_id) if example_id is not None else None,
        "group_key": _optional_str(metadata.get("group_key")),
        "rollout_key": _optional_str(metadata.get("rollout_key")),
        "policy_step": metadata.get("policy_step"),
        "stop_condition": _optional_str(metadata.get("stop_condition")),
        "is_truncated": bool(metadata.get("is_truncated", False)),
        "completion_token_count": metadata.get("completion_token_count"),
        "input_token_count": metadata.get("input_token_count"),
        "turn_count": metadata.get("turn_count", rollout.get("num_turns")),
        "tool_calls": rollout.get("tool_calls"),
        "error": _optional_str(rollout.get("error")),
        "sequence_tokens": len(input_ids) if isinstance(input_ids, list) else None,
        "trainable_tokens": (
            sum(1 for flag in loss_mask if flag)
            if isinstance(loss_mask, list)
            else None
        ),
        "logprob_mean": (
            sum(logprob_values) / len(logprob_values) if logprob_values else None
        ),
        "logprob_min": min(logprob_values) if logprob_values else None,
        "has_inference_logprobs": bool(logprob_values),
        "has_teacher_logprobs": isinstance(row.get("teacher_logprobs"), list),
        "prompt": _clip(_message_text(row.get("prompt")), max_text_chars),
        "completion": _clip(_message_text(row.get("completion")), max_text_chars),
    }
    return compact


def compact_eval_row(
    row: dict[str, Any], *, row_index: int, max_text_chars: int = MAX_TEXT_CHARS
) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    example_id = row.get("example_id")
    return {
        "row_index": row_index,
        "example_id": str(example_id) if example_id is not None else None,
        "reward": _numeric(row, "reward"),
        "has_error": row.get("error") is not None,
        "error": _optional_str(row.get("error")),
        "is_truncated": bool(row.get("is_truncated", False)),
        "stop_condition": _optional_str(row.get("stop_condition")),
        "completion_token_count": _eval_completion_tokens(row),
        "task": _optional_str(row.get("task")),
        "answer": _clip(_message_text(row.get("answer")), max_text_chars),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
        "prompt": _clip(
            _message_text(normalize_messages(row.get("prompt"))), max_text_chars
        ),
        "completion": _clip(
            _message_text(normalize_messages(row.get("completion"))), max_text_chars
        ),
    }


def _eval_completion_tokens(row: dict[str, Any]) -> int | None:
    trajectory = row.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        total = 0
        found = False
        for step in trajectory:
            tokens = step.get("tokens") if isinstance(step, dict) else None
            ids = tokens.get("completion_ids") if isinstance(tokens, dict) else None
            if isinstance(ids, list):
                total += len(ids)
                found = True
        if found:
            return total
    return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _clip(text: str | None, max_chars: int) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def sort_rows(
    rows: list[dict[str, Any]], *, key: str, descending: bool
) -> list[dict[str, Any]]:
    """Sort by ``key`` with missing values last regardless of direction."""
    present = [row for row in rows if row.get(key) is not None]
    missing = [row for row in rows if row.get(key) is None]
    if all(isinstance(row[key], int | float) for row in present):
        ordered = sorted(present, key=lambda row: float(row[key]), reverse=descending)
    else:
        ordered = sorted(present, key=lambda row: str(row[key]), reverse=descending)
    return [*ordered, *missing]


def histogram(values: list[float], *, bins: int = 20) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"bins": [], "counts": [], "min": None, "max": None}
    low, high = min(finite), max(finite)
    if low == high:
        return {"bins": [low, high], "counts": [len(finite)], "min": low, "max": high}
    width = (high - low) / bins
    counts = [0] * bins
    for value in finite:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    edges = [low + width * index for index in range(bins + 1)]
    return {"bins": edges, "counts": counts, "min": low, "max": high}


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
    }


def group_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("group_key") or f"example:{row.get('example_id')}"
        groups.setdefault(str(key), []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, members in groups.items():
        rewards = [r["reward"] for r in members if r.get("reward") is not None]
        advantages = [r["advantage"] for r in members if r.get("advantage") is not None]
        tokens = [
            r["completion_token_count"]
            for r in members
            if isinstance(r.get("completion_token_count"), int | float)
        ]
        reward_stats = numeric_summary(rewards)
        first = members[0]
        summaries.append(
            {
                "group_key": key,
                "env": first.get("env"),
                "example_id": first.get("example_id"),
                "policy_step": first.get("policy_step"),
                "size": len(members),
                "reward_mean": reward_stats["mean"],
                "reward_std": reward_stats["std"],
                "reward_min": reward_stats["min"],
                "reward_max": reward_stats["max"],
                "advantage_abs_mean": (
                    statistics.fmean(abs(a) for a in advantages) if advantages else None
                ),
                "solve_all": bool(rewards) and all(r >= 1.0 for r in rewards),
                "solve_none": bool(rewards) and all(r <= 0.0 for r in rewards),
                "zero_signal": bool(rewards) and reward_stats["std"] == 0.0,
                "truncated": sum(1 for r in members if r.get("is_truncated")),
                "completion_tokens_mean": statistics.fmean(tokens) if tokens else None,
                "row_indexes": [r["row_index"] for r in members],
                "prompt": first.get("prompt"),
            }
        )
    return summaries


def group_eval_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("example_id")), []).append(row)
    summaries: list[dict[str, Any]] = []
    for example_id, members in groups.items():
        rewards = [r["reward"] for r in members if r.get("reward") is not None]
        summaries.append(
            {
                "example_id": example_id,
                "attempts": len(members),
                "scored": len(rewards),
                "errors": sum(1 for r in members if r.get("has_error")),
                "reward_mean": statistics.fmean(rewards) if rewards else None,
                "solved_any": any(r >= 1.0 for r in rewards),
                "solved_all": bool(rewards) and all(r >= 1.0 for r in rewards),
                "truncated": sum(1 for r in members if r.get("is_truncated")),
                "row_indexes": [r["row_index"] for r in members],
                "prompt": members[0].get("prompt"),
                "answer": members[0].get("answer"),
            }
        )
    return summaries


class CompactRowCache:
    """Caches compact rows per file until the file changes on disk."""

    def __init__(self, *, max_files: int = 16) -> None:
        self._entries: dict[
            Path, tuple[tuple[int, int], list[dict[str, Any]], int, bool]
        ] = {}
        self._order: list[Path] = []
        self._max_files = max_files
        self._lock = threading.Lock()

    def rows(
        self,
        path: Path,
        *,
        kind: str,
        max_scan_rows: int = MAX_SCAN_ROWS,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        signature = file_signature(path)
        if signature is None:
            return [], 0, False
        with self._lock:
            entry = self._entries.get(path)
            if entry is not None and entry[0] == signature:
                return entry[1], entry[2], entry[3]
        compact_fn = compact_rollout_row if kind == "rollout" else compact_eval_row
        rows: list[dict[str, Any]] = []
        scanned = 0
        scan_limited = False
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if row_index >= max_scan_rows:
                    scan_limited = True
                    break
                scanned = row_index + 1
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(compact_fn(row, row_index=row_index))
        with self._lock:
            self._entries[path] = (signature, rows, scanned, scan_limited)
            if path in self._order:
                self._order.remove(path)
            self._order.append(path)
            while len(self._order) > self._max_files:
                evicted = self._order.pop(0)
                self._entries.pop(evicted, None)
        return rows, scanned, scan_limited


def full_row(path: Path, row_index: int) -> dict[str, Any] | None:
    """Return one raw JSONL row with large token arrays summarized."""
    for index, row in _iter_jsonl_dicts(path, limit=row_index + 1):
        if index != row_index:
            continue
        detail = dict(row)
        detail["row_index"] = row_index
        for key in ("prompt", "completion", "target_completion"):
            if key in detail:
                detail[key] = normalize_messages(detail[key])
        arrays = {}
        for key in (
            "input_ids",
            "target_ids",
            "loss_mask",
            "inference_logprobs",
            "teacher_logprobs",
            "sampling_mask",
        ):
            value = detail.pop(key, None)
            if isinstance(value, list):
                arrays[key] = _array_summary(value)
        if isinstance(detail.get("temperatures"), list):
            detail["temperatures"] = _array_summary(detail["temperatures"])
        detail["arrays"] = arrays
        return detail
    return None


def _array_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    summary: dict[str, Any] = {"length": len(values)}
    if values and all(isinstance(v, bool) for v in values):
        summary["true_count"] = sum(1 for v in values if v)
    elif numeric and len(numeric) == len(values):
        summary.update(
            {
                "min": min(numeric),
                "max": max(numeric),
                "mean": statistics.fmean(numeric),
                "sum": sum(numeric),
            }
        )
    return summary


_REPR_MESSAGE = re.compile(
    r"^(?:\w+\()?role=(?P<role>'[^']*'|\"[^\"]*\") "
    r"content=(?P<content>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",
    re.DOTALL,
)


def normalize_messages(value: Any) -> Any:
    """Turn ``repr``-serialized chat messages back into ``{"role", "content"}`` dicts.

    Older eval rollout files were written with ``json.dumps(default=str)``, so
    each message is a string such as ``role='user' content='...' tool_calls=None``.
    Anything that is not such a string is returned unchanged.
    """
    if isinstance(value, list):
        return [normalize_messages(item) for item in value]
    if not isinstance(value, str):
        return value
    match = _REPR_MESSAGE.match(value.strip())
    if match is None:
        return value
    try:
        role = ast.literal_eval(match.group("role"))
        content = ast.literal_eval(match.group("content"))
    except (ValueError, SyntaxError):
        return value
    return {"role": role, "content": content}

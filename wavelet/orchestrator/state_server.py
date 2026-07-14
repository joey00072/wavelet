from __future__ import annotations

import json
import random
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.queue import (
    build_queue_report,
    get_step_dir,
    resolve_policy_dir,
    resolve_queue_dir,
)
from wavelet.orchestrator.queue.types import (
    MANIFEST_FILENAME,
    STABLE_BATCH_MARKER,
    STEP_DIR_PREFIX,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                token in lowered for token in ("api_key", "token", "secret", "password")
            ):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _tail_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_jsonl_dicts(
    path: Path, *, limit: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    if limit <= 0:
        return
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row_index, row


def _numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _message_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(f"{role}: {content}" if role else content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts) if parts else None
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _compact_rollout_row(
    row: dict[str, Any],
    *,
    row_index: int,
    max_text_chars: int,
) -> dict[str, Any]:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    sample = {
        "row_index": row_index,
        "reward": _numeric(row, "reward"),
        "advantage": _numeric(row, "advantage"),
        "source": row.get("source"),
        "env_name": row.get("env_name"),
        "task": row.get("task"),
        "example_id": row.get("example_id"),
        "group_key": metadata.get("group_key"),
        "rollout_key": metadata.get("rollout_key"),
        "stop_condition": metadata.get("stop_condition"),
        "is_truncated": metadata.get("is_truncated"),
        "completion_token_count": metadata.get("completion_token_count"),
        "turn_count": metadata.get("turn_count"),
        "prompt": _message_text(row.get("prompt")),
        "completion": _message_text(row.get("completion")),
        "target_completion": _message_text(row.get("target_completion")),
    }
    for key in ("prompt", "completion", "target_completion"):
        value = sample.get(key)
        if isinstance(value, str) and len(value) > max_text_chars:
            sample[key] = value[:max_text_chars] + "...<truncated>"
    return {key: value for key, value in sample.items() if value is not None}


def _empty_stats() -> dict[str, Any]:
    return {"count": 0, "min": None, "max": None, "mean": None, "std": None}


def _finalize_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return _empty_stats()
    total = sum(values)
    count = len(values)
    mean = total / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": variance**0.5,
    }


def _unavailable_rollout_inspection(
    reason: str,
    *,
    queue_step: int | None,
    path: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "queue_step": queue_step,
        "path": None if path is None else str(path),
        "manifest": manifest,
        "scanned_rows": 0,
        "truncated": False,
        "stats": {"reward": _empty_stats(), "advantage": _empty_stats()},
        "samples": {
            "random": [],
            "min_reward": None,
            "max_reward": None,
            "near_mean_reward": None,
        },
    }


@dataclass
class _RolloutScan:
    reward_values: list[float]
    advantage_values: list[float]
    random_rows: list[dict[str, Any]]
    min_reward_row: dict[str, Any] | None
    max_reward_row: dict[str, Any] | None
    scanned_rows: int


def _scan_rollouts(
    path: Path,
    *,
    random_count: int,
    seed: int | None,
    max_scan_rows: int,
    max_text_chars: int,
) -> _RolloutScan:
    rng = random.Random(seed)
    rewards: list[float] = []
    advantages: list[float] = []
    random_rows: list[dict[str, Any]] = []
    min_pair: tuple[float, dict[str, Any]] | None = None
    max_pair: tuple[float, dict[str, Any]] | None = None
    scanned_rows = 0

    for scanned_rows, (row_index, row) in enumerate(
        _iter_jsonl_dicts(path, limit=max_scan_rows),
        start=1,
    ):
        compact = _compact_rollout_row(
            row,
            row_index=row_index,
            max_text_chars=max_text_chars,
        )
        reward = _numeric(row, "reward")
        if reward is not None:
            rewards.append(reward)
            if min_pair is None or reward < min_pair[0]:
                min_pair = (reward, compact)
            if max_pair is None or reward > max_pair[0]:
                max_pair = (reward, compact)

        advantage = _numeric(row, "advantage")
        if advantage is not None:
            advantages.append(advantage)

        if len(random_rows) < random_count:
            random_rows.append(compact)
        elif random_count > 0:
            replacement = rng.randint(0, scanned_rows - 1)
            if replacement < random_count:
                random_rows[replacement] = compact

    return _RolloutScan(
        reward_values=rewards,
        advantage_values=advantages,
        random_rows=random_rows,
        min_reward_row=None if min_pair is None else min_pair[1],
        max_reward_row=None if max_pair is None else max_pair[1],
        scanned_rows=scanned_rows,
    )


def _nearest_reward_row(
    path: Path,
    *,
    target: float | None,
    max_scan_rows: int,
    max_text_chars: int,
) -> dict[str, Any] | None:
    if target is None:
        return None
    nearest: tuple[float, dict[str, Any]] | None = None
    for row_index, row in _iter_jsonl_dicts(path, limit=max_scan_rows):
        reward = _numeric(row, "reward")
        if reward is None:
            continue
        distance = abs(reward - target)
        if nearest is None or distance < nearest[0]:
            nearest = (
                distance,
                _compact_rollout_row(
                    row,
                    row_index=row_index,
                    max_text_chars=max_text_chars,
                ),
            )
    return None if nearest is None else nearest[1]


def _file_exceeds_rows(path: Path, *, limit: int) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return any(index >= limit for index, _ in enumerate(handle))
    except OSError:
        return False


class OrchestratorRunState:
    def __init__(self, config: RLConfig, *, target_step: int) -> None:
        self._lock = threading.Lock()
        self._config = config
        self._state: dict[str, Any] = {
            "status": "starting",
            "phase": "initializing",
            "started_at": _now(),
            "updated_at": _now(),
            "target_step": target_step,
            "output_dir": str(config.output_dir),
            "launcher_mode": config.launcher.mode,
            "rollouts": {
                "next_queue_step_to_submit": 0,
                "next_queue_step_to_publish": 0,
                "pending_count": 0,
                "completed_count": 0,
                "submitted_tail": [],
                "completed_tail": [],
                "published_tail": [],
            },
            "events": [],
            "policy": {
                "loaded_step": None,
                "pending_load": False,
                "requested_step": None,
                "available_tail": [],
            },
            "errors": [],
        }

    def set_status(self, status: str, *, phase: str | None = None) -> None:
        patch: dict[str, Any] = {"status": status}
        if phase is not None:
            patch["phase"] = phase
        self.update(**patch)

    def set_error(self, error: BaseException) -> None:
        with self._lock:
            self._state["status"] = "failed"
            self._state["phase"] = "failed"
            self._state["updated_at"] = _now()
            self._state.setdefault("errors", []).append(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": _now(),
                }
            )

    def update(self, **patch: Any) -> None:
        with self._lock:
            self._state.update(patch)
            self._state["updated_at"] = _now()

    def update_rollouts(self, **patch: Any) -> None:
        with self._lock:
            self._state["rollouts"].update(patch)
            self._state["updated_at"] = _now()

    def update_policy(self, **patch: Any) -> None:
        with self._lock:
            self._state["policy"].update(patch)
            self._state["updated_at"] = _now()

    def mark_submitted(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        pending_count: int,
    ) -> None:
        item = {
            "type": "submitted",
            "queue_step": queue_step,
            "optimizer_step": optimizer_step,
            "chunk_index": chunk_index,
            "timestamp": _now(),
        }
        self._append_tail("submitted_tail", item, pending_count=pending_count)

    def mark_completed(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        pending_count: int,
        completed_count: int,
    ) -> None:
        item = {
            "type": "completed",
            "queue_step": queue_step,
            "optimizer_step": optimizer_step,
            "chunk_index": chunk_index,
            "timestamp": _now(),
        }
        self._append_tail(
            "completed_tail",
            item,
            pending_count=pending_count,
            completed_count=completed_count,
        )

    def mark_published(
        self,
        *,
        queue_step: int,
        optimizer_step: int | None = None,
        chunk_index: int | None = None,
        path: str,
        next_queue_step_to_publish: int,
        completed_count: int,
    ) -> None:
        item = {
            "type": "published",
            "queue_step": queue_step,
            "optimizer_step": optimizer_step,
            "chunk_index": chunk_index,
            "path": path,
            "timestamp": _now(),
        }
        self._append_tail(
            "published_tail",
            item,
            next_queue_step_to_publish=next_queue_step_to_publish,
            completed_count=completed_count,
        )

    def _append_tail(
        self, key: str, item: dict[str, Any], **rollout_patch: Any
    ) -> None:
        with self._lock:
            tail = list(self._state["rollouts"].get(key, []))
            tail.append(item)
            self._state["rollouts"][key] = tail[-20:]
            events = list(self._state.get("events", []))
            events.append(item)
            self._state["events"] = events[
                -self._config.orchestrator.state_server.max_events :
            ]
            self._state["rollouts"].update(rollout_patch)
            self._state["updated_at"] = _now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = json.loads(json.dumps(self._state))
        queue_report = self.queue_snapshot(detail=False, limit=0)
        if queue_report is not None:
            snapshot["queue_summary"] = queue_report.get("summary")
            snapshot["queue_rates"] = queue_report.get("rates")
        return snapshot

    def queue_snapshot(
        self,
        *,
        detail: bool,
        limit: int,
    ) -> dict[str, Any] | None:
        try:
            queue_dir = resolve_queue_dir(
                self._config.output_dir, self._config.transport
            )
            policy_dir = resolve_policy_dir(
                self._config.output_dir,
                self._config.policy_transfer,
            )
            return build_queue_report(
                queue_dir=queue_dir,
                policy_dir=policy_dir,
                events_dir=self._config.output_dir / "events",
                detail=detail,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - health should degrade only
            return {
                "summary": None,
                "policy": None,
                "rates": None,
                "errors": {"queue_snapshot": str(exc)},
                "items": [],
            }

    def sanitized_config(self) -> dict[str, Any]:
        return _redact(self._config.model_dump(mode="json", exclude_none=True))

    def metrics(self, *, limit: int) -> list[dict[str, Any]]:
        return _tail_jsonl(self._config.output_dir / "metrics.jsonl", limit=limit)

    def samples(self, *, limit: int) -> list[dict[str, Any]]:
        rows = _tail_jsonl(self._config.output_dir / "samples.jsonl", limit=limit)
        for row in rows:
            if "input_ids" in row:
                row["input_ids"] = "<omitted>"
        return rows

    def inspect_rollouts(
        self,
        *,
        step: int | None,
        random_count: int,
        seed: int | None,
        max_scan_rows: int,
        max_text_chars: int,
    ) -> dict[str, Any]:
        queue_dir = resolve_queue_dir(self._config.output_dir, self._config.transport)
        queue_step = (
            step if step is not None else self._latest_stable_rollout_step(queue_dir)
        )
        if queue_step is None:
            return _unavailable_rollout_inspection(
                "no stable rollout batches found",
                queue_step=None,
            )

        step_dir = get_step_dir(queue_dir, queue_step)
        rollout_path = step_dir / self._config.transport.rollout_filename
        stable = (step_dir / STABLE_BATCH_MARKER).exists()
        if not stable or not rollout_path.exists():
            return _unavailable_rollout_inspection(
                f"rollout batch step {queue_step} is not stable",
                queue_step=queue_step,
                path=rollout_path,
                manifest=_read_json(step_dir / MANIFEST_FILENAME),
            )

        scan = _scan_rollouts(
            rollout_path,
            random_count=random_count,
            seed=seed,
            max_scan_rows=max_scan_rows,
            max_text_chars=max_text_chars,
        )
        reward_stats = _finalize_stats(scan.reward_values)
        near_mean_row = _nearest_reward_row(
            rollout_path,
            target=reward_stats["mean"],
            max_scan_rows=max_scan_rows,
            max_text_chars=max_text_chars,
        )

        return {
            "available": True,
            "reason": None,
            "queue_step": queue_step,
            "path": str(rollout_path),
            "manifest": _read_json(step_dir / MANIFEST_FILENAME),
            "scanned_rows": scan.scanned_rows,
            "truncated": _file_exceeds_rows(rollout_path, limit=max_scan_rows),
            "stats": {
                "reward": reward_stats,
                "advantage": _finalize_stats(scan.advantage_values),
            },
            "samples": {
                "random": scan.random_rows,
                "min_reward": scan.min_reward_row,
                "max_reward": scan.max_reward_row,
                "near_mean_reward": near_mean_row,
            },
        }

    def _latest_stable_rollout_step(self, queue_dir: Path) -> int | None:
        if not queue_dir.exists():
            return None
        steps: list[int] = []
        for candidate in queue_dir.iterdir():
            if not candidate.is_dir() or not candidate.name.startswith(STEP_DIR_PREFIX):
                continue
            if not (candidate / STABLE_BATCH_MARKER).exists():
                continue
            if not (candidate / self._config.transport.rollout_filename).exists():
                continue
            try:
                steps.append(int(candidate.name.removeprefix(STEP_DIR_PREFIX)))
            except ValueError:
                continue
        return max(steps) if steps else None

    def events(self, *, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._state.get("events", []))[-limit:]


def _build_state_app(
    state: OrchestratorRunState,
    *,
    fastapi: Any,
    query: Any,
    cors_middleware: Any,
) -> Any:
    config = state._config.orchestrator.state_server
    app = fastapi(title="Wavelet Orchestrator State", version="1.0")
    app.add_middleware(
        cors_middleware,
        allow_origins=config.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        snapshot = state.snapshot()
        return {
            "ok": snapshot.get("status") != "failed",
            "status": snapshot.get("status"),
            "phase": snapshot.get("phase"),
            "updated_at": snapshot.get("updated_at"),
        }

    @app.get("/state")
    async def state_snapshot() -> dict[str, Any]:
        return state.snapshot()

    @app.get("/queues")
    async def queues(
        detail: bool = query(default=False),
        limit: int = query(default=100, ge=0, le=1000),
    ) -> dict[str, Any]:
        snapshot = state.queue_snapshot(detail=detail, limit=limit)
        return snapshot or {
            "summary": None,
            "policy": None,
            "rates": None,
            "errors": {"queue_snapshot": "unavailable"},
            "items": [],
        }

    @app.get("/config")
    async def run_config() -> dict[str, Any]:
        return state.sanitized_config()

    @app.get("/metrics")
    async def metrics(
        limit: int = query(default=20, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return state.metrics(limit=limit)

    @app.get("/events")
    async def events(
        limit: int = query(default=500, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return state.events(limit=limit)

    @app.get("/samples")
    async def samples(
        limit: int = query(default=10, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        return state.samples(limit=limit)

    @app.get("/rollouts/inspect")
    async def rollout_inspection(
        step: int | None = query(default=None, ge=0),
        random_count: int = query(default=3, ge=0, le=20),
        seed: int | None = query(default=None),
        max_scan_rows: int = query(default=5000, ge=1, le=50000),
        max_text_chars: int = query(default=4000, ge=200, le=20000),
    ) -> dict[str, Any]:
        return state.inspect_rollouts(
            step=step,
            random_count=random_count,
            seed=seed,
            max_scan_rows=max_scan_rows,
            max_text_chars=max_text_chars,
        )

    return app


class StateServerHandle:
    def __init__(self, state: OrchestratorRunState) -> None:
        self.state = state
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        config = self.state._config.orchestrator.state_server
        return f"http://{config.host}:{config.port}"

    def start(self) -> None:
        config = self.state._config.orchestrator.state_server
        if not config.enabled:
            return
        try:
            import uvicorn
            from fastapi import FastAPI, Query
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError as exc:
            raise RuntimeError(
                "orchestrator.state_server.enabled requires fastapi and uvicorn."
            ) from exc

        app = _build_state_app(
            self.state,
            fastapi=FastAPI,
            query=Query,
            cors_middleware=CORSMiddleware,
        )

        uvicorn_config = uvicorn.Config(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level,
        )
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="wavelet-state-server",
            daemon=True,
        )
        self._thread.start()
        self.state.update(state_server={"enabled": True, "url": self.url})

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


@contextmanager
def maybe_state_server(
    config: RLConfig,
    *,
    target_step: int,
) -> Iterator[OrchestratorRunState | None]:
    if not config.orchestrator.state_server.enabled:
        yield None
        return
    state = OrchestratorRunState(config, target_step=target_step)
    handle = StateServerHandle(state)
    handle.start()
    try:
        yield state
    except BaseException as exc:
        state.set_error(exc)
        raise
    finally:
        if state.snapshot().get("status") not in {"failed", "completed"}:
            state.set_status("stopped", phase="shutdown")
        handle.stop()

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig


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
            self._state["events"] = events[-self._config.orchestrator.state_server.max_events :]
            self._state["rollouts"].update(rollout_patch)
            self._state["updated_at"] = _now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

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

    def events(self, *, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._state.get("events", []))[-limit:]


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

        app = FastAPI(title="Wavelet Orchestrator State", version="1.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health() -> dict[str, Any]:
            snapshot = self.state.snapshot()
            return {
                "ok": snapshot.get("status") != "failed",
                "status": snapshot.get("status"),
                "phase": snapshot.get("phase"),
                "updated_at": snapshot.get("updated_at"),
            }

        @app.get("/state")
        async def state() -> dict[str, Any]:
            return self.state.snapshot()

        @app.get("/config")
        async def run_config() -> dict[str, Any]:
            return self.state.sanitized_config()

        @app.get("/metrics")
        async def metrics(
            limit: int = Query(default=20, ge=1, le=200),
        ) -> list[dict[str, Any]]:
            return self.state.metrics(limit=limit)

        @app.get("/events")
        async def events(
            limit: int = Query(default=500, ge=1, le=5000),
        ) -> list[dict[str, Any]]:
            return self.state.events(limit=limit)

        @app.get("/samples")
        async def samples(
            limit: int = Query(default=10, ge=1, le=50),
        ) -> list[dict[str, Any]]:
            return self.state.samples(limit=limit)

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

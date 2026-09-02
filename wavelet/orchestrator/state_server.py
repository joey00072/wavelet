from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from wavelet.configs.config import RLConfig
from wavelet.monitor import (
    RolloutStateEventsMixin,
    _file_exceeds_rows,
    _nearest_reward_row,
    _read_json,
    _scan_rollouts,
    latest_stable_step,
    summary_stats,
    tail_jsonl_rows,
    unavailable_rollout_inspection,
)
from wavelet.monitor import redact as _redact
from wavelet.transport.queue import (
    MANIFEST_FILENAME,
    STABLE_BATCH_MARKER,
    STEP_DIR_PREFIX,
    build_queue_report,
    get_step_dir,
    resolve_policy_dir,
    resolve_queue_dir,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OrchestratorRunState(RolloutStateEventsMixin):
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
        return tail_jsonl_rows(
            self._config.output_dir / "metrics.jsonl",
            limit=limit,
        )

    def eval_metrics(self, *, limit: int) -> list[dict[str, Any]]:
        return tail_jsonl_rows(
            self._config.output_dir / "eval_metrics.jsonl",
            limit=limit,
        )

    def samples(self, *, limit: int) -> list[dict[str, Any]]:
        rows = tail_jsonl_rows(
            self._config.output_dir / "samples.jsonl",
            limit=limit,
        )
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
            step
            if step is not None
            else latest_stable_step(
                queue_dir,
                prefix=STEP_DIR_PREFIX,
                marker=STABLE_BATCH_MARKER,
                filename=self._config.transport.rollout_filename,
            )
        )
        if queue_step is None:
            return unavailable_rollout_inspection(
                "no stable rollout batches found",
                queue_step=None,
            )

        step_dir = get_step_dir(queue_dir, queue_step)
        rollout_path = step_dir / self._config.transport.rollout_filename
        stable = (step_dir / STABLE_BATCH_MARKER).exists()
        if not stable or not rollout_path.exists():
            return unavailable_rollout_inspection(
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
        reward_stats = summary_stats(scan.reward_values)
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
                "advantage": summary_stats(scan.advantage_values),
            },
            "samples": {
                "random": scan.random_rows,
                "min_reward": scan.min_reward_row,
                "max_reward": scan.max_reward_row,
                "near_mean_reward": near_mean_row,
            },
        }

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

    @app.get("/eval-metrics")
    async def eval_metrics(
        limit: int = query(default=1000, ge=1, le=2000),
    ) -> list[dict[str, Any]]:
        return state.eval_metrics(limit=limit)

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

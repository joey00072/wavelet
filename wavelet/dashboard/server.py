"""HTTP API for the run dashboard.

The same router serves a standalone multi-run dashboard (``wavelet dashboard``)
and the live orchestrator state server, so the web UI has one contract.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wavelet.dashboard.artifacts import RunArtifacts, discover_runs
from wavelet.dashboard.metrics import MAX_SERIES_POINTS
from wavelet.dashboard.rows import RowFilters

API_PREFIX = "/api"
CURRENT_RUN_ALIAS = "current"
DEFAULT_MAX_READERS = 8


class RunRegistry:
    """Resolves run ids to cached artifact readers."""

    def __init__(
        self,
        *,
        roots: list[Path] | None = None,
        runs: list[Path] | None = None,
        live: dict[str, Path] | None = None,
        max_readers: int = DEFAULT_MAX_READERS,
    ) -> None:
        self.roots = [Path(root) for root in roots or []]
        self.explicit = [Path(run) for run in runs or []]
        self.live = {run_id: Path(path) for run_id, path in (live or {}).items()}
        # Insertion order doubles as LRU order; idle readers are evicted so the
        # server's memory is bounded by ``max_readers`` runs, not by history.
        self._readers: dict[str, RunArtifacts] = {}
        self._max_readers = max(int(max_readers), 1)
        self._lock = threading.Lock()

    def discover(self) -> dict[str, Path]:
        found = dict(self.live)
        for run_id, path in discover_runs(self.roots, self.explicit).items():
            if path.resolve() in {p.resolve() for p in found.values()}:
                continue
            found.setdefault(run_id, path)
        return found

    def reader(self, run_id: str) -> RunArtifacts | None:
        if run_id == CURRENT_RUN_ALIAS:
            current = self.current_run_id()
            return None if current is None else self.reader(current)
        with self._lock:
            reader = self._readers.get(run_id)
            if reader is not None and reader.output_dir.is_dir():
                self._readers[run_id] = self._readers.pop(run_id)
                return reader
        path = self.discover().get(run_id)
        if path is None:
            return None
        reader = RunArtifacts(run_id, path)
        with self._lock:
            self._readers.pop(run_id, None)
            self._readers[run_id] = reader
            evictable = [key for key in self._readers if key not in self.live]
            while len(self._readers) > self._max_readers and evictable:
                del self._readers[evictable.pop(0)]
        return reader

    def summaries(self) -> list[dict[str, Any]]:
        """Run summaries with the current run first, then newest to oldest."""
        summaries: list[dict[str, Any]] = []
        for run_id in self.discover():
            reader = self.reader(run_id)
            if reader is not None:
                summaries.append(reader.summary())
        current = current_run_id(summaries)
        for summary in summaries:
            summary["is_current"] = summary["id"] == current
        summaries.sort(
            key=lambda s: (s["is_current"], s.get("updated_at") or ""), reverse=True
        )
        return summaries

    def current_run_id(self) -> str | None:
        return current_run_id(self.summaries())


def current_run_id(summaries: list[dict[str, Any]]) -> str | None:
    """Pick the run a single-run host is working on right now.

    A fresh ``running`` heartbeat wins, then a ``running`` run whose heartbeat
    went stale, then whichever run wrote artifacts most recently.
    """
    if not summaries:
        return None
    rank = {"running": 2, "stale": 1}

    def key(summary: dict[str, Any]) -> tuple[int, str]:
        return rank.get(str(summary.get("status")), 0), summary.get("updated_at") or ""

    return str(max(summaries, key=key)["id"])


def build_dashboard_app(
    registry: RunRegistry,
    *,
    static_dir: Path | None = None,
    cors_allow_origins: list[str] | None = None,
    title: str = "Wavelet Dashboard",
) -> Any:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title=title, version="2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    register_run_routes(app, registry, http_exception=HTTPException, query=Query)
    if static_dir is not None and (static_dir / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="webui")
    return app


def register_run_routes(
    app: Any,
    registry: RunRegistry,
    *,
    http_exception: type[Exception],
    query: Callable[..., Any],
) -> None:
    def reader_or_404(run_id: str) -> RunArtifacts:
        reader = registry.reader(run_id)
        if reader is None:
            raise http_exception(status_code=404, detail=f"Unknown run '{run_id}'.")
        return reader

    def filters_from(
        env: str | None,
        group_key: str | None,
        example_id: str | None,
        min_reward: float | None,
        max_reward: float | None,
        truncated: bool | None,
        stop_condition: str | None,
        advantage: str | None,
        has_error: bool | None,
        search: str | None,
    ) -> RowFilters:
        return RowFilters(
            env=env,
            group_key=group_key,
            example_id=example_id,
            min_reward=min_reward,
            max_reward=max_reward,
            truncated=truncated,
            stop_condition=stop_condition,
            advantage_sign=advantage,
            has_error=has_error,
            search=search,
        )

    @app.get(f"{API_PREFIX}/health")
    async def api_health() -> dict[str, Any]:
        return {"ok": True, "runs": len(registry.discover())}

    @app.get(f"{API_PREFIX}/runs")
    async def api_runs() -> list[dict[str, Any]]:
        return registry.summaries()

    @app.get(f"{API_PREFIX}/current")
    async def api_current() -> dict[str, Any]:
        summaries = registry.summaries()
        current = next((s for s in summaries if s.get("is_current")), None)
        return {
            "id": None if current is None else current["id"],
            "runs": len(summaries),
        }

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/summary")
    async def api_summary(run_id: str) -> dict[str, Any]:
        return reader_or_404(run_id).summary()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/config")
    async def api_config(run_id: str) -> dict[str, Any]:
        return reader_or_404(run_id).sanitized_config()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/external")
    async def api_external(run_id: str) -> list[dict[str, Any]]:
        return reader_or_404(run_id).external_status()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/metrics/keys")
    async def api_metric_keys(run_id: str) -> dict[str, Any]:
        return reader_or_404(run_id).metric_keys()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/series")
    async def api_series(
        run_id: str,
        source: str = query(default="trainer"),
        keys: str = query(default=""),
        limit: int = query(default=0, ge=0, le=100_000),
        start: float | None = query(default=None),
        end: float | None = query(default=None),
        after: float | None = query(default=None),
        points: int = query(default=0, ge=0, le=MAX_SERIES_POINTS),
    ) -> dict[str, Any]:
        reader = reader_or_404(run_id)
        key_list = [key for key in keys.split(",") if key]
        try:
            return reader.series(
                source,
                key_list,
                limit=limit,
                start=start,
                end=end,
                after=after,
                points=points,
            )
        except KeyError as exc:
            raise http_exception(status_code=400, detail=str(exc)) from exc

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/nodes")
    async def api_nodes(run_id: str) -> dict[str, Any]:
        return reader_or_404(run_id).nodes()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/evals")
    async def api_evals(run_id: str) -> dict[str, Any]:
        return reader_or_404(run_id).evals()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/evals/{{step}}/{{env}}/rows")
    async def api_eval_rows(
        run_id: str,
        step: int,
        env: str,
        sort: str = query(default="row_index"),
        order: str = query(default="asc"),
        offset: int = query(default=0, ge=0),
        limit: int = query(default=50, ge=1, le=500),
        example_id: str | None = query(default=None),
        min_reward: float | None = query(default=None),
        max_reward: float | None = query(default=None),
        truncated: bool | None = query(default=None),
        has_error: bool | None = query(default=None),
        search: str | None = query(default=None),
    ) -> dict[str, Any]:
        return reader_or_404(run_id).eval_rows(
            step,
            env,
            sort=sort,
            descending=order == "desc",
            offset=offset,
            limit=limit,
            filters=filters_from(
                None,
                None,
                example_id,
                min_reward,
                max_reward,
                truncated,
                None,
                None,
                has_error,
                search,
            ),
        )

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/evals/{{step}}/{{env}}/rows/{{row_index}}")
    async def api_eval_row(
        run_id: str, step: int, env: str, row_index: int
    ) -> dict[str, Any]:
        row = reader_or_404(run_id).eval_row(step, env, row_index)
        if row is None:
            raise http_exception(status_code=404, detail="Eval row not found.")
        return row

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/rollouts")
    async def api_rollout_batches(
        run_id: str,
        limit: int = query(default=200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return reader_or_404(run_id).rollout_batches(limit=limit)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/rollouts/rows")
    async def api_rollout_rows(
        run_id: str,
        step: int | None = query(default=None, ge=0),
        sort: str = query(default="row_index"),
        order: str = query(default="asc"),
        offset: int = query(default=0, ge=0),
        limit: int = query(default=50, ge=1, le=500),
        env: str | None = query(default=None),
        group_key: str | None = query(default=None),
        example_id: str | None = query(default=None),
        min_reward: float | None = query(default=None),
        max_reward: float | None = query(default=None),
        truncated: bool | None = query(default=None),
        stop_condition: str | None = query(default=None),
        advantage: str | None = query(default=None),
        has_error: bool | None = query(default=None),
        search: str | None = query(default=None),
    ) -> dict[str, Any]:
        return reader_or_404(run_id).rollout_rows(
            step,
            sort=sort,
            descending=order == "desc",
            offset=offset,
            limit=limit,
            filters=filters_from(
                env,
                group_key,
                example_id,
                min_reward,
                max_reward,
                truncated,
                stop_condition,
                advantage,
                has_error,
                search,
            ),
        )

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/rollouts/{{step}}/rows/{{row_index}}")
    async def api_rollout_row(run_id: str, step: int, row_index: int) -> dict[str, Any]:
        row = reader_or_404(run_id).rollout_row(step, row_index)
        if row is None:
            raise http_exception(status_code=404, detail="Rollout row not found.")
        return row

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/queue")
    async def api_queue(
        run_id: str,
        detail: bool = query(default=True),
        limit: int = query(default=200, ge=0, le=5000),
    ) -> dict[str, Any]:
        return reader_or_404(run_id).queue(detail=detail, limit=limit)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/timeline")
    async def api_timeline(
        run_id: str,
        limit: int = query(default=500, ge=1, le=20000),
    ) -> dict[str, Any]:
        return reader_or_404(run_id).timeline(limit=limit)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
    async def api_events(
        run_id: str,
        limit: int = query(default=200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return reader_or_404(run_id).run_events(limit=limit)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/samples")
    async def api_samples(
        run_id: str,
        limit: int = query(default=20, ge=1, le=256),
    ) -> list[dict[str, Any]]:
        return reader_or_404(run_id).samples(limit=limit)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/logs")
    async def api_logs(run_id: str) -> list[dict[str, Any]]:
        return reader_or_404(run_id).logs()

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/logs/{{name}}")
    async def api_log_tail(
        run_id: str,
        name: str,
        lines: int = query(default=200, ge=1, le=5000),
    ) -> dict[str, Any]:
        tail = reader_or_404(run_id).log_tail(name, lines=lines)
        if tail is None:
            raise http_exception(status_code=404, detail="Log not found.")
        return tail

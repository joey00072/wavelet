"""Metric history from external trackers a run also logs to.

Two providers exist. Weights & Biases is read through the public API using the
run id the monitor persists in ``wandb_run_id.txt``. Trackio is read straight
from its SQLite store under ``TRACKIO_DIR``. Both are best effort: a fetch runs
in a background thread with a TTL, and callers always get the last good table
plus a status describing whether it is loading, ready, or failed. Nothing here
is required for the dashboard's own JSONL sources.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.dashboard.metrics import MetricTable

logger = logging.getLogger(__name__)

EXTERNAL_SOURCES = ("wandb", "trackio")
REFRESH_SECONDS = 120.0
TRACKIO_RUN_FILENAME = "trackio_run.json"


def _now() -> float:
    return datetime.now(UTC).timestamp()


def rows_to_table(rows: Iterable[dict[str, Any]]) -> MetricTable:
    table = MetricTable()
    for row in rows:
        table.append(row)
    return table


class ExternalSource:
    """One provider's cached table and fetch state."""

    def __init__(
        self,
        name: str,
        *,
        fetch: Callable[[], list[dict[str, Any]]],
        available: bool,
        reason: str,
        refresh_seconds: float = REFRESH_SECONDS,
    ) -> None:
        self.name = name
        self.available = available
        self.reason = reason
        self._fetch = fetch
        self._refresh_seconds = refresh_seconds
        self._table = MetricTable()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._fetched_at: float | None = None
        self._error: str | None = None

    def table(self) -> MetricTable:
        self._maybe_refresh()
        with self._lock:
            return self._table

    def status(self) -> dict[str, Any]:
        self._maybe_refresh()
        with self._lock:
            loading = self._thread is not None and self._thread.is_alive()
            if not self.available:
                state = "unavailable"
            elif self._error and self._fetched_at is None:
                state = "error"
            elif self._fetched_at is None:
                state = "loading"
            else:
                state = "ready"
            return {
                "source": self.name,
                "status": state,
                "reason": self.reason,
                "error": self._error,
                "rows": len(self._table),
                "keys": len(self._table.counts),
                "fetched_at": (
                    datetime.fromtimestamp(self._fetched_at, UTC).isoformat()
                    if self._fetched_at
                    else None
                ),
                "refreshing": loading,
            }

    def _maybe_refresh(self) -> None:
        if not self.available:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            fresh = (
                self._fetched_at is not None
                and _now() - self._fetched_at < self._refresh_seconds
            )
            if fresh:
                return
            self._thread = threading.Thread(
                target=self._run, name=f"external-{self.name}", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            rows = self._fetch()
            table = rows_to_table(rows)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as status
            logger.warning("External source %s failed: %s", self.name, exc)
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                if self._fetched_at is None:
                    self._fetched_at = None
                else:
                    self._fetched_at = _now()
            return
        with self._lock:
            self._table = table
            self._error = None
            self._fetched_at = _now()

    def refresh_now(self) -> None:
        """Synchronous fetch for tests and one-shot tooling."""
        self._run()


# ------------------------------------------------------------------ W&B


def wandb_run_path(config: RLConfig, output_dir: Path) -> str | None:
    run_id_file = output_dir / "wandb_run_id.txt"
    if not run_id_file.is_file():
        return None
    run_id = run_id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        return None
    if "/" in run_id:
        return run_id
    project = config.monitor.wandb.project or "wavelet"
    entity = config.monitor.wandb.entity
    return f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"


def fetch_wandb_history(path: str, *, api: Any | None = None) -> list[dict[str, Any]]:
    """Return W&B history rows as dashboard metric rows."""
    if api is None:
        import wandb

        api = wandb.Api(timeout=30)
    run = api.run(path)
    rows: list[dict[str, Any]] = []
    for record in run.scan_history():
        row: dict[str, Any] = {}
        for key, value in record.items():
            if key == "_step":
                row["step"] = value
            elif key == "_timestamp" and isinstance(value, int | float):
                row["timestamp"] = datetime.fromtimestamp(value, UTC).isoformat()
            elif key.startswith("_"):
                continue
            else:
                row[key] = value
        rows.append(row)
    return rows


def wandb_source(config: RLConfig, output_dir: Path) -> ExternalSource:
    wandb_config = config.monitor.wandb
    path = wandb_run_path(config, output_dir)
    if not wandb_config.enabled or wandb_config.mode == "disabled":
        available, reason = False, "monitor.wandb is disabled for this run"
    elif path is None:
        available, reason = False, "no wandb_run_id.txt in the run directory"
    else:
        available, reason = True, f"W&B run {path}"
    return ExternalSource(
        "wandb",
        fetch=lambda: fetch_wandb_history(path or ""),
        available=available,
        reason=reason,
    )


# ------------------------------------------------------------------ Trackio


def trackio_dir() -> Path:
    override = os.environ.get("TRACKIO_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "huggingface" / "trackio"


def trackio_identity(config: RLConfig, output_dir: Path) -> tuple[str | None, str]:
    """Return ``(project, run_name)`` for a run, from its sidecar or config."""
    sidecar = output_dir / TRACKIO_RUN_FILENAME
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            project = payload.get("project")
            run_name = payload.get("run") or payload.get("name")
            if run_name:
                return (str(project) if project else None, str(run_name))
    run_name = config.monitor.wandb.name or output_dir.name
    return (config.monitor.wandb.project, run_name)


def trackio_databases(root: Path, project: str | None) -> list[Path]:
    if not root.is_dir():
        return []
    if project:
        candidate = root / f"{project}.db"
        return [candidate] if candidate.is_file() else []
    return sorted(root.glob("*.db"))


def fetch_trackio_history(databases: list[Path], run_name: str) -> list[dict[str, Any]]:
    """Read Trackio's ``metrics`` table for one run across candidate databases."""
    rows: list[dict[str, Any]] = []
    for database in databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            cursor = connection.execute(
                "SELECT timestamp, step, metrics FROM metrics "
                "WHERE run_name = ? ORDER BY step, id",
                (run_name,),
            )
            for timestamp, step, metrics in cursor.fetchall():
                try:
                    payload = json.loads(metrics) if isinstance(metrics, str) else {}
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                row: dict[str, Any] = {"step": step, "timestamp": timestamp}
                row.update(payload)
                rows.append(row)
        except sqlite3.DatabaseError as exc:
            logger.debug("Trackio database %s unreadable: %s", database, exc)
        finally:
            connection.close()
        if rows:
            break
    return rows


def trackio_has_run(databases: list[Path], run_name: str) -> bool:
    for database in databases:
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        except sqlite3.DatabaseError:
            continue
        try:
            found = connection.execute(
                "SELECT 1 FROM metrics WHERE run_name = ? LIMIT 1", (run_name,)
            ).fetchone()
        except sqlite3.DatabaseError:
            found = None
        finally:
            connection.close()
        if found:
            return True
    return False


def trackio_source(
    config: RLConfig, output_dir: Path, *, root: Path | None = None
) -> ExternalSource:
    root = root or trackio_dir()
    project, run_name = trackio_identity(config, output_dir)
    databases = trackio_databases(root, project)
    if not databases:
        available, reason = False, f"no Trackio database under {root}"
    elif not trackio_has_run(databases, run_name):
        available, reason = False, f"run '{run_name}' not found in Trackio"
    else:
        available, reason = True, f"Trackio run {run_name} ({root})"
    return ExternalSource(
        "trackio",
        fetch=lambda: fetch_trackio_history(databases, run_name),
        available=available,
        reason=reason,
    )


class ExternalSources:
    """All external providers for one run directory."""

    def __init__(self, config: RLConfig, output_dir: Path) -> None:
        self.sources: dict[str, ExternalSource] = {
            "wandb": wandb_source(config, output_dir),
            "trackio": trackio_source(config, output_dir),
        }

    def table(self, source: str) -> MetricTable:
        return self.sources[source].table()

    def status(self) -> list[dict[str, Any]]:
        return [self.sources[name].status() for name in EXTERNAL_SOURCES]

    def ready(self) -> list[str]:
        return [
            name
            for name in EXTERNAL_SOURCES
            if self.sources[name].status()["status"] == "ready"
        ]

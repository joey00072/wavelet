"""Read-only access to the artifacts one RL run leaves under its output dir."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.dashboard.external import EXTERNAL_SOURCES, ExternalSources
from wavelet.dashboard.jsonl import JsonlCache, read_json
from wavelet.dashboard.metrics import MetricStore, MetricTable
from wavelet.dashboard.rows import (
    EVAL_SORT_KEYS,
    ROLLOUT_SORT_KEYS,
    CompactRowCache,
    RowFilters,
    full_row,
    group_eval_examples,
    group_rollouts,
    histogram,
    numeric_summary,
    sort_rows,
)
from wavelet.monitor import redact
from wavelet.transport.queue import (
    MANIFEST_FILENAME,
    QUEUE_EVENT_FILENAME,
    STABLE_BATCH_MARKER,
    STEP_DIR_PREFIX,
    build_queue_report,
    get_step_dir,
    parse_step,
    resolve_policy_dir,
    resolve_queue_dir,
)
from wavelet.utils.pathing import OUTPUT_RUN_STATE_MARKERS, get_config_dir
from wavelet.utils.serialization import load_yaml

logger = logging.getLogger(__name__)

METRIC_SOURCES = {
    "trainer": "metrics.jsonl",
    "orchestrator": "orchestrator_metrics.jsonl",
    "eval": "eval_metrics.jsonl",
}
HEARTBEAT_STALE_SECONDS = 120.0
_EVAL_KEY = re.compile(r"^eval/(?P<env>.+?)/(?P<metric>.+)$")
_LOG_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.log$")


MAX_EVENT_ROWS = 50_000
_NODE_KEY = re.compile(r"^node/(?P<node>[^/]+)/(?P<metric>.+)$")
_REPLICA_KEY = re.compile(r"^inference/(?P<replica>replica_\d+)/(?P<metric>.+)$")


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / marker).exists() for marker in OUTPUT_RUN_STATE_MARKERS
    )


def discover_runs(roots: list[Path], explicit: list[Path]) -> dict[str, Path]:
    """Map run ids to directories, scanning one level below each root."""
    found: dict[str, Path] = {}

    def add(path: Path) -> None:
        resolved = path.resolve()
        if any(existing == resolved for existing in found.values()):
            return
        run_id = resolved.name
        if run_id in found:
            run_id = f"{resolved.parent.name}--{resolved.name}"
        found[run_id] = resolved

    for path in explicit:
        if path.is_dir():
            add(path)
    for root in roots:
        if not root.is_dir():
            continue
        if is_run_dir(root):
            add(root)
        for child in sorted(root.iterdir()):
            if is_run_dir(child):
                add(child)
    return found


@dataclass(frozen=True, slots=True)
class RunSummary:
    id: str
    path: str


class RunArtifacts:
    """Reads one run directory without claiming, consuming, or rewriting state."""

    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.id = run_id
        self.output_dir = Path(output_dir)
        self._jsonl = JsonlCache(max_rows=MAX_EVENT_ROWS)
        self._metrics = MetricStore()
        self._rows = CompactRowCache()
        self._config: RLConfig | None = None
        self._config_signature: tuple[int, int] | None = None
        self._config_error: str | None = None
        self._external: ExternalSources | None = None

    # ---------------------------------------------------------------- config

    def _resolved_config_path(self) -> Path | None:
        config_dir = get_config_dir(self.output_dir)
        for name in ("rl_orchestrator.yaml", "rl.yaml"):
            candidate = config_dir / name
            if candidate.is_file():
                return candidate
        return None

    def config(self) -> RLConfig:
        path = self._resolved_config_path()
        signature = None
        if path is not None:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        if self._config is not None and signature == self._config_signature:
            return self._config
        config: RLConfig | None = None
        self._config_error = None
        if path is not None:
            try:
                config = RLConfig.model_validate(load_yaml(path))
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                self._config_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Dashboard could not validate %s: %s", path, exc)
        if config is None:
            config = RLConfig(output_dir=self.output_dir)
        self._config = config
        self._config_signature = signature
        return config

    def sanitized_config(self) -> dict[str, Any]:
        path = self._resolved_config_path()
        if path is not None:
            try:
                return redact(load_yaml(path))
            except Exception as exc:  # noqa: BLE001 - safe diagnostic for the UI
                self._config_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Dashboard could not read %s: %s", path, exc)
                return {
                    "_dashboard_error": "The resolved config could not be parsed.",
                    "_error": self._config_error,
                }
        return {"_dashboard_error": "No resolved config artifact was found for this run."}

    # ---------------------------------------------------------------- metrics

    def metric_table(self, source: str) -> MetricTable:
        if source in EXTERNAL_SOURCES:
            return self.external().table(source)
        filename = METRIC_SOURCES.get(source)
        if filename is None:
            raise KeyError(f"Unknown metric source '{source}'.")
        return self._metrics.table(self.output_dir / filename)

    def external(self) -> ExternalSources:
        """External tracker histories (W&B, Trackio) for this run, lazily built."""
        if self._external is None:
            self._external = ExternalSources(self.config(), self.output_dir)
        return self._external

    def external_status(self) -> list[dict[str, Any]]:
        return self.external().status()

    def metric_rows(self, source: str) -> list[dict[str, Any]]:
        """Materialize rows; only for small sources such as ``eval``."""
        table = self.metric_table(source)
        # Background and baseline evaluations finish out of step order.
        indices = table.indices(sort_by_step=source == "eval")
        return [table.row(index) for index in indices]

    def metric_keys(self) -> dict[str, list[dict[str, Any]]]:
        keys = {source: self.metric_table(source).keys() for source in METRIC_SOURCES}
        for source in self.external().ready():
            keys[source] = self.metric_table(source).keys()
        return keys

    def series(
        self,
        source: str,
        keys: list[str],
        *,
        limit: int = 0,
        start: float | None = None,
        end: float | None = None,
        after: float | None = None,
        points: int = 0,
    ) -> dict[str, Any]:
        """Columnar series over a step window.

        ``limit`` keeps the newest rows, ``start``/``end`` select a step range,
        ``after`` returns only rows past a step (incremental polling), and
        ``points`` caps the number of returned points by bucket-averaging with a
        min/max envelope so a 100k-step run costs the same as a 1k-step run.
        """
        table = self.metric_table(source)
        indices = table.indices(
            sort_by_step=source == "eval",
            start=start,
            end=end,
            after=after,
            limit=limit,
        )
        payload = table.series(keys, indices, points=points)
        payload["source"] = source
        payload["total_rows"] = len(table)
        payload["parse_errors"] = table.parse_errors
        return payload

    def latest_rows(self) -> dict[str, dict[str, Any] | None]:
        latest: dict[str, dict[str, Any] | None] = {}
        for source in METRIC_SOURCES:
            table = self.metric_table(source)
            if source == "eval" and len(table):
                indices = table.indices(sort_by_step=True)
                latest[source] = table.row(indices[-1])
            else:
                latest[source] = _merge_same_step_tail(table)
        return latest

    # ---------------------------------------------------------------- nodes

    def nodes(self) -> dict[str, Any]:
        """Latest per-node, per-rank, and per-replica throughput and memory.

        Node history lives in ``node/<name>/*`` trainer metrics; the per-rank
        table comes from the heartbeat, which the trainer overwrites each step.
        """
        trainer = self.metric_table("trainer").last_row or {}
        orchestrator = self.metric_table("orchestrator").last_row or {}
        heartbeat = read_json(self.output_dir / "heartbeat.json") or {}
        nodes: dict[str, dict[str, Any]] = {}
        for key, value in trainer.items():
            match = _NODE_KEY.match(key)
            if match is None or not isinstance(value, int | float):
                continue
            nodes.setdefault(match.group("node"), {"name": match.group("node")})[
                match.group("metric")
            ] = value
        replicas: dict[str, dict[str, Any]] = {}
        for key, value in orchestrator.items():
            match = _REPLICA_KEY.match(key)
            if match is None or not isinstance(value, int | float):
                continue
            replicas.setdefault(
                match.group("replica"), {"name": match.group("replica")}
            )[match.group("metric")] = value
        ranks = heartbeat.get("ranks")
        return {
            "step": _int_or_none(trainer.get("step")),
            "timestamp": trainer.get("timestamp"),
            "nodes": [nodes[name] for name in sorted(nodes)],
            "ranks": ranks if isinstance(ranks, list) else [],
            "replicas": [replicas[name] for name in sorted(replicas)],
            "world": (read_json(self.output_dir / "run_metadata.json") or {}).get(
                "world"
            ),
        }

    # ---------------------------------------------------------------- evals

    def evals(self) -> dict[str, Any]:
        rows = self.metric_rows("eval")
        sets = self.eval_sets()
        envs: dict[str, dict[str, int]] = {}
        history: list[dict[str, Any]] = []
        for row in rows:
            per_env: dict[str, dict[str, float]] = {}
            for key, value in row.items():
                match = _EVAL_KEY.match(key)
                if match is None or not isinstance(value, int | float):
                    continue
                env = match.group("env")
                metric = match.group("metric")
                per_env.setdefault(env, {})[metric] = float(value)
                envs.setdefault(env, {})[metric] = envs.get(env, {}).get(metric, 0) + 1
            history.append(
                {
                    "step": _int_or_none(row.get("step")),
                    "policy_step": _int_or_none(row.get("progress/policy_step")),
                    "timestamp": row.get("timestamp"),
                    "envs": per_env,
                }
            )
        for entry in sets:
            envs.setdefault(str(entry["env"]), {})
        return {
            "envs": [
                {"name": name, "metrics": sorted(metrics)}
                for name, metrics in sorted(envs.items())
            ],
            "history": history,
            "sets": sets,
        }

    def eval_sets(self) -> list[dict[str, Any]]:
        eval_dir = self.output_dir / "evals"
        if not eval_dir.is_dir():
            return []
        sets: list[dict[str, Any]] = []
        for step_dir in sorted(eval_dir.iterdir()):
            step = parse_step(step_dir)
            if step is None or not step_dir.is_dir():
                continue
            for path in sorted(step_dir.glob("*.jsonl")):
                sets.append(
                    {
                        "step": step,
                        "env": path.stem,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                    }
                )
        return sets

    def eval_set_path(self, step: int, env: str) -> Path | None:
        if "/" in env or env in {"", ".", ".."}:
            return None
        path = (
            self.output_dir / "evals" / f"{STEP_DIR_PREFIX}{step:06d}" / f"{env}.jsonl"
        )
        return path if path.is_file() else None

    def eval_rows(
        self,
        step: int,
        env: str,
        *,
        sort: str,
        descending: bool,
        offset: int,
        limit: int,
        filters: RowFilters,
    ) -> dict[str, Any]:
        path = self.eval_set_path(step, env)
        if path is None:
            return _unavailable(f"no eval rollouts for step {step} env {env}")
        rows, scanned, scan_limited = self._rows.rows(path, kind="eval")
        if sort not in EVAL_SORT_KEYS:
            sort = "row_index"
        filtered = [row for row in rows if filters.matches(row)]
        ordered = sort_rows(filtered, key=sort, descending=descending)
        rewards = [r["reward"] for r in rows if r.get("reward") is not None]
        return {
            "available": True,
            "step": step,
            "env": env,
            "path": str(path),
            "total": len(rows),
            "scanned": scanned,
            "scan_limited": scan_limited,
            "filtered": len(filtered),
            "stats": {
                "reward": numeric_summary(rewards),
                "reward_histogram": histogram(rewards, bins=10),
                "errors": sum(1 for r in rows if r.get("has_error")),
                "truncated": sum(1 for r in rows if r.get("is_truncated")),
            },
            "examples": group_eval_examples(rows),
            "rows": ordered[offset : offset + limit],
        }

    def eval_row(self, step: int, env: str, row_index: int) -> dict[str, Any] | None:
        path = self.eval_set_path(step, env)
        return None if path is None else full_row(path, row_index)

    # ---------------------------------------------------------------- rollouts

    def queue_dir(self) -> Path:
        config = self.config()
        return resolve_queue_dir(self.output_dir, config.transport)

    def policy_dir(self) -> Path:
        config = self.config()
        return resolve_policy_dir(self.output_dir, config.policy_transfer)

    def rollout_path(self, step: int) -> Path:
        return (
            get_step_dir(self.queue_dir(), step)
            / self.config().transport.rollout_filename
        )

    def queue(self, *, detail: bool, limit: int) -> dict[str, Any]:
        try:
            return build_queue_report(
                queue_dir=self.queue_dir(),
                policy_dir=self.policy_dir(),
                events_dir=self.output_dir / "events",
                detail=detail,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            return {
                "summary": None,
                "policy": None,
                "rates": None,
                "errors": {"queue_snapshot": f"{type(exc).__name__}: {exc}"},
                "items": [],
            }

    def rollout_batches(self, *, limit: int) -> list[dict[str, Any]]:
        report = self.queue(detail=True, limit=limit)
        batches: list[dict[str, Any]] = []
        for item in report.get("items", []):
            manifest = item.get("manifest") or {}
            batches.append(
                {
                    "queue_step": item.get("queue_step"),
                    "status": item.get("status"),
                    "stable": item.get("stable"),
                    "optimizer_step": manifest.get("optimizer_step"),
                    "chunk_index": manifest.get("chunk_index"),
                    "policy_step": manifest.get("policy_step"),
                    "rows": manifest.get("rows"),
                    "tokens": manifest.get("tokens"),
                    "reward_mean": manifest.get("reward_mean"),
                    "producer_id": manifest.get("producer_id"),
                    "created_at": manifest.get("created_at"),
                    "payload_bytes": manifest.get("payload_bytes"),
                    "age_seconds": item.get("age_seconds"),
                    "claimed_at": (item.get("claim") or {}).get("claimed_at"),
                    "consumed_at": (item.get("consumed") or {}).get("consumed_at"),
                    "parse_errors": item.get("parse_errors") or [],
                }
            )
        batches.sort(key=lambda batch: batch["queue_step"] or 0, reverse=True)
        return batches

    def latest_stable_rollout_step(self) -> int | None:
        queue_dir = self.queue_dir()
        if not queue_dir.is_dir():
            return None
        steps = [
            step
            for candidate in queue_dir.iterdir()
            if (step := parse_step(candidate)) is not None
            and (candidate / STABLE_BATCH_MARKER).exists()
        ]
        return max(steps, default=None)

    def rollout_rows(
        self,
        step: int | None,
        *,
        sort: str,
        descending: bool,
        offset: int,
        limit: int,
        filters: RowFilters,
    ) -> dict[str, Any]:
        if step is None:
            step = self.latest_stable_rollout_step()
            if step is None:
                return _unavailable("no stable rollout batches found")
        path = self.rollout_path(step)
        step_dir = path.parent
        if not path.is_file():
            return _unavailable(f"rollout batch step {step} has no payload", step=step)
        rows, scanned, scan_limited = self._rows.rows(path, kind="rollout")
        if sort not in ROLLOUT_SORT_KEYS:
            sort = "row_index"
        filtered = [row for row in rows if filters.matches(row)]
        ordered = sort_rows(filtered, key=sort, descending=descending)
        rewards = [r["reward"] for r in rows if r.get("reward") is not None]
        advantages = [r["advantage"] for r in rows if r.get("advantage") is not None]
        tokens = [
            float(r["completion_token_count"])
            for r in rows
            if isinstance(r.get("completion_token_count"), int | float)
        ]
        return {
            "available": True,
            "queue_step": step,
            "stable": (step_dir / STABLE_BATCH_MARKER).exists(),
            "path": str(path),
            "manifest": read_json(step_dir / MANIFEST_FILENAME),
            "total": len(rows),
            "scanned": scanned,
            "scan_limited": scan_limited,
            "filtered": len(filtered),
            "stats": {
                "reward": numeric_summary(rewards),
                "advantage": numeric_summary(advantages),
                "completion_tokens": numeric_summary(tokens),
                "reward_histogram": histogram(rewards, bins=10),
                "advantage_histogram": histogram(advantages, bins=20),
                "completion_token_histogram": histogram(tokens, bins=20),
                "truncated": sum(1 for r in rows if r.get("is_truncated")),
                "errors": sum(1 for r in rows if r.get("error")),
                "stop_conditions": _count_values(rows, "stop_condition"),
                "envs": _count_values(rows, "env"),
                "policy_steps": _count_values(rows, "policy_step"),
            },
            "groups": group_rollouts(rows),
            "rows": ordered[offset : offset + limit],
        }

    def rollout_row(self, step: int, row_index: int) -> dict[str, Any] | None:
        path = self.rollout_path(step)
        return None if not path.is_file() else full_row(path, row_index)

    # ---------------------------------------------------------------- lifecycle

    def timeline(self, *, limit: int) -> dict[str, Any]:
        events = self._jsonl.rows(self.output_dir / "events" / QUEUE_EVENT_FILENAME)
        queue_steps: dict[int, dict[str, Any]] = {}
        policies: dict[int, dict[str, Any]] = {}
        for event in events:
            kind = event.get("kind")
            details = event.get("details") or {}
            time = event.get("time")
            queue_step = event.get("queue_step")
            policy_step = event.get("policy_step")
            if kind in {
                "rollout_published",
                "rollout_received",
                "rollout_claimed",
                "rollout_consumed",
            } and isinstance(queue_step, int):
                entry = queue_steps.setdefault(
                    queue_step,
                    {
                        "queue_step": queue_step,
                        "optimizer_step": event.get("optimizer_step"),
                        "policy_step": policy_step,
                    },
                )
                entry[kind.removeprefix("rollout_") + "_at"] = time
                if kind == "rollout_published":
                    entry["payload_bytes"] = details.get("payload_bytes")
                    entry["producer_id"] = event.get("producer_id")
                if kind == "rollout_received":
                    entry["trainer_wait_seconds"] = details.get("wait_seconds")
                    entry["consumer_id"] = event.get("consumer_id")
            elif kind in {
                "policy_export_completed",
                "policy_received",
                "policy_load_completed",
            } and isinstance(policy_step, int):
                entry = policies.setdefault(policy_step, {"policy_step": policy_step})
                key = {
                    "policy_export_completed": "exported_at",
                    "policy_received": "received_at",
                    "policy_load_completed": "loaded_at",
                }[kind]
                entry[key] = time
                if kind == "policy_received":
                    entry["payload_bytes"] = details.get("payload_bytes")
                    entry["inference_wait_seconds"] = details.get("wait_seconds")
                if kind == "policy_load_completed":
                    entry["load_seconds"] = details.get("load_seconds")
        for entry in queue_steps.values():
            entry["publish_to_claim_seconds"] = _seconds_between(
                entry.get("published_at"), entry.get("claimed_at")
            )
            entry["claim_to_consume_seconds"] = _seconds_between(
                entry.get("claimed_at"), entry.get("consumed_at")
            )
        for entry in policies.values():
            entry["export_to_load_seconds"] = _seconds_between(
                entry.get("exported_at"), entry.get("loaded_at")
            )
        ordered_steps = sorted(queue_steps.values(), key=lambda e: e["queue_step"])
        ordered_policies = sorted(policies.values(), key=lambda e: e["policy_step"])
        return {
            "queue_steps": ordered_steps[-limit:],
            "policies": ordered_policies[-limit:],
            "event_count": len(events),
            "parse_errors": self._jsonl.parse_errors(
                self.output_dir / "events" / QUEUE_EVENT_FILENAME
            ),
            "dropped_events": self._jsonl.dropped(
                self.output_dir / "events" / QUEUE_EVENT_FILENAME
            ),
        }

    def run_events(self, *, limit: int) -> list[dict[str, Any]]:
        return self._jsonl.rows(self.output_dir / "events.jsonl")[-limit:]

    def samples(self, *, limit: int) -> list[dict[str, Any]]:
        rows = self._jsonl.rows(self.output_dir / "samples.jsonl")[-limit:]
        return [
            {
                key: ("<omitted>" if key == "input_ids" else value)
                for key, value in row.items()
            }
            for row in rows
        ]

    # ---------------------------------------------------------------- logs

    def log_dir(self) -> Path | None:
        logs_root = self.output_dir / "logs"
        latest = logs_root / "latest"
        if latest.is_dir():
            return latest
        return logs_root if logs_root.is_dir() else None

    def logs(self) -> list[dict[str, Any]]:
        log_dir = self.log_dir()
        if log_dir is None:
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(log_dir.glob("*.log")):
            stat = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, UTC
                    ).isoformat(),
                }
            )
        return entries

    def log_tail(self, name: str, *, lines: int) -> dict[str, Any] | None:
        log_dir = self.log_dir()
        if log_dir is None or not _LOG_NAME.match(name):
            return None
        path = log_dir / name
        if not path.is_file():
            return None
        return {"name": name, "lines": _tail_lines(path, lines)}

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        config = self.config()
        heartbeat = read_json(self.output_dir / "heartbeat.json")
        run_metadata = read_json(self.output_dir / "run_metadata.json")
        latest = self.latest_rows()
        report = self.queue(detail=False, limit=0)
        trainer = latest["trainer"]
        orchestrator = latest["orchestrator"]
        eval_row = latest["eval"]
        trainer_step = _int_or_none((trainer or {}).get("step"))
        has_config = self._resolved_config_path() is not None
        valid_config = has_config and self._config_error is None
        status, status_reason = _derive_status(
            heartbeat, run_metadata, trainer, config if valid_config else None
        )
        envs = self._env_names(config) if valid_config else []
        return {
            "id": self.id,
            "path": str(self.output_dir),
            "status": status,
            "status_reason": status_reason,
            "started_at": (run_metadata or {}).get("started_at"),
            "updated_at": _latest_timestamp(
                heartbeat,
                trainer,
                orchestrator,
                eval_row,
                fallback_paths=(
                    self.output_dir,
                    self.output_dir / "run_metadata.json",
                    self.output_dir / "heartbeat.json",
                    *(self.output_dir / name for name in METRIC_SOURCES.values()),
                    self.output_dir / "rollouts",
                    self.output_dir / "evals",
                    self.output_dir / "policies",
                ),
            ),
            "heartbeat": heartbeat,
            "world": (run_metadata or {}).get("world"),
            "resumed_from": (run_metadata or {}).get("resumed_from"),
            "config_error": self._config_error,
            "has_config": has_config,
            "target_step": config.max_steps if valid_config else None,
            "trainer_step": trainer_step,
            "orchestrator_step": _int_or_none((orchestrator or {}).get("step")),
            "eval_step": _int_or_none((eval_row or {}).get("step")),
            "model": config.model.name if valid_config else None,
            "algo": config.algo.type if valid_config else None,
            "loss": config.loss.type if valid_config else None,
            "launcher_mode": config.launcher.mode if valid_config else None,
            "envs": envs,
            "lora": config.lora is not None if valid_config else None,
            "batch": {
                "examples_per_step": config.orchestrator.examples_per_step,
                "rollouts_per_example": config.orchestrator.rollouts_per_example,
                "token_batch_size": config.orchestrator.token_batch_size,
                "max_async_level": config.orchestrator.max_async_level,
                "max_off_policy_steps": config.orchestrator.max_off_policy_steps,
            }
            if valid_config
            else None,
            "latest": latest,
            "queue_summary": report.get("summary"),
            "queue_rates": report.get("rates"),
            "policy": report.get("policy"),
            "queue_errors": report.get("errors"),
            "eval_envs": sorted(
                {
                    match.group("env")
                    for key in (eval_row or {})
                    if (match := _EVAL_KEY.match(key)) is not None
                }
                | {str(entry["env"]) for entry in self.eval_sets()}
            ),
            "logs": self.logs(),
        }

    @staticmethod
    def _env_names(config: RLConfig) -> list[str]:
        names: list[str] = []
        if config.orchestrator.verifier_env_id:
            names.append(config.orchestrator.verifier_env_id)
        for env in config.orchestrator.envs or []:
            name = getattr(env, "name", None) or getattr(env, "id", None)
            if name and name not in names:
                names.append(str(name))
        return names


def _step_of(row: dict[str, Any], source: str) -> int | float | None:
    value = row.get("step")
    if value is None and source == "trainer":
        value = row.get("progress/step")
    return _finite(value)


def _finite(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    finite = _finite(value)
    return None if finite is None else int(finite)


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        label = "none" if value is None else str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _seconds_between(start: str | None, end: str | None) -> float | None:
    start_time = _timestamp(start)
    end_time = _timestamp(end)
    if start_time is None or end_time is None:
        return None
    delta = end_time - start_time
    return delta.total_seconds()


def _latest_timestamp(
    *rows: dict[str, Any] | None, fallback_paths: tuple[Path, ...] = ()
) -> str | None:
    stamps = [
        parsed
        for row in rows
        if row
        for parsed in [_timestamp(row.get("timestamp"))]
        if parsed is not None
    ]
    if stamps:
        return max(stamps).isoformat()
    modified = [
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        for path in fallback_paths
        if path.exists()
    ]
    return max(modified).isoformat() if modified else None


def _derive_status(
    heartbeat: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None,
    trainer: dict[str, Any] | None,
    config: RLConfig | None,
) -> tuple[str, str]:
    if heartbeat is None:
        if run_metadata is None and trainer is None:
            return "unknown", "no heartbeat or run metadata found"
        return "unknown", "no heartbeat file"
    status = str(heartbeat.get("status") or "unknown")
    stamp = heartbeat.get("timestamp")
    if status == "running" and isinstance(stamp, str):
        parsed = _timestamp(stamp)
        age = None if parsed is None else (datetime.now(UTC) - parsed).total_seconds()
        if age is not None and age > HEARTBEAT_STALE_SECONDS:
            return "stale", f"heartbeat is {int(age)}s old"
        return status, "heartbeat fresh"
    if status == "completed":
        step = heartbeat.get("step")
        target = f" of {config.max_steps}" if config is not None else ""
        return status, f"finished at step {step}{target}"
    return status, str(heartbeat.get("reason") or "")


def _tail_lines(path: Path, lines: int) -> list[str]:
    chunk = 64 * 1024
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        data = b""
        while size > 0 and data.count(b"\n") <= lines:
            read = min(chunk, size)
            size -= read
            handle.seek(size)
            data = handle.read(read) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def _unavailable(reason: str, *, step: int | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "queue_step": step,
        "rows": [],
        "groups": [],
        "total": 0,
        "filtered": 0,
    }


def _merge_same_step_tail(table: Any) -> dict[str, Any] | None:
    """Merge the trailing rows that share the newest step into one row.

    Trainers log several rows per optimizer step (loss row, telemetry row), so
    the newest row alone is missing most keys. Later rows win on conflicts.
    """
    count = len(table)
    if count == 0:
        return None
    last = table.row(count - 1)
    step = last.get("step")
    merged: dict[str, Any] = {}
    index = count - 1
    while index >= 0:
        row = table.row(index)
        if row.get("step") != step:
            break
        merged = {**row, **merged}
        index -= 1
    return merged or last

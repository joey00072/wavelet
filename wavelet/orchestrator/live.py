"""Context-local, bounded tracing hooks for live verifier episodes."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import math
import os
import socket
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Any, Self

from wavelet.orchestrator.eval_utils import atomic_json, canonical_json

_OWNER: contextvars.ContextVar[LiveEpisode | None] = contextvars.ContextVar(
    "wavelet_live_episode", default=None
)
_MARK = "_wavelet_live_trace_hook"
_METHODS = (
    "run_rollout",
    "rollout",
    "setup_state",
    "get_model_response",
    "add_model_response",
    "env_response",
    "cleanup",
    "_cleanup",
)
_RUBRIC_METHODS = ("score_rollout", "dummy_score_rollout", "score_group", "cleanup")
_MAX_EVENTS = 128
_MAX_TEXT = 8192
_MAX_ITEMS = 128
_MAX_BYTES = 256 * 1024


def _safe(value: Any) -> Any:
    remaining = 512

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 6:
            return "<truncated>"
        if isinstance(item, str):
            return item[:_MAX_TEXT] + ("…<truncated>" if len(item) > _MAX_TEXT else "")
        if item is None or isinstance(item, (int, bool)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else str(item)
        if isinstance(item, BaseException):
            return {"type": type(item).__name__, "message": visit(str(item), depth + 1)}
        if isinstance(item, dict):
            return {
                str(k)[:256]: visit(v, depth + 1)
                for k, v in islice(item.items(), min(_MAX_ITEMS, max(remaining, 0)))
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(v, depth + 1)
                for v in islice(item, min(_MAX_ITEMS, max(remaining, 0)))
            ]
        names = getattr(type(item), "model_fields", None)
        if names is None and is_dataclass(item) and not isinstance(item, type):
            names = (field.name for field in fields(item))
        if names is not None:
            return {
                name: visit(getattr(item, name), depth + 1)
                for name in islice(names, min(_MAX_ITEMS, max(remaining, 0)))
            }
        return f"<{type(item).__name__}>"

    try:
        return visit(value)
    except Exception:  # noqa: BLE001 - instrumentation must never alter outcomes.
        return "<unavailable>"


def _payload(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if name == "get_model_response":
        out["prompt"] = _safe(
            kwargs.get(
                "prompt", args[1] if len(args) > 1 else args[0] if args else None
            )
        )
    elif name == "env_response":
        out["messages"] = _safe(kwargs.get("messages", args[0] if args else None))
    if isinstance(result, dict) and "message" in result:
        out["response"] = _safe(result["message"])
    elif name in {"get_model_response", "env_response"} and result is not None:
        out["response"] = _safe(result)
    return out


class LiveEpisode:
    def __init__(
        self,
        output_dir: Path,
        env_name: str,
        kind: str,
        step: int | None = None,
        policy_step: int | None = None,
        example_id: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        self.path = Path(output_dir) / "traces" / "live" / f"{uuid.uuid4()}.json"
        self.data: dict[str, Any] = {
            "format_version": 1,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_start": _process_start(os.getpid()),
            "parent_id": parent_id,
            "id": self.path.stem,
            "status": "running",
            "env": env_name,
            "kind": kind,
            "step": step,
            "policy_step": policy_step,
            "example_id": example_id,
            "started_at": datetime.now(UTC).isoformat(),
            "events": [],
        }
        self._token: contextvars.Token[LiveEpisode | None] | None = None
        self._write_lock = asyncio.Lock()

    def __enter__(self) -> Self:
        self._token = _OWNER.set(self)
        self.event("episode_started")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        try:
            if exc is not None:
                status = (
                    "cancelled"
                    if exc_type is not None and exc_type.__name__ == "CancelledError"
                    else "error"
                )
                self.finish(status, error=exc)
            elif self.data["status"] == "running":
                self.finish("completed")
        finally:
            if self._token is not None:
                _OWNER.reset(self._token)

    async def __aenter__(self) -> Self:
        self._token = _OWNER.set(self)
        try:
            await self.aevent("episode_started")
        except asyncio.CancelledError as exc:
            try:
                await self.afinish("cancelled", error=exc)
            finally:
                _OWNER.reset(self._token)
            raise
        except BaseException:
            _OWNER.reset(self._token)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        try:
            if exc is not None:
                status = (
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                await self.afinish(status, error=exc)
            elif self.data["status"] == "running":
                await self.afinish("completed")
        finally:
            if self._token is not None:
                _OWNER.reset(self._token)

    async def _write_async(
        self, method: Callable[..., None], *args: Any, **kwargs: Any
    ) -> None:
        async with self._write_lock:
            task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # A running filesystem write cannot be cancelled. Finish it before
                # publishing terminal state to the same atomic temporary path.
                await task
                raise

    async def aevent(self, phase: str, **fields: Any) -> None:
        await self._write_async(self.event, phase, **fields)

    async def afinish(self, status: str = "completed", **fields: Any) -> None:
        await self._write_async(self.finish, status, **fields)

    def finish(self, status: str = "completed", **fields: Any) -> None:
        self.data["status"] = status
        self.data.update({key: _safe(value) for key, value in fields.items()})
        self.data["finished_at"] = datetime.now(UTC).isoformat()
        self._publish()

    def event(self, phase: str, **fields: Any) -> None:
        events = self.data["events"]
        events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "phase": phase,
                **{k: _safe(v) for k, v in fields.items()},
            }
        )
        del events[:-_MAX_EVENTS]
        self.data["phase"] = phase
        self._publish()

    def _publish(self) -> None:
        try:
            encoded = canonical_json(self.data)
            if len(encoded.encode()) > _MAX_BYTES:
                self.data["truncated"] = True
                self.data["events"] = self.data["events"][-16:]
                encoded = canonical_json(self.data)
            if len(encoded.encode()) > _MAX_BYTES:
                # Keep lifecycle metadata even when a verifier returns an
                # unexpectedly large object.
                self.data["events"] = []
                self.data["payload_truncated"] = True
            atomic_json(self.path, self.data)
            atomic_json(
                self.path.with_suffix(".meta.json"),
                {
                    key: self.data.get(key)
                    for key in (
                        "format_version",
                        "pid",
                        "hostname",
                        "process_start",
                        "parent_id",
                        "id",
                        "env",
                        "kind",
                        "status",
                        "phase",
                        "step",
                        "policy_step",
                        "example_id",
                        "started_at",
                        "finished_at",
                    )
                }
                | {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "revision": str(time.time_ns()),
                },
            )
            if self.data["status"] != "running":
                self._prune_terminal()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Live trace publication failed: %s", exc
            )

    def _prune_terminal(self) -> None:
        terminal: list[tuple[str, Path]] = []
        for path in islice(self.path.parent.glob("*.meta.json"), 2048):
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as handle:
                    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                        continue
                    raw = handle.read(16385)
                if len(raw) > 16384:
                    continue
                data = json.loads(raw)
                if not isinstance(data, dict):
                    continue
                if data.get("status") != "running" or episode_abandoned(data):
                    terminal.append((str(data.get("updated_at", "")), path))
            except (OSError, ValueError, RecursionError):
                continue
        terminal.sort(key=lambda item: item[0])
        for _, path in terminal[:-256]:
            try:
                path.with_name(path.name.removesuffix(".meta.json") + ".json").unlink(
                    missing_ok=True
                )
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def episode_abandoned(data: dict[str, Any]) -> bool:
    """Detect a dead local writer without expiring legitimately slow requests."""
    pid = data.get("pid")
    if data.get("hostname") != socket.gethostname() or not isinstance(pid, int):
        return False
    start = data.get("process_start")
    return start is not None and _process_start(pid) != start


def live_episode(
    output_dir: Path, env_name: str, kind: str, **kwargs: Any
) -> LiveEpisode:
    return LiveEpisode(output_dir, env_name, kind, **kwargs)


def _wrap(obj: Any, name: str) -> None:
    original = getattr(obj, name, None)
    if original is None or not callable(original) or getattr(original, _MARK, False):
        return
    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            owner = _OWNER.get()
            if (
                name in {"run_rollout", "rollout"}
                and owner is not None
                and owner.data["kind"] == "group"
            ):
                async with live_episode(
                    owner.path.parents[2],
                    owner.data["env"],
                    "rollout",
                    step=owner.data["step"],
                    policy_step=owner.data["policy_step"],
                    example_id=owner.data["example_id"],
                    parent_id=owner.data["id"],
                ) as child:
                    result = await original(*args, **kwargs)
                    if isinstance(result, dict) and result.get("error") is not None:
                        await child.afinish("error", error=result["error"])
                    return result
            if owner is not None:
                await owner.aevent(name + ".started", **_payload(name, args, kwargs))
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                if owner is not None:
                    await owner.aevent(name + ".error", error=exc)
                raise
            if owner is not None:
                await owner.aevent(
                    name + ".finished", **_payload(name, args, kwargs, result)
                )
            return result
    else:

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            owner = _OWNER.get()
            if owner is not None:
                owner.event(name + ".started", **_payload(name, args, kwargs))
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                if owner is not None:
                    owner.event(name + ".error", error=exc)
                raise
            if owner is not None:
                owner.event(name + ".finished", **_payload(name, args, kwargs, result))
            return result

    setattr(wrapped, _MARK, True)
    setattr(obj, name, wrapped)


def install_verifier_trace_hooks(env: Any) -> Any:
    for name in _METHODS:
        _wrap(env, name)
    rubric = getattr(env, "rubric", None)
    if rubric is not None:
        for name in _RUBRIC_METHODS:
            _wrap(rubric, name)
    return env

"""Backend-independent HTTP admin client and endpoint resolution."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wavelet.configs.config import RLConfig
from wavelet.contracts.queue_records import PolicySnapshot
from wavelet.transport.weights.handshake import NCCL_READY_MARKER

ADMIN_REQUEST_ATTEMPTS = 3
ADMIN_RETRY_BACKOFF_SECONDS = 1.0
ADMIN_CONTROL_TIMEOUT_SECONDS = 300.0
POLICY_LOAD_TIMEOUT_SECONDS = 720.0


class RetryableHTTPError(RuntimeError):
    """Transient HTTP response that an idempotent admin call may retry."""


def http_base_urls(config: RLConfig, count: int | None = None) -> list[str]:
    """Resolve unique inference HTTP endpoints from the inference config."""
    if count is None:
        count = len(config.inference.http.ports or []) or len(
            config.inference.http.hosts or []
        )
        count = count or 1
    hosts = config.inference.http.hosts
    if hosts is None:
        hosts = [config.inference.http.host] * count
    elif len(hosts) != count:
        raise ValueError(
            "inference.http.hosts must have exactly "
            f"{count} entries when launcher.inference_num_replicas={count}."
        )
    ports = config.inference.http.ports
    if ports is None:
        ports = [config.inference.http.port + offset for offset in range(count)]
    elif len(ports) != count:
        raise ValueError(
            "inference.http.ports must have exactly "
            f"{count} entries when launcher.inference_num_replicas={count}."
        )
    endpoints = list(zip(hosts, ports, strict=True))
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("inference HTTP host/port pairs must be unique.")
    return [f"http://{host}:{port}" for host, port in endpoints]


class HTTPEngineAdmin:
    """HTTP control-plane client for one inference replica."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 30.0,
        policy_load_timeout_seconds: float = POLICY_LOAD_TIMEOUT_SECONDS,
        request: Callable[[str, str, dict[str, Any] | None], dict[str, Any]]
        | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.policy_load_timeout_seconds = policy_load_timeout_seconds
        self._request_override = request

    def health(self) -> None:
        self._request("GET", "/health")

    def pause(self) -> None:
        self._request_with_retries("POST", "/pause")

    def resume(self) -> None:
        self._request_with_retries("POST", "/resume")

    def load_policy(
        self,
        snapshot: PolicySnapshot | Path,
        *,
        adapter_name: str | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(snapshot, PolicySnapshot):
            step = snapshot.step
            policy_dir = snapshot.step_dir
        else:
            policy_dir = Path(snapshot)
            if step is None:
                step = self._step_from_dir(policy_dir)
        payload: dict[str, Any] = {"policy_dir": str(policy_dir), "step": step}
        if adapter_name is not None:
            payload.update({"adapter_name": adapter_name, "load_inplace": True})
        return self._request_with_retries("POST", "/load_policy", payload)

    def init_weight_receiver(self, init_info: dict[str, Any]) -> None:
        self._request("POST", "/init_broadcaster", init_info)

    def sleep(self, *, level: int = 1) -> None:
        self._request("POST", "/sleep", {"level": level})

    def wake(self, *, tags: list[str] | None = None) -> None:
        self._request("POST", "/wake", {} if tags is None else {"tags": tags})

    def mark_nccl_ready(self, snapshot: PolicySnapshot | Path) -> None:
        directory = (
            snapshot.step_dir if isinstance(snapshot, PolicySnapshot) else snapshot
        )
        (Path(directory) / NCCL_READY_MARKER).touch()

    def _request_with_retries(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(1, ADMIN_REQUEST_ATTEMPTS + 1):
            try:
                return self._request(method, path, payload)
            except (RetryableHTTPError, OSError):
                if attempt == ADMIN_REQUEST_ATTEMPTS:
                    raise
                time.sleep(ADMIN_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._request_override is not None:
            return self._request_override(method, path, payload)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        timeout = (
            self.policy_load_timeout_seconds
            if path == "/load_policy"
            else (
                ADMIN_CONTROL_TIMEOUT_SECONDS
                if path in {"/pause", "/resume"}
                else self.request_timeout_seconds
            )
        )
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error_type = RetryableHTTPError if exc.code >= 500 else RuntimeError
            raise error_type(
                f"Inference HTTP server returned {exc.code} for {path}: {detail}"
            ) from exc
        return {} if not raw else json.loads(raw.decode("utf-8"))

    @staticmethod
    def _step_from_dir(path: Path) -> int:
        try:
            return int(path.name.removeprefix("step-"))
        except ValueError as exc:
            raise ValueError(f"Policy path does not contain a step: {path}") from exc

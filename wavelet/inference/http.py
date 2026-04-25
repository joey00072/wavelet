from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference.policy import PolicyInferenceEngine
from wavelet.inference.serialization import (
    rl_examples_from_payload,
    rl_examples_to_payload,
)


class HTTPPolicyInferenceEngine(PolicyInferenceEngine):
    """HTTP client for a persistent Wavelet vLLM rollout server."""

    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.base_url = (
            f"http://{config.inference.http.host}:{config.inference.http.port}"
        )

    def setup(self) -> None:
        deadline = time.monotonic() + self.config.inference.http.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._request("GET", "/health")
                return
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            f"Timed out waiting for vLLM HTTP server at {self.base_url}."
        ) from last_error

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        payload = {"policy_dir": str(policy_dir), "step": step}
        self._request("POST", "/load_policy", payload)
        self.policy_step = step

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        response = self._request(
            "POST",
            "/annotate",
            {"records": rl_examples_to_payload(records)},
        )
        self.policy_step = response.get("policy_step")
        return rl_examples_from_payload(response["records"])

    def close(self) -> None:
        return None

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.inference.http.request_timeout_seconds,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP server returned {exc.code} for {path}: {detail}"
            ) from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

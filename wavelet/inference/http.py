from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
        ports = config.inference.http.ports or [config.inference.http.port]
        self.base_urls = [
            f"http://{config.inference.http.host}:{port}"
            for port in ports
        ]
        self.base_url = self.base_urls[0]

    def setup(self) -> None:
        deadline = time.monotonic() + self.config.inference.http.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                for base_url in self.base_urls:
                    self._request("GET", "/health", base_url=base_url)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            "Timed out waiting for vLLM HTTP server(s) at "
            f"{', '.join(self.base_urls)}."
        ) from last_error

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        payload = {"policy_dir": str(policy_dir), "step": step}
        if len(self.base_urls) == 1:
            self._request("POST", "/load_policy", payload, base_url=self.base_url)
        else:
            with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
                futures = [
                    executor.submit(
                        self._request,
                        "POST",
                        "/load_policy",
                        payload,
                        base_url=base_url,
                    )
                    for base_url in self.base_urls
                ]
                for future in futures:
                    future.result()
        self.policy_step = step

    def sleep(self) -> None:
        self._request_all("POST", "/sleep", {"level": 1})

    def wake(self, *, tags: list[str] | None = None) -> None:
        payload: dict[str, Any] = {}
        if tags is not None:
            payload["tags"] = tags
        self._request_all("POST", "/wake", payload)

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        if len(self.base_urls) == 1 or len(records) <= 1:
            response = self._request(
                "POST",
                "/annotate",
                {"records": rl_examples_to_payload(records)},
            )
            self.policy_step = response.get("policy_step")
            return rl_examples_from_payload(response["records"])

        chunks: list[tuple[int, list[RLExample]]] = [
            (index, []) for index in range(len(self.base_urls))
        ]
        for index, record in enumerate(records):
            chunks[index % len(chunks)][1].append(record)

        annotated_by_chunk: dict[int, list[RLExample]] = {}
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = {
                executor.submit(
                    self._request,
                    "POST",
                    "/annotate",
                    {"records": rl_examples_to_payload(chunk)},
                    base_url=self.base_urls[index],
                ): index
                for index, chunk in chunks
                if chunk
            }
            for future, index in futures.items():
                response = future.result()
                self.policy_step = response.get("policy_step")
                annotated_by_chunk[index] = rl_examples_from_payload(
                    response["records"]
                )

        positions = {index: 0 for index in annotated_by_chunk}
        merged: list[RLExample] = []
        for index in range(len(records)):
            chunk_index = index % len(self.base_urls)
            position = positions[chunk_index]
            merged.append(annotated_by_chunk[chunk_index][position])
            positions[chunk_index] = position + 1
        return merged

    def _annotate_single_server(self, records: list[RLExample]) -> list[RLExample]:
        response = self._request(
            "POST",
            "/annotate",
            {"records": rl_examples_to_payload(records)},
        )
        self.policy_step = response.get("policy_step")
        return rl_examples_from_payload(response["records"])

    def close(self) -> None:
        return None

    def _request_all(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if len(self.base_urls) == 1:
            self._request(method, path, payload, base_url=self.base_url)
            return
        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as executor:
            futures = [
                executor.submit(
                    self._request,
                    method,
                    path,
                    payload,
                    base_url=base_url,
                )
                for base_url in self.base_urls
            ]
            for future in futures:
                future.result()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{base_url or self.base_url}{path}",
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

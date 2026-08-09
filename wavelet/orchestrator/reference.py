"""Clients for algorithm-owned frozen reference model endpoints."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from wavelet.configs.rl_config import FrozenModelConfig


class ReferenceScorer(Protocol):
    """Return one causal prefill log-probability per supplied token."""

    def score(self, token_ids: list[int]) -> list[float]: ...


class VLLMReferenceScorer:
    """Score token IDs through vLLM's token-in/token-out generate endpoint."""

    def __init__(self, config: FrozenModelConfig) -> None:
        urls = (
            config.base_url if isinstance(config.base_url, list) else [config.base_url]
        )
        if not urls:
            raise ValueError("A frozen model requires at least one base_url.")
        self.base_urls = [_server_root(url) for url in urls]
        self.model = config.name
        self.timeout_seconds = config.timeout_seconds
        self.api_key = os.environ.get(config.api_key_var)
        self._next_url = 0
        self._lock = threading.Lock()

    def score(self, token_ids: list[int]) -> list[float]:
        if not token_ids:
            return []
        base_url = self._select_url()
        payload = {
            "model": self.model,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 1,
            },
        }
        response = self._request(base_url, payload)
        return extract_prompt_logprobs(response, token_ids=token_ids)

    def _select_url(self) -> str:
        with self._lock:
            url = self.base_urls[self._next_url % len(self.base_urls)]
            self._next_url += 1
        return url

    def _request(self, base_url: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{base_url}/inference/v1/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Frozen reference server returned "
                f"{exc.code} for /inference/v1/generate: {detail}"
            ) from exc
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Frozen reference server response must be an object.")
        return payload


def extract_prompt_logprobs(
    response: dict[str, object],
    *,
    token_ids: Sequence[int],
) -> list[float]:
    """Flatten vLLM prompt-logprob rows while preserving token alignment."""
    raw_rows = response.get("prompt_logprobs")
    if not isinstance(raw_rows, list):
        raise TypeError("Frozen reference response did not include prompt_logprobs.")
    if len(raw_rows) != len(token_ids):
        raise ValueError(
            "Frozen reference prompt_logprobs must align with token_ids "
            f"({len(raw_rows)} != {len(token_ids)})."
        )

    values: list[float] = []
    for index, (token_id, row) in enumerate(zip(token_ids, raw_rows, strict=True)):
        if row is None:
            if index != 0:
                raise ValueError(
                    "Frozen reference response returned a null logprob for "
                    f"token {token_id} at position {index}."
                )
            values.append(0.0)
            continue
        if not isinstance(row, dict) or not row:
            raise TypeError("Each prompt_logprobs row must be an object or null.")
        entry = row.get(str(token_id), row.get(token_id))
        if isinstance(entry, dict):
            entry = entry.get("logprob")
        elif hasattr(entry, "logprob"):
            entry = entry.logprob
        if entry is None:
            raise ValueError(
                f"Frozen reference response omitted logprob for token {token_id}."
            )
        values.append(float(entry))
    return values


def _server_root(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    normalized = normalized.removesuffix("/v1")
    if not normalized:
        raise ValueError("Frozen model base_url must not be empty.")
    return normalized

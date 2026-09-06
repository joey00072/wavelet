from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import urllib.request
from dataclasses import dataclass

from wavelet.orchestrator.concurrency import EngineLoadSample

logger = logging.getLogger(__name__)

_KV_USAGE_NAMES = {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}
_RUNNING_NAMES = {"vllm:num_requests_running"}
_WAITING_NAMES = {"vllm:num_requests_waiting"}
_WAITING_BY_REASON_NAME = "vllm:num_requests_waiting_by_reason"
_CACHE_CONFIG_NAME = "vllm:cache_config_info"
_PREEMPTION_NAMES = {"vllm:num_preemptions", "vllm:num_preemptions_total"}
_GENERATION_TOKEN_NAMES = {"vllm:generation_tokens", "vllm:generation_tokens_total"}
_PROMPT_TOKEN_NAMES = {"vllm:prompt_tokens", "vllm:prompt_tokens_total"}


@dataclass(frozen=True, slots=True)
class ParsedVLLMMetrics:
    recognized: bool = False
    kv_cache_usage: float = 0.0
    running: int = 0
    waiting: int = 0
    waiting_capacity: int | None = None
    kv_capacity_tokens: int | None = None
    preemptions_total: int = 0
    generation_tokens_total: float = 0.0
    prompt_tokens_total: float = 0.0
    token_counters: bool = False


def parse_vllm_metrics(text: str) -> ParsedVLLMMetrics:
    """Parse the small Prometheus subset used by adaptive concurrency."""
    values: dict[str, list[float]] = {}
    waiting_capacity: list[float] = []
    cache_capacities: list[int] = []
    supported = (
        _KV_USAGE_NAMES
        | _RUNNING_NAMES
        | _WAITING_NAMES
        | _PREEMPTION_NAMES
        | _GENERATION_TOKEN_NAMES
        | _PROMPT_TOKEN_NAMES
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric = parts[0]
        name = metric.split("{", 1)[0]
        labels = _prometheus_labels(metric)
        if name == _CACHE_CONFIG_NAME:
            capacity = _cache_capacity_tokens(labels)
            if capacity is not None:
                cache_capacities.append(capacity)
            continue
        if name == _WAITING_BY_REASON_NAME and labels.get("reason") == "capacity":
            value = _finite_float(parts[1])
            if value is not None:
                waiting_capacity.append(value)
            continue
        if name not in supported:
            continue
        value = _finite_float(parts[1])
        if value is not None:
            values.setdefault(name, []).append(value)

    def summed(names: set[str]) -> float:
        return sum(sum(values.get(name, ())) for name in names)

    usage_values = [value for name in _KV_USAGE_NAMES for value in values.get(name, ())]
    return ParsedVLLMMetrics(
        recognized=bool(values or waiting_capacity or cache_capacities),
        kv_cache_usage=max(usage_values, default=0.0),
        running=int(summed(_RUNNING_NAMES)),
        waiting=int(summed(_WAITING_NAMES)),
        waiting_capacity=(int(sum(waiting_capacity)) if waiting_capacity else None),
        kv_capacity_tokens=sum(cache_capacities) or None,
        preemptions_total=int(
            max((summed({name}) for name in _PREEMPTION_NAMES), default=0.0)
        ),
        generation_tokens_total=max(
            (summed({name}) for name in _GENERATION_TOKEN_NAMES), default=0.0
        ),
        prompt_tokens_total=max(
            (summed({name}) for name in _PROMPT_TOKEN_NAMES), default=0.0
        ),
        token_counters=any(
            name in values for name in _GENERATION_TOKEN_NAMES | _PROMPT_TOKEN_NAMES
        ),
    )


def _finite_float(text: str) -> float | None:
    """Parse a Prometheus sample value, rejecting unparsable or non-finite text."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _prometheus_labels(metric: str) -> dict[str, str]:
    """Read string labels from one Prometheus sample name."""
    if "{" not in metric:
        return {}
    labels_text = metric.split("{", 1)[1].rsplit("}", 1)[0]
    return {
        key: value.replace(r"\"", '"').replace(r"\\", "\\")
        for key, value in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', labels_text)
    }


def _cache_capacity_tokens(labels: dict[str, str]) -> int | None:
    size_tokens = labels.get("kv_cache_size_tokens", "")
    if size_tokens.isdigit():
        return int(size_tokens)
    blocks = labels.get("num_gpu_blocks", "")
    block_size = labels.get("block_size", "")
    if blocks.isdigit() and block_size.isdigit():
        return int(blocks) * int(block_size)
    return None


class InferenceMetricsScraper:
    """Best-effort vLLM metrics scraper with per-replica observations."""

    def __init__(self, base_urls: list[str], *, timeout_seconds: float) -> None:
        self.base_urls = [url.rstrip("/").removesuffix("/v1") for url in base_urls]
        self.timeout_seconds = timeout_seconds
        self.previous_preemptions: dict[str, int] = {}
        self.previous_tokens: dict[str, tuple[float, float, float]] = {}
        self.max_model_len_by_url: dict[str, int] = {}
        self.latest_metrics: dict[str, float] = {}
        self._clock = time.monotonic

    async def scrape(self) -> list[EngineLoadSample]:
        results = await asyncio.gather(
            *(asyncio.to_thread(self._fetch, base_url) for base_url in self.base_urls),
            return_exceptions=True,
        )
        samples: list[EngineLoadSample] = []
        metrics: dict[str, float] = {}
        for index, (base_url, result) in enumerate(
            zip(self.base_urls, results, strict=True)
        ):
            replica = f"replica_{index}"
            prefix = f"inference/{replica}"
            if isinstance(result, BaseException):
                metrics[f"{prefix}/scrape_success"] = 0.0
                logger.debug(
                    "Inference metrics scrape failed for %s: %r", base_url, result
                )
                continue
            parsed = parse_vllm_metrics(result)
            if not parsed.recognized:
                metrics[f"{prefix}/scrape_success"] = 0.0
                logger.debug(
                    "Inference metrics endpoint %s exposed no supported vLLM metrics.",
                    base_url,
                )
                continue
            previous = self.previous_preemptions.get(replica)
            preemptions_delta = (
                0 if previous is None else max(parsed.preemptions_total - previous, 0)
            )
            self.previous_preemptions[replica] = parsed.preemptions_total
            token_rates = (
                self._token_rates(replica, parsed) if parsed.token_counters else {}
            )
            if (
                parsed.kv_capacity_tokens is not None
                and base_url not in self.max_model_len_by_url
            ):
                try:
                    self.max_model_len_by_url[base_url] = await asyncio.to_thread(
                        self._fetch_model_max_len,
                        base_url,
                    )
                except (OSError, TypeError, ValueError) as error:
                    logger.debug(
                        "Inference model metadata fetch failed for %s: %r",
                        base_url,
                        error,
                    )
            samples.append(
                EngineLoadSample(
                    replica=replica,
                    kv_cache_usage=parsed.kv_cache_usage,
                    running=parsed.running,
                    waiting=parsed.waiting,
                    preemptions_delta=preemptions_delta,
                    kv_capacity_tokens=parsed.kv_capacity_tokens,
                    max_model_len=self.max_model_len_by_url.get(base_url),
                    waiting_capacity=parsed.waiting_capacity,
                )
            )
            metrics.update(
                {
                    f"{prefix}/scrape_success": 1.0,
                    f"{prefix}/kv_cache_usage": parsed.kv_cache_usage,
                    f"{prefix}/requests_running": float(parsed.running),
                    f"{prefix}/requests_waiting": float(parsed.waiting),
                    f"{prefix}/preemptions_total": float(parsed.preemptions_total),
                    f"{prefix}/preemptions_delta": float(preemptions_delta),
                }
            )
            if parsed.token_counters:
                metrics[f"{prefix}/generation_tokens_total"] = (
                    parsed.generation_tokens_total
                )
                metrics[f"{prefix}/prompt_tokens_total"] = parsed.prompt_tokens_total
                metrics.update(
                    {f"{prefix}/{key}": value for key, value in token_rates.items()}
                )
            if parsed.kv_capacity_tokens is not None:
                metrics[f"{prefix}/kv_capacity_tokens"] = float(
                    parsed.kv_capacity_tokens
                )
            if max_model_len := self.max_model_len_by_url.get(base_url):
                metrics[f"{prefix}/max_model_len"] = float(max_model_len)
        self.latest_metrics = metrics
        return samples if len(samples) == len(self.base_urls) else []

    def _token_rates(self, replica: str, parsed: ParsedVLLMMetrics) -> dict[str, float]:
        """Tokens per second per replica from the cumulative vLLM counters."""
        now = self._clock()
        previous = self.previous_tokens.get(replica)
        self.previous_tokens[replica] = (
            now,
            parsed.generation_tokens_total,
            parsed.prompt_tokens_total,
        )
        if previous is None:
            return {}
        then, generation, prompt = previous
        elapsed = now - then
        if elapsed <= 0:
            return {}
        return {
            "generation_tokens_per_second": max(
                parsed.generation_tokens_total - generation, 0.0
            )
            / elapsed,
            "prompt_tokens_per_second": max(parsed.prompt_tokens_total - prompt, 0.0)
            / elapsed,
        }

    def _fetch(self, base_url: str) -> str:
        request = urllib.request.Request(
            f"{base_url}/metrics",
            headers={"Accept": "text/plain"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

    def _fetch_model_max_len(self, base_url: str) -> int:
        request = urllib.request.Request(
            f"{base_url}/v1/models",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Inference model metadata must be a JSON object.")
        lengths = [
            model.get("max_model_len")
            for model in payload.get("data", [])
            if isinstance(model, dict)
        ]
        lengths = [value for value in lengths if isinstance(value, int) and value > 0]
        if not lengths:
            raise ValueError("Inference model metadata omitted max_model_len.")
        return max(lengths)

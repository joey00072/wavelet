from __future__ import annotations

import asyncio
import logging
import math
import urllib.request
from dataclasses import dataclass

from wavelet.orchestrator.concurrency import EngineLoadSample

logger = logging.getLogger(__name__)

_KV_USAGE_NAMES = {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}
_RUNNING_NAMES = {"vllm:num_requests_running"}
_WAITING_NAMES = {"vllm:num_requests_waiting"}
_PREEMPTION_NAMES = {"vllm:num_preemptions", "vllm:num_preemptions_total"}


@dataclass(frozen=True, slots=True)
class ParsedVLLMMetrics:
    recognized: bool = False
    kv_cache_usage: float = 0.0
    running: int = 0
    waiting: int = 0
    preemptions_total: int = 0


def parse_vllm_metrics(text: str) -> ParsedVLLMMetrics:
    """Parse the small Prometheus subset used by adaptive concurrency."""
    values: dict[str, list[float]] = {}
    supported = _KV_USAGE_NAMES | _RUNNING_NAMES | _WAITING_NAMES | _PREEMPTION_NAMES
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        if name not in supported:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(value):
            values.setdefault(name, []).append(value)

    def summed(names: set[str]) -> float:
        return sum(sum(values.get(name, ())) for name in names)

    usage_values = [value for name in _KV_USAGE_NAMES for value in values.get(name, ())]
    return ParsedVLLMMetrics(
        recognized=bool(values),
        kv_cache_usage=max(usage_values, default=0.0),
        running=int(summed(_RUNNING_NAMES)),
        waiting=int(summed(_WAITING_NAMES)),
        preemptions_total=int(
            max((summed({name}) for name in _PREEMPTION_NAMES), default=0.0)
        ),
    )


class InferenceMetricsScraper:
    """Best-effort vLLM metrics scraper with per-replica observations."""

    def __init__(self, base_urls: list[str], *, timeout_seconds: float) -> None:
        self.base_urls = [url.rstrip("/").removesuffix("/v1") for url in base_urls]
        self.timeout_seconds = timeout_seconds
        self.previous_preemptions: dict[str, int] = {}
        self.latest_metrics: dict[str, float] = {}

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
            samples.append(
                EngineLoadSample(
                    replica=replica,
                    kv_cache_usage=parsed.kv_cache_usage,
                    running=parsed.running,
                    waiting=parsed.waiting,
                    preemptions_delta=preemptions_delta,
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
        self.latest_metrics = metrics
        return samples if len(samples) == len(self.base_urls) else []

    def _fetch(self, base_url: str) -> str:
        request = urllib.request.Request(
            f"{base_url}/metrics",
            headers={"Accept": "text/plain"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

from __future__ import annotations

import asyncio

from wavelet.configs.config import RLAdaptiveConcurrencyConfig
from wavelet.orchestrator.concurrency import (
    AdaptiveConcurrencyController,
    EngineLoadSample,
)
from wavelet.orchestrator.inference_metrics import (
    InferenceMetricsScraper,
    parse_vllm_metrics,
)


def _sample(
    *,
    usage: float,
    running: int = 8,
    waiting: int = 0,
    preemptions: int = 0,
) -> EngineLoadSample:
    return EngineLoadSample(
        replica="replica_0",
        kv_cache_usage=usage,
        running=running,
        waiting=waiting,
        preemptions_delta=preemptions,
    )


def test_controller_additively_grows_and_multiplicatively_cuts() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            min_inflight=2,
            max_inflight=16,
            initial_inflight=8,
            additive_increase=2,
            decrease_factor=0.5,
        ),
        fallback_limit=12,
        minimum_burst=2,
    )

    grown = controller.observe([_sample(usage=0.2)], inflight=8)
    cut = controller.observe([_sample(usage=0.95)], inflight=10)

    assert grown.limit == 10
    assert grown.cancel_rollouts == 0
    assert cut.limit == 5
    assert cut.cancel_rollouts == 5
    assert cut.reason == "KV cache pressure"
    assert controller.metrics()["generation/concurrency/signal"] == 4.0


def test_controller_keeps_static_fallback_without_samples() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(),
        fallback_limit=24,
        minimum_burst=4,
    )

    decision = controller.observe([], inflight=24)

    assert decision.limit == 24
    assert decision.cancel_rollouts == 0
    assert controller.metrics()["generation/concurrency/signal"] == 0.0


def test_controller_only_grows_while_limit_is_binding() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            min_inflight=2,
            max_inflight=16,
            initial_inflight=8,
            additive_increase=2,
        ),
        fallback_limit=12,
        minimum_burst=2,
    )

    decision = controller.observe([_sample(usage=0.2)], inflight=3)

    assert decision.limit == 8
    assert decision.reason is None


def test_controller_requires_persistent_queue_before_cutting() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            max_inflight=16,
            initial_inflight=8,
            queue_persistence_polls=2,
        ),
        fallback_limit=8,
        minimum_burst=1,
    )

    first = controller.observe([_sample(usage=0.65, waiting=1)], inflight=8)
    second = controller.observe([_sample(usage=0.65, waiting=1)], inflight=8)

    assert first.limit == 8
    assert first.reason is None
    assert second.limit < 8
    assert second.reason == "request queue"


def test_parse_vllm_metrics_supports_current_and_legacy_kv_names() -> None:
    parsed = parse_vllm_metrics(
        """
        # HELP vllm:kv_cache_usage_perc KV usage
        vllm:kv_cache_usage_perc{engine="0"} 0.72
        vllm:gpu_cache_usage_perc{engine="legacy"} 0.61
        vllm:num_requests_running{engine="0"} 7
        vllm:num_requests_waiting{engine="0"} 2
        vllm:num_preemptions_total{engine="0"} 3
        """
    )

    assert parsed.kv_cache_usage == 0.72
    assert parsed.recognized is True
    assert parsed.running == 7
    assert parsed.waiting == 2
    assert parsed.preemptions_total == 3


def test_scraper_tracks_preemption_deltas_and_replica_metrics(monkeypatch) -> None:
    scraper = InferenceMetricsScraper(
        ["http://localhost:8000/v1"],
        timeout_seconds=1.0,
    )
    responses = iter(
        [
            "vllm:num_preemptions_total 3\nvllm:kv_cache_usage_perc 0.2\n",
            "vllm:num_preemptions_total 5\nvllm:kv_cache_usage_perc 0.3\n",
        ]
    )
    monkeypatch.setattr(scraper, "_fetch", lambda _base_url: next(responses))

    first = asyncio.run(scraper.scrape())
    second = asyncio.run(scraper.scrape())

    assert first[0].preemptions_delta == 0
    assert second[0].preemptions_delta == 2
    assert scraper.latest_metrics == {
        "inference/replica_0/scrape_success": 1.0,
        "inference/replica_0/kv_cache_usage": 0.3,
        "inference/replica_0/requests_running": 0.0,
        "inference/replica_0/requests_waiting": 0.0,
        "inference/replica_0/preemptions_total": 5.0,
        "inference/replica_0/preemptions_delta": 2.0,
    }


def test_scraper_requires_supported_metrics_from_every_replica(monkeypatch) -> None:
    scraper = InferenceMetricsScraper(
        ["http://localhost:8000/v1", "http://localhost:8001/v1"],
        timeout_seconds=1.0,
    )
    responses = {
        "http://localhost:8000": "vllm:kv_cache_usage_perc 0.2\n",
        "http://localhost:8001": "<html>not prometheus</html>\n",
    }
    monkeypatch.setattr(scraper, "_fetch", responses.__getitem__)

    samples = asyncio.run(scraper.scrape())

    assert samples == []
    assert scraper.latest_metrics["inference/replica_0/scrape_success"] == 1.0
    assert scraper.latest_metrics["inference/replica_1/scrape_success"] == 0.0


def test_scraper_reports_token_throughput_per_replica(monkeypatch) -> None:
    scraper = InferenceMetricsScraper(
        ["http://localhost:8000/v1"],
        timeout_seconds=1.0,
    )
    responses = iter(
        [
            (
                "vllm:kv_cache_usage_perc 0.2\n"
                "vllm:generation_tokens_total 1000\n"
                "vllm:prompt_tokens_total 400\n"
            ),
            (
                "vllm:kv_cache_usage_perc 0.3\n"
                "vllm:generation_tokens_total 1600\n"
                "vllm:prompt_tokens_total 500\n"
            ),
        ]
    )
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(scraper, "_fetch", lambda _base_url: next(responses))
    monkeypatch.setattr(scraper, "_clock", lambda: next(clock))

    asyncio.run(scraper.scrape())
    first = dict(scraper.latest_metrics)
    asyncio.run(scraper.scrape())
    second = scraper.latest_metrics

    assert first["inference/replica_0/generation_tokens_total"] == 1000.0
    assert "inference/replica_0/generation_tokens_per_second" not in first
    assert second["inference/replica_0/generation_tokens_per_second"] == 300.0
    assert second["inference/replica_0/prompt_tokens_per_second"] == 50.0
    assert second["inference/replica_0/prompt_tokens_total"] == 500.0

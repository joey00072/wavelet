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
    capacity: int | None = None,
    max_model_len: int | None = None,
    waiting_capacity: int | None = None,
) -> EngineLoadSample:
    return EngineLoadSample(
        replica="replica_0",
        kv_cache_usage=usage,
        running=running,
        waiting=waiting,
        preemptions_delta=preemptions,
        kv_capacity_tokens=capacity,
        max_model_len=max_model_len,
        waiting_capacity=waiting_capacity,
    )


def test_controller_bootstraps_from_kv_capacity_and_model_context() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            min_inflight=2,
            max_inflight=16,
        ),
        fallback_limit=12,
        minimum_burst=2,
        fallback_cost=128,
    )

    decision = controller.observe(
        [_sample(usage=0.2, capacity=1_000, max_model_len=100)],
        inflight=2,
    )

    assert decision.limit == 10
    assert controller.capacity == 1_000
    assert controller.metrics()["generation/concurrency/capacity_tokens"] == 1_000


def test_controller_stays_at_safe_floor_without_engine_metrics() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(),
        fallback_limit=24,
        minimum_burst=4,
        fallback_cost=128,
    )

    decision = controller.observe([], inflight=24)

    assert decision.limit == 4
    assert decision.cancel_rollouts == 0
    assert controller.metrics()["generation/concurrency/signal"] == 0.0


def test_controller_grows_by_one_factor_per_pipeline_turnover() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            min_inflight=2,
            max_inflight=16,
            initial_inflight=8,
            growth_factor_per_turnover=2.0,
        ),
        fallback_limit=12,
        minimum_burst=2,
        fallback_cost=128,
    )

    observed = controller.observe([_sample(usage=0.2)], inflight=8)
    decisions = []
    while controller.turnover < 1.0:
        decisions.append(
            controller.record_episode(tokens=10, inflight=controller.limit)
        )

    assert observed.limit == 8
    assert decisions[-1].limit == 16
    assert controller.turnover >= 1.0


def test_controller_growth_gate_lifetime_derives_from_poll_cadence(
    monkeypatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(
        "wavelet.orchestrator.concurrency.time.monotonic",
        lambda: now[0],
    )
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            max_inflight=16,
            initial_inflight=8,
            poll_interval_seconds=2.0,
            growth_gate_polls=2,
        ),
        fallback_limit=16,
        minimum_burst=1,
        fallback_cost=128,
    )
    controller.observe([_sample(usage=0.2)], inflight=8)

    now[0] = 3.9
    controller.record_episode(tokens=10, inflight=8)
    cap_before_expiry = controller.cap
    now[0] = 4.1
    controller.record_episode(tokens=10, inflight=8)

    assert cap_before_expiry > 8.0
    assert controller.cap == cap_before_expiry


def test_controller_requires_persistent_queue_before_cutting() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(
            max_inflight=16,
            initial_inflight=8,
            queue_persistence_polls=2,
            queue_decrease_factor=0.75,
        ),
        fallback_limit=8,
        minimum_burst=1,
        fallback_cost=128,
    )

    first = controller.observe(
        [_sample(usage=0.65, waiting=5, waiting_capacity=5)],
        inflight=8,
    )
    second = controller.observe(
        [_sample(usage=0.65, waiting=5, waiting_capacity=5)],
        inflight=8,
    )

    assert first.limit == 8
    assert first.reason is None
    assert second.limit == 6
    assert second.cancel_rollouts == 2
    assert second.reason == "queue overload"


def test_controller_soft_trim_drains_and_hard_trim_cancels() -> None:
    controller = AdaptiveConcurrencyController(
        RLAdaptiveConcurrencyConfig(max_inflight=16, initial_inflight=10),
        fallback_limit=16,
        minimum_burst=1,
        fallback_cost=128,
    )

    soft = controller.observe([_sample(usage=0.85)], inflight=10)

    assert soft.limit == 8
    assert soft.cancel_rollouts == 0
    assert soft.reason == "KV headroom (usage 0.85, soft trim)"

    controller.trim_cooldown = 0
    hard = controller.observe([_sample(usage=0.95)], inflight=10)

    assert hard.limit == 7
    assert hard.cancel_rollouts == 3
    assert hard.reason == "KV headroom (usage 0.95, hard trim)"


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


def test_parse_vllm_metrics_reads_capacity_and_capacity_queue() -> None:
    parsed = parse_vllm_metrics(
        '''
        vllm:cache_config_info{block_size="16",num_gpu_blocks="2048"} 1
        vllm:num_requests_waiting 9
        vllm:num_requests_waiting_by_reason{reason="capacity"} 7
        vllm:kv_cache_usage_perc 0.4
        '''
    )

    assert parsed.kv_capacity_tokens == 32_768
    assert parsed.waiting_capacity == 7


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


def test_scraper_exposes_kv_capacity_and_served_context(monkeypatch) -> None:
    scraper = InferenceMetricsScraper(
        ["http://localhost:8000/v1"],
        timeout_seconds=1.0,
    )
    monkeypatch.setattr(
        scraper,
        "_fetch",
        lambda _base_url: (
            'vllm:cache_config_info{kv_cache_size_tokens="327680"} 1\n'
            "vllm:kv_cache_usage_perc 0.2\n"
        ),
    )
    monkeypatch.setattr(
        scraper,
        "_fetch_model_max_len",
        lambda _base_url: 2_048,
    )

    samples = asyncio.run(scraper.scrape())

    assert samples[0].kv_capacity_tokens == 327_680
    assert samples[0].max_model_len == 2_048
    assert scraper.latest_metrics["inference/replica_0/kv_capacity_tokens"] == (
        327_680.0
    )
    assert scraper.latest_metrics["inference/replica_0/max_model_len"] == 2_048.0


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

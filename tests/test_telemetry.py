from __future__ import annotations

import torch

from wavelet.trainer.distributed import World
from wavelet.trainer.telemetry import (
    RankTelemetry,
    gather_rank_telemetry,
    node_metrics,
    rank_rows,
    sample_rank_telemetry,
)


def _rank(
    rank: int, node: str, tokens: float, seconds: float, peak: float
) -> RankTelemetry:
    return RankTelemetry(
        rank=rank,
        local_rank=rank % 2,
        node=node,
        device=f"cuda:{rank % 2}",
        tokens=tokens,
        seconds=seconds,
        memory_allocated_gib=peak - 3.0,
        peak_memory_gib=peak,
    )


def test_node_metrics_aggregate_tokens_and_memory_per_node() -> None:
    ranks = [
        _rank(0, "host-a", 600.0, 2.0, 20.0),
        _rank(1, "host-a", 400.0, 2.5, 21.0),
        _rank(2, "host-b", 900.0, 2.0, 18.0),
        _rank(3, "host-b", 1100.0, 2.0, 19.0),
    ]

    metrics = node_metrics(ranks)

    assert metrics["perf/nodes"] == 2.0
    assert metrics["node/host-a/ranks"] == 2.0
    assert metrics["node/host-a/tokens"] == 1000.0
    # Node throughput uses the slowest rank's wall time.
    assert metrics["node/host-a/tokens_per_second"] == 1000.0 / 2.5
    assert metrics["node/host-a/peak_memory_gib"] == 21.0
    assert metrics["node/host-a/memory_allocated_gib"] == 18.0
    assert metrics["node/host-b/tokens_per_second"] == 1000.0
    assert metrics["perf/rank_tokens_per_second_min"] == 160.0
    assert metrics["perf/rank_tokens_per_second_max"] == 550.0
    assert metrics["perf/rank_peak_memory_gib_max"] == 21.0
    assert node_metrics([]) == {}


def test_rank_rows_expose_per_rank_rate() -> None:
    rows = rank_rows([_rank(0, "host-a", 600.0, 2.0, 20.0)])

    assert rows[0]["rank"] == 0
    assert rows[0]["node"] == "host-a"
    assert rows[0]["tokens_per_second"] == 300.0


def test_sample_divides_tokens_by_model_parallel_replication(monkeypatch) -> None:
    monkeypatch.setenv("WAVELET_NODE_NAME", "unit/test")
    world = World(
        rank=3,
        local_rank=1,
        world_size=4,
        local_world_size=2,
        device=torch.device("cpu"),
    )

    sample = sample_rank_telemetry(world, tokens=1000, seconds=4.0, replication=2)

    assert sample.rank == 3
    assert sample.local_rank == 1
    assert sample.node == "unit_test"
    assert sample.tokens == 500.0
    assert sample.tokens_per_second == 125.0
    # Without a process group the gather is the local sample alone.
    assert gather_rank_telemetry(sample, world) == [sample]
    assert gather_rank_telemetry(sample, None) == [sample]

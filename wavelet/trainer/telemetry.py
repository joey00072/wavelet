"""Per-rank step telemetry gathered to the main rank and aggregated per node.

Every rank samples its own token count, compute time, and CUDA memory once per
optimizer step. One small ``all_gather_object`` moves the samples to every
rank; the main rank turns them into ``node/<name>/*`` metrics for
``metrics.jsonl`` and a per-rank table for ``heartbeat.json``. Nothing is
written per rank, so the artifact footprint does not grow with world size.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wavelet.trainer.distributed import World

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class RankTelemetry:
    rank: int
    local_rank: int
    node: str
    device: str
    tokens: float
    seconds: float
    memory_allocated_gib: float
    peak_memory_gib: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / max(self.seconds, 1e-9)


def node_name() -> str:
    """Stable node label: ``WAVELET_NODE_NAME`` override, else the hostname."""
    override = os.environ.get("WAVELET_NODE_NAME")
    if override:
        return _safe_label(override)
    return _safe_label(socket.gethostname() or "local")


def _safe_label(value: str) -> str:
    return value.strip().replace("/", "_") or "local"


def sample_rank_telemetry(
    world: World | None,
    *,
    tokens: float,
    seconds: float,
    replication: int = 1,
) -> RankTelemetry:
    """Sample this rank.

    ``replication`` is the number of ranks that process the same tokens (tensor,
    context, and pipeline parallel degrees combined), so summing ``tokens`` over
    a node yields that node's share of the global token count.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / _GIB
        peak = torch.cuda.max_memory_reserved() / _GIB
    else:
        allocated = peak = 0.0
    return RankTelemetry(
        rank=world.rank if world is not None else 0,
        local_rank=world.local_rank if world is not None else 0,
        node=node_name(),
        device=str(world.device) if world is not None else "cpu",
        tokens=float(tokens) / max(int(replication), 1),
        seconds=float(seconds),
        memory_allocated_gib=float(allocated),
        peak_memory_gib=float(peak),
    )


def gather_rank_telemetry(
    sample: RankTelemetry, world: World | None
) -> list[RankTelemetry]:
    """Collect every rank's sample; a collective when the process group is up."""
    if (
        world is None
        or world.world_size <= 1
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return [sample]
    gathered: list[dict[str, Any] | None] = [None] * world.world_size
    torch.distributed.all_gather_object(gathered, asdict(sample))
    ranks = [RankTelemetry(**item) for item in gathered if item is not None]
    return sorted(ranks, key=lambda item: item.rank)


def node_metrics(ranks: list[RankTelemetry]) -> dict[str, float]:
    """Flat per-node metrics plus rank imbalance for ``metrics.jsonl``."""
    if not ranks:
        return {}
    by_node: dict[str, list[RankTelemetry]] = {}
    for item in ranks:
        by_node.setdefault(item.node, []).append(item)
    metrics: dict[str, float] = {"perf/nodes": float(len(by_node))}
    for node, members in sorted(by_node.items()):
        seconds = max(member.seconds for member in members)
        prefix = f"node/{node}"
        metrics[f"{prefix}/ranks"] = float(len(members))
        metrics[f"{prefix}/tokens"] = sum(member.tokens for member in members)
        metrics[f"{prefix}/tokens_per_second"] = metrics[f"{prefix}/tokens"] / max(
            seconds, 1e-9
        )
        metrics[f"{prefix}/step_seconds"] = seconds
        metrics[f"{prefix}/peak_memory_gib"] = max(m.peak_memory_gib for m in members)
        metrics[f"{prefix}/memory_allocated_gib"] = max(
            m.memory_allocated_gib for m in members
        )
    rates = [item.tokens_per_second for item in ranks]
    metrics["perf/rank_tokens_per_second_min"] = min(rates)
    metrics["perf/rank_tokens_per_second_max"] = max(rates)
    metrics["perf/rank_peak_memory_gib_max"] = max(r.peak_memory_gib for r in ranks)
    return metrics


def rank_rows(ranks: list[RankTelemetry]) -> list[dict[str, Any]]:
    """Per-rank table for the heartbeat; latest state only, never appended."""
    rows: list[dict[str, Any]] = []
    for item in ranks:
        row = asdict(item)
        row["tokens_per_second"] = item.tokens_per_second
        rows.append(row)
    return rows

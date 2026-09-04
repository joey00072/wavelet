from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from wavelet.configs.config import SchedulerConfig
from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import SFTConfig
from wavelet.data.rl import FakeRLDataset, PackedRLDataset, RLDataset, RLExample
from wavelet.trainer import distributed as distributed_module
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.optim import setup_cosine_scheduler, setup_sqrt_scheduler
from wavelet.trainer.rl import RLTrainer
from wavelet.trainer.trainer import SFTTrainer


def _cpu_world(rank: int = 0, world_size: int = 1) -> World:
    return World(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        device=torch.device("cpu"),
    )


def _rl_example(length: int) -> RLExample:
    return RLExample(
        prompt=[],
        completion=[],
        advantage=1.0,
        reward=1.0,
        input_ids=list(range(1, length + 1)),
        target_ids=list(range(2, length + 2)),
        loss_mask=[True] * length,
        inference_logprobs=[-1.0] * length,
        temperatures=[1.0] * length,
    )


def _rl_dataset(config: RLConfig, count: int) -> RLDataset:
    return RLDataset(
        records=[_rl_example(2) for _ in range(count)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config.data,
    )


# ── epochs → optimizer steps ──────────────────────────────────────────────────


def test_epochs_total_steps_count_records_per_optimizer_batch() -> None:
    config = RLConfig(
        data={"batch_size": 4, "micro_batch_size": 1, "seq_len": 8}, epochs=3
    )
    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = _rl_dataset(config, 10)

    # 10 records * 3 epochs / 4 per optimizer step, rounded up.
    assert trainer._compute_total_steps() == 8


def test_explicit_max_steps_wins_over_epochs() -> None:
    config = RLConfig(
        data={"batch_size": 4, "micro_batch_size": 1, "seq_len": 8},
        epochs=3,
        max_steps=5,
    )
    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = _rl_dataset(config, 10)

    assert trainer._compute_total_steps() == 5


def test_epochs_require_max_steps_for_packed_datasets() -> None:
    config = RLConfig(
        data={
            "batch_size": 2,
            "micro_batch_size": 1,
            "seq_len": 8,
            "pack_sequences": True,
        }
    )
    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = PackedRLDataset(
        records=[_rl_example(2) for _ in range(4)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=config.data,
    )

    with pytest.raises(ValueError, match="packed"):
        trainer._compute_total_steps()


def test_epochs_require_max_steps_without_record_count() -> None:
    config = RLConfig(data={"batch_size": 2, "micro_batch_size": 1, "seq_len": 8})
    trainer = RLTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = FakeRLDataset(
        seq_len=8, vocab_size=16, length_mode="fixed", input_mode="random", seed=0
    )

    with pytest.raises(ValueError, match="record count"):
        trainer._compute_total_steps()


def test_sft_epochs_total_steps_use_dataset_records() -> None:
    config = SFTConfig(data={"batch_size": 4, "micro_batch_size": 1}, epochs=2)
    trainer = SFTTrainer(config)
    trainer.world = _cpu_world()
    trainer.dataset = SimpleNamespace(records=list(range(10)))  # type: ignore[assignment]

    assert trainer._compute_total_steps() == 5


# ── data partition with expert parallelism ────────────────────────────────────


def test_expert_parallel_ranks_keep_distinct_data_shards() -> None:
    config = RLConfig(data={"batch_size": 4, "micro_batch_size": 1, "seq_len": 8})
    trainer = RLTrainer(config)
    trainer.parallel_dims = ParallelDims(dp_shard=4, ep=2, world_size=4)

    partitions = []
    for rank in range(4):
        trainer.world = _cpu_world(rank=rank, world_size=4)
        partitions.append(trainer._data_partition())

    assert partitions == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert all(
        trainer._is_data_parallel_metric_leader()
        for trainer.world in (_cpu_world(rank=r, world_size=4) for r in range(4))
    )


# ── backend selection ─────────────────────────────────────────────────────────


def test_hybrid_backend_builds_cuda_device_mesh(monkeypatch) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "cpu:gloo,cuda:nccl")

    assert distributed_module._mesh_device_type() == "cuda"

    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    assert distributed_module._mesh_device_type() == "cpu"


def test_auto_backend_uses_hybrid_group_on_cuda(monkeypatch) -> None:
    trainer = RLTrainer(RLConfig())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer._distributed_backend() == "cpu:gloo,cuda:nccl"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert trainer._distributed_backend() == "gloo"


# ── learning-rate schedules ───────────────────────────────────────────────────


def _lr_curve(optimizer: torch.optim.Optimizer, scheduler, steps: int) -> list[float]:
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps - 1):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_cosine_scheduler_floor_honors_min_lr_factor() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = setup_cosine_scheduler(
        optimizer,
        total_steps=8,
        warmup_steps=2,
        lr=1.0,
        min_lr=0.0,
        min_lr_factor=0.1,
    )

    lrs = _lr_curve(optimizer, scheduler, 8)

    assert lrs[0] == pytest.approx(0.1)
    assert lrs[2] == pytest.approx(1.0)
    assert min(lrs) == pytest.approx(0.1)
    assert lrs[-1] == pytest.approx(0.1)


def test_sqrt_scheduler_holds_peak_until_decay_steps() -> None:
    def _curve(decay_steps: int) -> list[float]:
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = setup_sqrt_scheduler(
            optimizer,
            total_steps=10,
            warmup_steps=0,
            decay_steps=decay_steps,
            lr=1.0,
            min_lr=0.0,
        )
        return _lr_curve(optimizer, scheduler, 10)

    short = _curve(3)
    full = _curve(9)

    assert short[:7] == pytest.approx([1.0] * 7)
    assert short[-1] == pytest.approx(0.0, abs=1e-6)
    assert full[1] < 1.0
    assert short != full


def test_zero_decay_steps_is_rejected_instead_of_ignored() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(type="linear", decay_steps=0)

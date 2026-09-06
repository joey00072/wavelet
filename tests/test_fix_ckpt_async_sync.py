from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest
import torch

from wavelet.configs.config import CheckpointConfig
from wavelet.trainer.ckpt import (
    STABLE_CHECKPOINT_MARKER,
    CheckpointManager,
    TrainerState,
)
from wavelet.trainer.distributed import World


def _async_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> CheckpointManager:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    world = World(
        rank=0,
        local_rank=0,
        world_size=2,
        local_world_size=2,
        device=torch.device("cpu"),
    )
    manager = CheckpointManager(
        model,
        optimizer,
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        world,
    )

    def async_save(**kwargs):
        response: Future[None] = Future()
        response.set_result(None)
        return response

    monkeypatch.setattr("wavelet.trainer.ckpt.dcp.async_save", async_save)
    monkeypatch.setattr("wavelet.trainer.ckpt.barrier", lambda world: None)
    monkeypatch.setattr(
        "wavelet.trainer.distributed.distributed_uses_cuda", lambda: False
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    return manager


def test_poll_pending_save_waits_until_every_rank_reports_done(
    monkeypatch, tmp_path
) -> None:
    manager = _async_manager(monkeypatch, tmp_path)
    reduce_ops: list[object] = []
    peer_done = [False]

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        reduce_ops.append(op)
        assert tensor.device.type == "cpu"
        if not peer_done[0]:
            tensor.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    assert manager.save(TrainerState(step=1, micro_step=1))
    checkpoint_dir = tmp_path / "checkpoint-1"

    manager.poll_pending_save()

    assert reduce_ops == [torch.distributed.ReduceOp.MIN]
    assert manager.pending_save is not None
    assert not (checkpoint_dir / "meta.json").exists()
    assert not (checkpoint_dir / STABLE_CHECKPOINT_MARKER).exists()

    peer_done[0] = True
    manager.poll_pending_save()

    assert len(reduce_ops) == 2
    assert manager.pending_save is None
    assert (checkpoint_dir / "meta.json").exists()
    assert (checkpoint_dir / STABLE_CHECKPOINT_MARKER).exists()


def test_blocking_wait_finalizes_without_polling_collective(
    monkeypatch, tmp_path
) -> None:
    manager = _async_manager(monkeypatch, tmp_path)

    def unexpected_all_reduce(*args: object, **kwargs: object) -> None:
        raise AssertionError("blocking wait must not poll the done flag")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_all_reduce)
    assert manager.save(TrainerState(step=1, micro_step=1))

    manager.wait_for_pending_save()

    assert manager.pending_save is None
    assert (tmp_path / "checkpoint-1" / "meta.json").exists()

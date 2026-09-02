from __future__ import annotations

from concurrent.futures import Future

import torch

from wavelet.configs.config import CheckpointConfig
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import World


def test_async_checkpoint_uses_threads_without_shared_memory(
    monkeypatch, tmp_path
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
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
    captured = {}

    def async_save(**kwargs):
        captured.update(kwargs)
        response: Future[None] = Future()
        response.set_result(None)
        return response

    monkeypatch.setattr("wavelet.trainer.ckpt.dcp.async_save", async_save)

    assert manager.save(TrainerState(step=1, micro_step=1))
    manager.wait_for_pending_save()

    assert str(captured["async_checkpointer_type"].value) == "thread"
    assert captured["async_stager"]._config.use_shared_memory is False

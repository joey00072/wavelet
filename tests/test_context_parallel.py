from __future__ import annotations

from contextlib import contextmanager

import torch

from wavelet.trainer.context_parallel import (
    context_parallel_batch,
    prepare_context_parallel_batch,
)
from wavelet.trainer.distributed import ParallelDims


def test_context_parallel_batch_padding_preserves_packed_streams() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {
        "input_ids": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[7, 8, -100]]),
        "loss_mask": torch.tensor([[True, True, False]]),
        "temperatures": torch.tensor([[0.5, 0.5, 1.0]]),
        "rewards": torch.tensor([1.0]),
    }

    padded = prepare_context_parallel_batch(batch, dims, configured_seq_len=8)

    assert padded["input_ids"].shape == (1, 8)
    assert padded["input_ids"][0, -1].item() == 0
    assert padded["attention_mask"][0, -1].item() == 0
    assert padded["position_ids"][0].tolist() == list(range(8))
    assert padded["labels"][0, -1].item() == -100
    assert padded["loss_mask"][0, -1].item() is False
    assert padded["temperatures"][0, -1].item() == 1.0
    assert padded["rewards"].shape == (1,)


def test_context_parallel_padding_uses_head_tail_divisor() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {"input_ids": torch.ones(1, 5, dtype=torch.long)}

    padded = prepare_context_parallel_batch(batch, dims)

    assert padded["input_ids"].shape == (1, 8)


def test_context_parallel_batch_is_noop_for_cp_one() -> None:
    dims = ParallelDims(world_size=1)
    batch = {"input_ids": torch.ones(1, 3, dtype=torch.long)}

    with context_parallel_batch(batch, dims):
        assert batch["input_ids"].shape == (1, 3)


def test_context_parallel_batch_requires_sequence_fields() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)

    try:
        with context_parallel_batch({"rewards": torch.ones(1)}, dims):
            pass
    except ValueError as exc:
        assert "sequence-shaped" in str(exc)
    else:
        raise AssertionError("missing sequence fields should be rejected")


def test_context_parallel_batch_passes_all_sequence_buffers(monkeypatch) -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }
    calls = {}

    @contextmanager
    def fake_context_parallel(mesh, *, buffers, buffer_seq_dims):
        calls["mesh"] = mesh
        calls["buffers"] = buffers
        calls["seq_dims"] = buffer_seq_dims
        yield

    monkeypatch.setattr(
        "torch.distributed.tensor.experimental.context_parallel",
        fake_context_parallel,
    )
    monkeypatch.setattr(dims, "get_mesh", lambda name: "cp-mesh")
    attention_mask = torch.zeros(1, 1, 4, 4)
    with context_parallel_batch(
        batch,
        dims,
        extra_buffers=[(attention_mask, 2)],
    ):
        pass

    assert calls["seq_dims"] == [1, 1, 2]
    assert calls["buffers"][-1] is attention_mask

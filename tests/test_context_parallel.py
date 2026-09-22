from __future__ import annotations

import pytest
import torch

from wavelet.trainer.context_parallel import (
    context_parallel_batch,
    prepare_context_parallel_batch,
)
from wavelet.trainer.distributed import ParallelDims
from wavelet.trainer.trainer import SFTTrainer


def test_context_parallel_batch_padding_preserves_packed_streams() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {
        "input_ids": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[7, 8, -100]]),
        "loss_mask": torch.tensor([[True, True, False]]),
        "temperatures": torch.tensor([[0.5, 0.5, 1.0]]),
        "sampling_mask_ids": torch.tensor([[[4, 9], [5, 0], [0, 0]]]),
        "sampling_mask_lengths": torch.tensor([[2, 1, 0]]),
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
    assert padded["sampling_mask_ids"].shape == (1, 8, 2)
    assert padded["sampling_mask_ids"][0, -1].tolist() == [0, 0]
    assert padded["sampling_mask_lengths"][0, -1].item() == 0
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


def test_context_parallel_batch_rejects_explicit_attention_bias() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }
    attention_mask = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="explicit 4D"), context_parallel_batch(
        batch, dims, extra_buffers=[(attention_mask, 2)]
    ):
        pass


def test_context_parallel_batch_rejects_padding_attention_mask() -> None:
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }
    with pytest.raises(ValueError, match="all-ones 2D"), context_parallel_batch(
        batch, dims
    ):
        pass


def test_sft_loss_uses_global_supervised_token_count_under_cp(monkeypatch) -> None:
    trainer = object.__new__(SFTTrainer)
    trainer.parallel_dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer.parallel_dims, "get_mesh", lambda name: _FakeMesh())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def all_reduce(value, *, op, group):
        del op, group
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    labels = torch.tensor([[0, -100]])
    output = trainer.compute_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :1].reshape(-1, 2), labels[:, :1].reshape(-1)
    )
    assert torch.allclose(output.loss, expected)


class _FakeMesh:
    def get_group(self):
        return object()

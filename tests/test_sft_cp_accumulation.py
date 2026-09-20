from __future__ import annotations

import contextlib
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from wavelet.configs.config import SFTConfig
from wavelet.trainer.distributed import ParallelDims, World
from wavelet.trainer.trainer import SFTTrainer


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 3)
        self.fused_calls = 0
        self.logits_calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        shift_labels: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        logits = self.head(self.embedding(input_ids))
        if shift_labels is None:
            self.logits_calls += 1
            return SimpleNamespace(logits=logits)
        # Model the fused kernel's contract without requiring a CUDA kernel.
        assert (shift_labels != -100).any(), "empty shard entered fused kernel"
        self.fused_calls += 1
        return SimpleNamespace(
            loss=F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten())
        )


def _batch(rank: int, microstep: int, count: int) -> dict[str, torch.Tensor]:
    ids = torch.tensor([[(rank * 3 + microstep + i) % 8 for i in range(4)]])
    labels = (ids + rank + microstep) % 3
    labels[:, count:] = -100
    return {
        "input_ids": ids,
        "labels": labels,
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones_like(ids),
    }


def _accumulation_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        # Two successive optimizer windows also check that token counts reset.
        windows = [
            [(1, 0), (3, 1)],
            [(1, 0), (0, 0)],
            [(0, 0), (0, 0)],
        ]
        for loss_impl in ("torch", "liger_fused"):
            torch.manual_seed(83)
            model = _TinyTextModel()
            reference = copy.deepcopy(model)
            trainer = SFTTrainer(SFTConfig(loss_impl=loss_impl, max_grad_norm=0.0))
            trainer.world = World(
                rank=rank,
                local_rank=rank,
                world_size=2,
                local_world_size=2,
                device=torch.device("cpu"),
            )
            trainer.parallel_dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
            trainer.model = DistributedDataParallel(model)
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(
                trainer.optimizer, factor=1.0
            )
            trainer.accumulation_steps = 2
            # Feed already-sharded rows; this test exercises loss collectives
            # and real gradient averaging, not CUDA ring attention.
            trainer._context_parallel_batch = lambda batch: contextlib.nullcontext()
            trainer._finish_step_performance_metrics = dict
            expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
            for window_index, counts in enumerate(windows):
                global_sum = torch.zeros(())
                global_count = 0
                for microstep, per_rank in enumerate(counts):
                    for source_rank, count in enumerate(per_rank):
                        batch = _batch(source_rank, microstep, count)
                        logits = reference(**batch).logits
                        global_sum = global_sum + F.cross_entropy(
                            logits.flatten(0, 1),
                            batch["labels"].flatten(),
                            reduction="sum",
                        )
                        global_count += count
                    actual = trainer._train_step(
                        _batch(rank, microstep, per_rank[rank])
                    )
                    assert actual.stepped == (microstep == 1)
                (global_sum / max(global_count, 1)).backward()
                expected_optimizer.step()
                expected_optimizer.zero_grad(set_to_none=True)
                for parameter, expected in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        parameter,
                        expected,
                        atol=2e-7,
                        rtol=2e-6,
                        msg=f"{loss_impl}, window {window_index}, rank {rank}",
                    )
                assert trainer.step == window_index + 1
            if loss_impl == "liger_fused":
                assert model.fused_calls > 0
                assert model.logits_calls > 0
    finally:
        dist.destroy_process_group()


def test_cp_accumulation_matches_global_token_objective_and_empty_fused_shards(
    tmp_path: Path,
) -> None:
    workers = mp.spawn(
        _accumulation_worker,
        args=(str(tmp_path / "gloo-rendezvous"),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("CP normalization collectives did not finish")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_nonempty_fused_loss_preserves_nonfinite_failure() -> None:
    class NonfiniteFusedModel(nn.Module):
        def forward(self, **kwargs: torch.Tensor) -> SimpleNamespace:
            assert (kwargs["shift_labels"] != -100).any()
            return SimpleNamespace(loss=torch.tensor(float("nan")))

    trainer = SFTTrainer(SFTConfig(loss_impl="liger_fused"))
    trainer.model = NonfiniteFusedModel()
    output = trainer._forward_loss(_batch(rank=0, microstep=0, count=1))
    assert torch.isnan(output.loss), "nonempty fused failure was silently zeroed"
    assert torch.isnan(output.metrics["_loss_sum"])
    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer._require_finite_loss(output.loss, label="SFT loss")

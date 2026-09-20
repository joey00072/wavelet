from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from wavelet.trainer.context_parallel import context_parallel_batch
from wavelet.trainer.distributed import ParallelDims


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.qkv = nn.Linear(8, 24)
        self.out = nn.Linear(8, 8)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        q, k, v = self.qkv(x).view(x.shape[0], x.shape[1], 3, 2, 4).unbind(2)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.out(attended.transpose(1, 2).reshape_as(x))


def main() -> None:
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(7)
    model = TinyAttention().to(device)
    reference = TinyAttention().to(device)
    reference.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    labels = torch.tensor([[0, 1, -100, 2, -100, -100, 3, -100]], device=device)
    dims = ParallelDims(cp=2, dp_shard=1, world_size=2)
    batch = {"input_ids": ids.clone(), "labels": labels.clone()}
    with context_parallel_batch(batch, dims):
        local_logits = model(batch["input_ids"])
        local_labels = batch["labels"]
        count = (local_labels != -100).sum().float()
        global_count = count.clone()
        group = dims.get_mesh("dp_cp").get_group()
        dist.all_reduce(global_count, group=group)
        per_token = F.mse_loss(
            local_logits, torch.zeros_like(local_logits), reduction="none"
        ).mean(dim=-1)
        (
            per_token.masked_select(local_labels != -100).sum() * 2 / global_count
        ).backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    for grad in grads:
        dist.all_reduce(grad, group=group)
        grad.div_(2)
    expected = (
        F.mse_loss(reference(ids), torch.zeros_like(reference(ids)), reduction="none")
        .mean(dim=-1)
        .masked_select(labels != -100)
        .sum()
        / global_count
    )
    expected.backward()
    if rank == 0:
        for grad, parameter in zip(grads, reference.parameters(), strict=True):
            torch.testing.assert_close(grad, parameter.grad, rtol=2e-3, atol=2e-3)
        print("cp ring sdpa gradient parity ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

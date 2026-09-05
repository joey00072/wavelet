"""First-stage context-parallel input handling for SDPA models."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import Tensor

from wavelet.trainer.distributed import ParallelDims


_SEQUENCE_FIELDS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "target_ids",
    "labels",
    "loss_mask",
    "advantages",
    "rl_weights",
    "ce_weights",
    "ref_kl_weights",
    "inference_logprobs",
    "teacher_logprobs",
    "temperatures",
}


def _padding_value(name: str, tensor: Tensor) -> int | float | bool:
    if name == "input_ids":
        return 0
    if name == "labels":
        return -100
    if name == "temperatures":
        return 1.0
    if tensor.dtype == torch.bool:
        return False
    return 0


def _pad_sequence_tensor(
    name: str,
    tensor: Tensor,
    target_length: int,
) -> Tensor:
    padding = target_length - tensor.shape[1]
    if padding <= 0:
        return tensor
    if name == "position_ids":
        tail = torch.arange(
            tensor.shape[1],
            target_length,
            device=tensor.device,
            dtype=tensor.dtype,
        ).expand(tensor.shape[0], -1)
    else:
        tail_shape = (*tensor.shape[:1], padding, *tensor.shape[2:])
        tail = torch.full(
            tail_shape,
            _padding_value(name, tensor),
            dtype=tensor.dtype,
            device=tensor.device,
        )
    return torch.cat((tensor, tail), dim=1)


def prepare_context_parallel_batch(
    batch: dict[str, Tensor],
    parallel_dims: ParallelDims | None,
    *,
    configured_seq_len: int | None = None,
) -> dict[str, Tensor]:
    """Pad sequence-shaped batch fields to a common CP-compatible length.

    CP ranks receive the same packed rows, so sequence fields must have an
    identical length and that length must be divisible by the CP degree. The
    configured sequence length is used when available to keep shapes stable
    across data-parallel ranks; padding is ignored by the masks and labels.
    """
    if parallel_dims is None or not parallel_dims.cp_enabled:
        return batch

    sequence_lengths = {
        value.shape[1]
        for name, value in batch.items()
        if name in _SEQUENCE_FIELDS and value.ndim >= 2
    }
    if not sequence_lengths:
        return batch
    current_length = max(sequence_lengths)
    if len(sequence_lengths) != 1:
        raise ValueError(
            "Context-parallel sequence fields must have the same sequence length."
        )
    target_length = max(current_length, configured_seq_len or 0)
    divisor = parallel_dims.seq_len_divisor
    target_length = ((target_length + divisor - 1) // divisor) * divisor
    if target_length == current_length:
        return batch
    return {
        name: (
            _pad_sequence_tensor(name, value, target_length)
            if name in _SEQUENCE_FIELDS and value.ndim >= 2
            else value
        )
        for name, value in batch.items()
    }


@contextlib.contextmanager
def context_parallel_batch(
    batch: dict[str, Tensor],
    parallel_dims: ParallelDims | None,
    *,
    extra_buffers: list[tuple[Tensor, int]] | None = None,
) -> Iterator[None]:
    """Shard batch sequence fields while SDPA forwards and backwards run."""
    if parallel_dims is None or not parallel_dims.cp_enabled:
        yield
        return

    from torch.distributed.tensor.experimental import context_parallel

    buffers = [
        (value, 1)
        for name, value in batch.items()
        if name in _SEQUENCE_FIELDS and value.ndim >= 2
    ]
    buffers.extend(extra_buffers or [])
    if not buffers:
        raise ValueError("Context parallelism requires sequence-shaped batch fields.")
    for buffer, seq_dim in buffers:
        if buffer.shape[seq_dim] % parallel_dims.cp != 0:
            raise ValueError(
                "Context-parallel buffers must be divisible by the CP degree; "
                f"got length {buffer.shape[seq_dim]} and cp={parallel_dims.cp}."
            )
    with context_parallel(
        parallel_dims.get_mesh("cp"),
        buffers=[value for value, _ in buffers],
        buffer_seq_dims=[seq_dim for _, seq_dim in buffers],
    ):
        yield

"""CPU-offloaded gradient checkpointing with sqrt(N)-layer selection.

Standard GC checkpoints every transformer layer (N checkpoints, N CPU↔GPU
transfers per step).  Unsloth's approach checkpoints only every ≈√N layers,
reducing transfers from N to √N ≈ 5 for a 28-layer model.

This module implements the same strategy:
  1. Locate the decoder layer list in the model.
  2. Compute k = ceil(√N) evenly-spaced checkpoint indices.
  3. For checkpoint layers: replace _gradient_checkpointing_func with
     _cpu_offload_checkpoint (offloads input hidden-state to pinned CPU RAM).
  4. For non-checkpoint layers: disable gradient_checkpointing entirely so
     they store intermediate activations on GPU without any transfer.

Memory footprint:
  - Standard GC (all layers, GPU):  N × 10 MB ≈ 280 MB on GPU
  - Full CPU-offload (all layers):  N × 10 MB ≈ 280 MB on CPU, 0 MB on GPU
    but N PCIe round-trips per step → bottleneck on a single GPU
  - Sqrt-layer CPU-offload (this):  k × 10 MB ≈ 50 MB on CPU + some GPU
    activations for intra-segment layers, k≈5 PCIe round-trips per step

Usage — call `patch_model_gradient_checkpointing(model)` AFTER
`gradient_checkpointing_enable()`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


_PIN_MEMORY = True


class _CPUOffloadCheckpointFn(torch.autograd.Function):
    """Gradient checkpointing that stores recompute inputs in CPU pinned memory.

    forward:  runs fn(*args) with no_grad, then offloads the input tensors to
              pinned CPU memory for storage.
    backward: streams inputs back to GPU, reruns fn to get activations, then
              runs the standard backward pass.
    """

    @staticmethod
    def forward(ctx, fn, *args):  # type: ignore[override]
        ctx.fn = fn

        # Partition into tensors (offloaded) and non-tensors (kept as-is).
        ctx.tensor_indices = [i for i, a in enumerate(args) if isinstance(a, Tensor)]
        ctx.non_tensor_indices = [
            i for i, a in enumerate(args) if not isinstance(a, Tensor)
        ]
        ctx.non_tensor_args = [args[i] for i in ctx.non_tensor_indices]
        ctx.tensor_requires_grad = [args[i].requires_grad for i in ctx.tensor_indices]

        # Run forward on GPU (no grad — recomputed on backward).
        with torch.no_grad():
            outputs = fn(*args)

        # Offload tensor inputs to pinned CPU memory asynchronously.
        # pin_memory=True enables fast DMA transfer on the next backward.
        cpu_tensors = []
        for i in ctx.tensor_indices:
            t = args[i].detach()
            try:
                cpu = t.to("cpu", non_blocking=True)
                if _PIN_MEMORY:
                    cpu = cpu.pin_memory()
            except RuntimeError:
                cpu = t.to("cpu", non_blocking=True)
            cpu_tensors.append(cpu)
        ctx.save_for_backward(*cpu_tensors)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(); "
                "use .backward() for gradient computation."
            )
        # Restore tensor inputs from CPU to GPU.
        device = grad_outputs[0].device if grad_outputs else torch.device("cuda")
        gpu_tensors = [
            t.to(device, non_blocking=True).requires_grad_(rg)
            for t, rg in zip(ctx.saved_tensors, ctx.tensor_requires_grad)
        ]

        # Reconstruct the original args list.
        n_args = len(ctx.tensor_indices) + len(ctx.non_tensor_indices)
        args: list = [None] * n_args
        for idx, t in zip(ctx.tensor_indices, gpu_tensors):
            args[idx] = t
        for idx, v in zip(ctx.non_tensor_indices, ctx.non_tensor_args):
            args[idx] = v

        # Recompute forward with gradients enabled.
        with torch.enable_grad():
            outputs = ctx.fn(*args)

        # Run backward through the recomputed graph.
        if isinstance(outputs, Tensor):
            outputs_tuple = (outputs,)
        else:
            outputs_tuple = tuple(outputs)

        torch.autograd.backward(outputs_tuple, grad_outputs)

        # Collect gradients for the tensor inputs.
        input_grads: list = [None] * n_args
        for idx, t in zip(ctx.tensor_indices, gpu_tensors):
            input_grads[idx] = t.grad if t.requires_grad else None

        return (None,) + tuple(input_grads)


def _cpu_offload_checkpoint(fn, *args, **kwargs):
    """Drop-in replacement for torch.utils.checkpoint.checkpoint.

    Ignores `use_reentrant` and `preserve_rng_state` kwargs (always behaves as
    use_reentrant=True but with CPU-offloaded inputs).
    """
    kwargs.pop("use_reentrant", None)
    kwargs.pop("preserve_rng_state", None)
    if kwargs:
        # Pass unknown kwargs through to fn (e.g. attention_mask)
        import functools

        wrapped = functools.partial(fn, **kwargs)
        return _CPUOffloadCheckpointFn.apply(wrapped, *args)
    return _CPUOffloadCheckpointFn.apply(fn, *args)


def _find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module] | None:
    """Return the flat list of transformer decoder layers, or None if not found.

    Tries common attribute paths used by HuggingFace + PEFT:
      PeftModel  → base_model (LoraModel) → model (PreTrainedModel) → model.layers
    """
    candidates = [
        "base_model.model.model.layers",  # PEFT QLoRA wrapped Llama/Qwen style
        "base_model.model.transformer.h",  # PEFT wrapped GPT-2 style
        "model.model.layers",  # single PEFT wrap
        "model.transformer.h",
        "model.layers",
        "transformer.h",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    return None


def patch_model_gradient_checkpointing(
    model: torch.nn.Module,
    *,
    pin_memory: bool = True,
    max_checkpoints: int | None = None,
) -> None:
    """Apply sqrt(N)-layer CPU-offloaded gradient checkpointing.

    Must be called AFTER gradient_checkpointing_enable().

    Selects k = ceil(sqrt(N)) evenly-spaced layers to be CPU-offload
    checkpoint boundaries.  The remaining N-k layers keep standard gradient
    checkpointing (boundary hidden-states stored on GPU, ~10 MB each).

    Memory layout (N=28, k=6):
      - 6 checkpoint layers: boundary hidden-state in pinned CPU RAM → 0 GPU
      - 22 standard-GC layers: boundary hidden-state on GPU → 220 MB GPU
      - PCIe transfers per step: 6 (vs 28 for full CPU offload)

    Disabling GC entirely on non-checkpoint layers would keep ALL their
    intermediate activations on GPU throughout the forward pass, causing OOM
    on tight-memory setups.  Keeping standard GC on those layers avoids this
    while still reducing transfers by 4–5×.
    """
    import transformers

    global _PIN_MEMORY
    _PIN_MEMORY = pin_memory

    layers = _find_decoder_layers(model)

    if layers is None:
        # Fallback: patch every module that uses _gradient_checkpointing_func.
        # This is full CPU offload (N transfers) — used when layer list is not found.
        for m in [model] + list(model.modules()):
            if hasattr(m, "_gradient_checkpointing_func"):
                m._gradient_checkpointing_func = _cpu_offload_checkpoint  # type: ignore[assignment]
        torch.utils.checkpoint.checkpoint = _cpu_offload_checkpoint  # type: ignore[assignment]
        if hasattr(transformers, "modeling_utils"):
            transformers.modeling_utils.checkpoint = _cpu_offload_checkpoint
        return

    n = len(layers)
    k = math.ceil(math.sqrt(n))  # number of CPU-offload checkpoints ≈ sqrt(N)
    if max_checkpoints is not None:
        k = max(1, min(k, max_checkpoints))
    seg_size = math.ceil(n / k)  # layers per segment

    # Select the first layer of each segment for CPU offloading.
    # Non-selected layers keep their existing (standard GPU) checkpoint function.
    checkpoint_indices: set[int] = {j * seg_size for j in range(k) if j * seg_size < n}

    for i, layer in enumerate(layers):
        if i in checkpoint_indices and hasattr(layer, "_gradient_checkpointing_func"):
            layer._gradient_checkpointing_func = _cpu_offload_checkpoint  # type: ignore[assignment]

    # Leave the global checkpoint function alone here so non-selected layers keep
    # the standard PyTorch checkpoint path. Only the chosen boundary layers use
    # CPU offload.

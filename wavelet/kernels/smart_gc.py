"""Wavelet smart gradient checkpointing — no unsloth dependency.

Monkey-patches torch.utils.checkpoint.CheckpointFunction so that large
activation tensors saved at each checkpoint boundary are streamed to
pinned CPU RAM during forward and fetched back during backward.

Based on the approach described in https://unsloth.ai/docs/blog/500k-context-length-fine-tuning#unsloth-gradient-checkpointing-enhancements
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Tensors smaller than 2 MB are not worth the CPU↔GPU round-trip overhead.
_MINIMUM_OFFLOAD_NUMEL = 2 * 1024 * 1024 // 2  # elements (assumes bf16/fp16)

_original_CheckpointFunction: type | None = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _get_device_states(*args) -> tuple[list[int], list]:
    seen: set[int] = set()
    devices: list[int] = []
    for arg in args:
        if torch.is_tensor(arg) and arg.is_cuda:
            dev = arg.get_device()
            if dev not in seen:
                devices.append(dev)
                seen.add(dev)
    states = [torch.cuda.get_rng_state(d) for d in devices]
    return devices, states


def _set_device_states(devices: list[int], states: list) -> None:
    for dev, state in zip(devices, states):
        torch.cuda.set_rng_state(state, dev)


# ── WaveletCheckpointFunction ─────────────────────────────────────────────────


class WaveletCheckpointFunction(torch.autograd.Function):
    """Reentrant gradient checkpointing with CPU offloading of saved tensors.

    Large CUDA tensors (> 2 MB) are moved to pinned CPU RAM immediately
    after saving, freeing GPU memory between the forward and backward passes.
    They are moved back to GPU just before the recomputation in backward.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):  # type: ignore[override]
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.autocast_states = _capture_autocast_states()
        ctx.had_cuda_in_fwd = torch.cuda._initialized
        if ctx.had_cuda_in_fwd:
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = _get_device_states(*args)

        with torch.no_grad():
            outputs = run_function(*args)

        # Split tensor / non-tensor args.  Non-tensors go into ctx.inputs
        # directly; tensors are saved via save_for_backward (possibly as
        # CPU copies for large ones).
        ctx.inputs: list = list(args)
        ctx.tensor_indices: list[int] = []
        ctx.offload_info: list[tuple[bool, torch.device | None, bool]] = []
        tensor_saves: list[torch.Tensor] = []

        for i, arg in enumerate(args):
            if not torch.is_tensor(arg):
                continue
            ctx.tensor_indices.append(i)
            if arg.is_cuda and arg.numel() > _MINIMUM_OFFLOAD_NUMEL:
                cpu = torch.empty(
                    arg.shape, dtype=arg.dtype, device="cpu", pin_memory=True
                )
                cpu.copy_(arg, non_blocking=False)
                tensor_saves.append(cpu)
                ctx.offload_info.append((True, arg.device, arg.requires_grad))
                ctx.inputs[i] = None  # release GPU reference
            else:
                tensor_saves.append(arg)
                ctx.offload_info.append((False, None, arg.requires_grad))

        ctx.save_for_backward(*tensor_saves)
        return outputs

    @staticmethod
    def backward(ctx, *args):  # type: ignore[override]
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "WaveletCheckpointFunction is not compatible with .grad(). "
                "Use .backward() instead."
            )

        inputs = _restore_checkpoint_inputs(ctx)
        detached, outputs = _recompute_checkpoint_outputs(ctx, inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad, args_with_grad = [], []
        for out, grad in zip(outputs, args):
            if torch.is_tensor(out) and out.requires_grad:
                outputs_with_grad.append(out)
                args_with_grad.append(grad)

        if not outputs_with_grad:
            raise RuntimeError(
                "WaveletCheckpointFunction: none of the recomputed outputs "
                "require grad — check your model configuration."
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        return (None, None) + tuple(
            inp.grad if torch.is_tensor(inp) and inp.requires_grad else None
            for inp in detached
        )


def _restore_checkpoint_inputs(ctx) -> list:
    """Restore saved activations to their original devices."""
    inputs = list(ctx.inputs)
    saved = list(ctx.saved_tensors)
    for saved_index, (input_index, offload_info) in enumerate(
        zip(ctx.tensor_indices, ctx.offload_info)
    ):
        was_offloaded, device, _ = offload_info
        if was_offloaded:
            saved[saved_index] = saved[saved_index].to(device, non_blocking=False)
        inputs[input_index] = saved[saved_index]
    return inputs


def _recompute_checkpoint_outputs(ctx, inputs: list) -> tuple[tuple, object]:
    requires_grad = {
        index: value
        for index, (_, _, value) in zip(ctx.tensor_indices, ctx.offload_info)
    }
    rng_devices = (
        ctx.fwd_gpu_devices if ctx.preserve_rng_state and ctx.had_cuda_in_fwd else []
    )
    with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda_in_fwd:
                _set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
        detached = tuple(
            value.detach().requires_grad_(requires_grad.get(index, False))
            if torch.is_tensor(value)
            else value
            for index, value in enumerate(inputs)
        )
        with torch.enable_grad(), _autocast_contexts(ctx.autocast_states):
            outputs = ctx.run_function(*detached)
    return detached, outputs


_AUTOCAST_DEVICE_TYPES = ("cuda", "cpu")


def _capture_autocast_states() -> dict[str, tuple[bool, torch.dtype]]:
    return {
        device_type: (
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
        )
        for device_type in _AUTOCAST_DEVICE_TYPES
    }


class _autocast_contexts:
    """Re-enter the forward's autocast state for every device type."""

    def __init__(self, states: dict[str, tuple[bool, torch.dtype]]) -> None:
        self._contexts = [
            torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
            for device_type, (enabled, dtype) in states.items()
            if enabled
        ]

    def __enter__(self) -> None:
        for context in self._contexts:
            context.__enter__()

    def __exit__(self, *exc_info: object) -> None:
        for context in reversed(self._contexts):
            context.__exit__(*exc_info)


# ── patch ───────────────────────────────────────────────────────────────────


def patch_smart_gc(
    model: torch.nn.Module,
    *,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Apply Wavelet smart gradient checkpointing.

    - Monkey-patches torch.utils.checkpoint.CheckpointFunction with
      WaveletCheckpointFunction which offloads large (> 2 MB) activation
      tensors to pinned CPU RAM during forward and restores them in backward.
    - Re-enables model gradient checkpointing with use_reentrant=True so the
      patched CheckpointFunction is actually used.

    For seq_len < 512 the CPU I/O overhead exceeds the VRAM savings, so
    standard GC is left in place and this function returns False.

    Returns True if the patch was applied.
    """
    global _original_CheckpointFunction

    if seq_len < 512:
        logger.info("patch_smart_gc: seq_len=%d < 512 — skipping CPU offload", seq_len)
        return False

    if _original_CheckpointFunction is not None:
        logger.debug("patch_smart_gc: already patched")
        return True

    import torch.utils.checkpoint as _cp

    _original_CheckpointFunction = _cp.CheckpointFunction
    original_checkpoint = _cp.checkpoint
    _cp.CheckpointFunction = WaveletCheckpointFunction

    def _wavelet_checkpoint(function, *args, **kwargs):
        preserve = kwargs.pop("preserve_rng_state", True)
        use_reentrant = kwargs.pop("use_reentrant", True)
        if not use_reentrant:
            # Non-reentrant path: fall back to original (no offloading)
            return original_checkpoint(function, *args, use_reentrant=False, **kwargs)
        return WaveletCheckpointFunction.apply(function, preserve, *args)

    _cp.checkpoint = _wavelet_checkpoint

    # Switch the model to reentrant GC so our patched CheckpointFunction runs.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    logger.info(
        "patch_smart_gc: WaveletCheckpointFunction active (seq_len=%d)", seq_len
    )
    return True

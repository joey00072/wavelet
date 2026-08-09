from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Sequence
from contextlib import nullcontext

import psutil
import torch
from torch import Tensor
from torch.autograd.graph import saved_tensors_hooks
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from wavelet.configs.sft import (
    ActivationOffloadingConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def setup_optimizer(
    config: OptimizerConfig,
    named_params: Iterable[tuple[str, nn.Parameter]],
) -> Optimizer:
    named_params = list(named_params)
    _validate_single_trainable_lora_adapter(named_params)
    params = [param for _, param in named_params if param.requires_grad]

    if config.type in ("adamw", "adamw_8bit", "paged_adamw_8bit"):
        if config.type == "adamw_8bit":
            from bitsandbytes.optim import AdamW8bit

            # 8-bit optimizers don't support fused/foreach; force for-loop
            return _build_optimizer(
                AdamW8bit,
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
                implementation="for-loop",
            )
        if config.type == "paged_adamw_8bit":
            from bitsandbytes.optim import PagedAdamW8bit

            return _build_optimizer(
                PagedAdamW8bit,
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
                implementation="for-loop",
            )
        return _build_optimizer(
            AdamW,
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
            implementation=config.implementation,
        )
    if config.type in ("adam", "adam_8bit"):
        if config.type == "adam_8bit":
            from bitsandbytes.optim import Adam8bit

            return _build_optimizer(
                Adam8bit,
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
                implementation="for-loop",
            )
        return _build_optimizer(
            Adam,
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.betas1, config.betas2),
            implementation=config.implementation,
        )
    if config.type == "sgd":
        return _build_optimizer(
            SGD,
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            implementation=config.implementation,
        )
    raise ValueError(f"Unsupported optimizer type: {config.type}")


def _validate_single_trainable_lora_adapter(
    named_params: Sequence[tuple[str, nn.Parameter]],
) -> None:
    adapter_names = {
        adapter_name
        for name, param in named_params
        if param.requires_grad
        for adapter_name in [_lora_adapter_name_from_parameter(name)]
        if adapter_name is not None
    }
    if len(adapter_names) > 1:
        raise RuntimeError(
            "Wavelet optimizers support trainable parameters from exactly one "
            f"LoRA adapter; found {sorted(adapter_names)}."
        )


def _lora_adapter_name_from_parameter(name: str) -> str | None:
    for marker in (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
    ):
        if marker not in name:
            continue
        suffix = name.split(marker, 1)[1]
        if "." not in suffix:
            return None
        return suffix.split(".", 1)[0]
    return None


def _build_optimizer(
    cls: type[Optimizer],
    params: list[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    implementation: str,
    betas: tuple[float, float] | None = None,
    momentum: float | None = None,
    nesterov: bool | None = None,
) -> Optimizer:
    kwargs: dict[str, object] = {}
    if implementation == "fused":
        kwargs["fused"] = True
    elif implementation == "foreach":
        kwargs["foreach"] = True

    if cls in {AdamW, Adam}:
        assert betas is not None
        kwargs["betas"] = betas
    if cls is SGD:
        assert momentum is not None
        assert nesterov is not None
        kwargs["momentum"] = momentum
        kwargs["nesterov"] = nesterov

    if implementation == "fused" and not torch.cuda.is_available():
        warnings.warn(
            "Fused optimizer requested for non-CUDA runtime; "
            "using for-loop implementation."
        )
        kwargs.pop("fused")

    try:
        return cls(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    except TypeError as exc:
        if kwargs.get("fused") is True:
            warnings.warn(
                f"{cls.__name__} fused optimizer unsupported in this runtime; "
                f"falling back to for-loop. {exc}"
            )
            kwargs.pop("fused")
            return cls(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        if kwargs.get("foreach") is True:
            warnings.warn(
                f"{cls.__name__} foreach optimizer unsupported in this runtime; "
                f"falling back to for-loop. {exc}"
            )
            kwargs.pop("foreach")
            return cls(
                params=params,
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        raise


def _resolve_warmup_steps(
    scheduler_config: SchedulerConfig,
    total_steps: int,
) -> int:
    if scheduler_config.warmup_steps is not None:
        return scheduler_config.warmup_steps
    return int(total_steps * scheduler_config.warmup_ratio)


def _resolve_decay_steps(
    scheduler_config: SchedulerConfig,
    total_steps: int,
    warmup_steps: int,
) -> int:
    if scheduler_config.decay_steps is not None and scheduler_config.decay_steps > 0:
        return scheduler_config.decay_steps
    return max(int(total_steps * scheduler_config.decay_ratio) - warmup_steps, 1)


def _resolve_min_lr_factor(
    min_lr: float,
    min_lr_factor: float | None,
    lr: float,
) -> float:
    if min_lr_factor is not None:
        return min_lr_factor
    if min_lr <= 0.0:
        return 1e-8
    return min_lr / lr


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    *,
    total_steps: int,
    lr: float,
) -> LRScheduler:
    warmup_steps = _resolve_warmup_steps(scheduler_config, total_steps)
    decay_steps = _resolve_decay_steps(
        scheduler_config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    if scheduler_config.type == "constant":
        return setup_constant_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    if scheduler_config.type == "linear":
        return setup_linear_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    if scheduler_config.type == "cosine":
        return setup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
            decay_steps=decay_steps,
        )
    if scheduler_config.type == "sqrt":
        return setup_sqrt_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            lr=lr,
            min_lr=scheduler_config.min_lr,
            min_lr_factor=scheduler_config.min_lr_factor,
        )
    raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")


def setup_constant_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    min_lr_factor = _resolve_min_lr_factor(
        min_lr,
        min_lr_factor,
        lr,
    )

    if total_steps <= 0:
        raise ValueError("Constant scheduler requires total_steps > 0")
    if warmup_steps <= 0:
        return ConstantLR(optimizer, factor=1.0)

    warmup_steps = min(warmup_steps, total_steps)
    schedulers: list[LRScheduler] = [
        LinearLR(
            optimizer,
            start_factor=min_lr_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    ]
    milestones: list[int] = [warmup_steps]

    constant_steps = max(total_steps - warmup_steps, 1)
    schedulers.append(
        LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=constant_steps,
        )
    )
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup_linear_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    min_lr_factor = _resolve_min_lr_factor(
        min_lr,
        min_lr_factor,
        lr,
    )

    if total_steps <= 0:
        raise ValueError("Linear scheduler requires total_steps > 0")
    if warmup_steps <= 0 and decay_steps <= 0:
        raise ValueError(
            "Linear scheduler requires warmup_steps or decay_steps to be set."
        )

    # Clamp warmup so it never exceeds total_steps (e.g. short benchmark runs)
    warmup_steps = min(warmup_steps, total_steps)

    effective_total = max(total_steps - 1, 1)

    decay_steps = max(min(decay_steps, effective_total), 1)

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    decay_start = min(total_steps - decay_steps, effective_total)
    if decay_steps > 0 and decay_start >= warmup_steps:
        constant_steps = decay_start - warmup_steps
        if constant_steps > 0:
            schedulers.append(
                LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1.0,
                    total_iters=constant_steps,
                )
            )
            milestones.append(decay_start)
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr_factor,
                total_iters=max(decay_steps - 1, 1),
            )
        )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


_PIN_MEMORY = True


class OffloadActivations(saved_tensors_hooks):
    """Offload large, non-parameter saved activations to CPU during forward."""

    def __init__(
        self,
        *,
        use_pin_memory: bool = True,
        min_offload_size: int = 1024,
    ) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Activation offloading requires a CUDA device.")
        self.use_pin_memory = use_pin_memory
        self.min_tensor_size_bytes = min_offload_size
        self.virtual_memory_safe_pct = 60
        self._tracker: dict[int, tuple[torch.Tensor, bool]] = {}
        self._tensor_id = 0
        self._is_first_forward_call = True
        self._is_first_backward_call = True
        self._warned_on_ram = False
        super().__init__(self._pack_tensor, self._unpack_tensor)

    def _next_tensor_id(self) -> int:
        self._tensor_id += 1
        return self._tensor_id

    def _verify_virtual_memory(self) -> None:
        if self._warned_on_ram:
            return
        current_pct = psutil.virtual_memory().percent
        if current_pct > self.virtual_memory_safe_pct:
            logging.getLogger(__name__).warning(
                "CPU memory usage is high during activation offloading: %s%% > %s%%",
                current_pct,
                self.virtual_memory_safe_pct,
            )
        self._warned_on_ram = True

    def _pack_tensor(self, activation: torch.Tensor) -> int:
        if self._is_first_forward_call:
            assert not self._tracker, (
                "Activation tracker should be empty at the start of a forward pass."
            )
            self._is_first_forward_call = False
            self._is_first_backward_call = True

        tensor_id = self._next_tensor_id()
        num_bytes = activation.element_size() * activation.nelement()
        should_offload = (
            activation.device.type == "cuda"
            and num_bytes >= self.min_tensor_size_bytes
            and not isinstance(activation, torch.nn.Parameter)
            and not (
                hasattr(torch.nn, "Buffer") and isinstance(activation, torch.nn.Buffer)
            )
        )
        if should_offload:
            cpu_tensor = torch.empty_like(
                activation,
                pin_memory=self.use_pin_memory,
                device="cpu",
            )
            cpu_tensor.copy_(activation, non_blocking=True)
            self._tracker[tensor_id] = (cpu_tensor, True)
        else:
            self._tracker[tensor_id] = (activation, False)
        return tensor_id

    def _unpack_tensor(self, tensor_id: int) -> torch.Tensor:
        if self._is_first_backward_call:
            self._is_first_backward_call = False
            self._is_first_forward_call = True
            if self.use_pin_memory:
                self._verify_virtual_memory()
        if tensor_id not in self._tracker:
            raise KeyError(f"Untracked activation tensor id: {tensor_id}")
        tensor, offloaded = self._tracker.pop(tensor_id)
        return tensor.to("cuda", non_blocking=True) if offloaded else tensor


def maybe_activation_offloading(
    config: ActivationOffloadingConfig | None,
) -> OffloadActivations | nullcontext:
    if config is None:
        return nullcontext()
    return OffloadActivations(use_pin_memory=config.pin_memory)


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


def setup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
    decay_steps: int | None = None,
) -> LRScheduler:
    if total_steps <= 0:
        raise ValueError("Cosine scheduler requires total_steps > 0")

    min_lr_factor = _resolve_min_lr_factor(min_lr, min_lr_factor, lr)

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    effective_total = max(total_steps - 1, 1)
    decay_steps = min(
        max(effective_total - warmup_steps, 1), decay_steps or effective_total
    )
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=min_lr,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def setup_sqrt_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    lr: float,
    min_lr: float,
    min_lr_factor: float | None = None,
) -> LRScheduler:
    if total_steps <= 0:
        raise ValueError("SQRT scheduler requires total_steps > 0")

    min_lr_factor = _resolve_min_lr_factor(min_lr, min_lr_factor, lr)
    decay_steps = max(decay_steps, 1)

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    schedulers: list[LRScheduler] = []
    milestones: list[int] = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=min_lr_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    decay_span = max(total_steps - warmup_steps - 1, 1)

    def _sqrt_factor(step: int) -> float:
        if step <= 0:
            return 1.0
        t = min(step / decay_span, 1.0)
        return min_lr_factor + (1.0 - min_lr_factor) * math.sqrt(max(1.0 - t, 0.0))

    schedulers.append(
        LambdaLR(
            optimizer,
            _sqrt_factor,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

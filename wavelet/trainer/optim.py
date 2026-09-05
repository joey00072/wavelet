from __future__ import annotations

import copy
import logging
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext

import psutil
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import DTensor
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.hooks import RemovableHandle

from wavelet.configs.sft import (
    ActivationOffloadingConfig,
    OptimizerConfig,
    SchedulerConfig,
)


class SignSGD(Optimizer):
    """Stateless sign-gradient descent with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients.")
                if weight_decay > 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(param.grad.sign(), alpha=-lr)
        return loss


class OptimizerStateOffloader:
    """Keep optimizer state in pinned CPU memory between optimizer steps."""

    def __init__(self, optimizer: Optimizer, *, pin_memory: bool = True) -> None:
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._handles: list[RemovableHandle] = []

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("Optimizer state offload hooks are already installed.")
        self._handles = [
            self.optimizer.register_step_pre_hook(self._before_step),
            self.optimizer.register_step_post_hook(self._after_step),
            self.optimizer.register_state_dict_pre_hook(self._before_state_dict),
            self.optimizer.register_state_dict_post_hook(self._after_state_dict),
            self.optimizer.register_load_state_dict_post_hook(
                self._after_load_state_dict
            ),
        ]
        self.move_to_cpu()

    def move_to_parameters(self) -> None:
        for parameter, state in self.optimizer.state.items():
            for key, value in list(state.items()):
                state[key] = _move_optimizer_state_value(
                    value,
                    parameter.device,
                    pin_memory=False,
                )

    def move_to_cpu(self) -> None:
        should_pin = self.pin_memory and torch.cuda.is_available()
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                state[key] = _move_optimizer_state_value(
                    value,
                    torch.device("cpu"),
                    pin_memory=should_pin,
                )

    def _before_step(
        self,
        _optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        self.move_to_parameters()

    def _after_step(
        self,
        _optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        self.move_to_cpu()

    def _before_state_dict(self, _optimizer: Optimizer) -> None:
        self.move_to_parameters()

    def _after_state_dict(
        self,
        _optimizer: Optimizer,
        state_dict: dict[str, object],
    ) -> dict[str, object]:
        self.move_to_cpu()
        return state_dict

    def _after_load_state_dict(self, _optimizer: Optimizer) -> None:
        self.move_to_cpu()


def _move_optimizer_state_value(
    value: object,
    device: torch.device,
    *,
    pin_memory: bool,
) -> object:
    if isinstance(value, DTensor):
        local_tensor = _move_optimizer_state_value(
            value._local_tensor,
            device,
            pin_memory=pin_memory,
        )
        moved = copy.copy(value)
        moved._local_tensor = local_tensor
        return moved
    if torch.is_tensor(value):
        moved = value.to(
            device,
            non_blocking=device.type == "cuda" and value.is_pinned(),
        )
        if pin_memory and not moved.is_pinned():
            moved = moved.pin_memory()
        return moved
    if isinstance(value, dict):
        return {
            key: _move_optimizer_state_value(
                item,
                device,
                pin_memory=pin_memory,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _move_optimizer_state_value(item, device, pin_memory=pin_memory)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _move_optimizer_state_value(item, device, pin_memory=pin_memory)
            for item in value
        )
    return value


def enable_optimizer_state_offload(
    optimizer: Optimizer,
    *,
    pin_memory: bool = True,
) -> OptimizerStateOffloader:
    """Install state movement hooks without changing optimizer identity."""
    existing = getattr(optimizer, "_wavelet_state_offloader", None)
    if existing is not None:
        raise RuntimeError("Optimizer state offload is already enabled.")
    offloader = OptimizerStateOffloader(optimizer, pin_memory=pin_memory)
    offloader.install()
    optimizer._wavelet_state_offloader = offloader
    return offloader


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
    if config.type == "sign_sgd":
        return SignSGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
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
    if scheduler_config.decay_steps is not None:
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
    # Decay may not start before warmup ends; clamp so the phase is attached
    # instead of silently dropped (which would hold the LR at peak forever).
    if total_steps - warmup_steps > 0:
        decay_steps = min(decay_steps, total_steps - warmup_steps)

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
    # The floor must agree with the warmup start and with the other schedulers,
    # which all express the minimum as min_lr_factor * lr.
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=max(min_lr, min_lr_factor * lr),
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

    post_warmup_span = max(total_steps - warmup_steps - 1, 1)
    decay_span = min(decay_steps, post_warmup_span)
    # Hold the peak LR until the final ``decay_span`` steps, matching the
    # constant-then-decay phases of the linear scheduler.
    constant_span = post_warmup_span - decay_span

    def _sqrt_factor(step: int) -> float:
        decay_step = step - constant_span
        if decay_step <= 0:
            return 1.0
        t = min(decay_step / decay_span, 1.0)
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

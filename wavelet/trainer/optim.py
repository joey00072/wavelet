from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from wavelet.configs.sft import OptimizerConfig


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

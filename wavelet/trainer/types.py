from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(slots=True)
class LossOutput:
    """Differentiable loss and tensor metrics produced by a loss function."""

    loss: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class TrainOutput:
    """Result of one trainer micro-batch."""

    loss: LossOutput
    stepped: bool
    metrics: dict[str, float] = field(default_factory=dict)


LORA_STATE_ATTRS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def lora_adapter_name_from_key(key: str) -> str | None:
    """Return the adapter name embedded in a LoRA parameter/state-dict key."""
    for attr in LORA_STATE_ATTRS:
        marker = f".{attr}."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        if "." not in suffix:
            return None
        return suffix.split(".", 1)[0]
    return None

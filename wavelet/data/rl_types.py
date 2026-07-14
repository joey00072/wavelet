from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict

from torch import Tensor


class RLSample(TypedDict):
    """One tokenized RL example before tensor collation."""

    input_ids: list[int]
    position_ids: list[int]
    target_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: NotRequired[list[float]]
    teacher_logprobs: NotRequired[list[float]]
    temperatures: list[float]
    reward: float | None
    sample_count: NotRequired[int]


class RLBatch(TypedDict):
    """Padded tensors consumed by the RL trainer."""

    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    target_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    advantages: Tensor
    rewards: Tensor
    has_inference_logprobs: Tensor
    inference_logprobs: Tensor
    has_teacher_logprobs: Tensor
    teacher_logprobs: Tensor
    temperatures: Tensor
    sample_counts: Tensor


@dataclass
class RLExample:
    """Serializable rollout record shared by orchestration and training."""

    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    advantage: float | list[float] | None
    reward: float | None
    input_ids: list[int] | None = None
    target_ids: list[int] | None = None
    loss_mask: list[bool] | None = None
    target_completion: list[dict[str, str]] | None = None
    inference_logprobs: list[float] | None = None
    teacher_logprobs: list[float] | None = None
    temperatures: float | list[float] | None = None
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    source: str = "dataset"


def rl_example_to_payload(record: RLExample) -> dict[str, Any]:
    """Convert one rollout record to its JSON-compatible wire payload."""
    return asdict(record)


def rl_example_from_payload(payload: dict[str, Any]) -> RLExample:
    """Construct one rollout record from its wire payload."""
    return RLExample(
        prompt=payload["prompt"],
        completion=payload["completion"],
        advantage=payload.get("advantage"),
        reward=payload.get("reward"),
        input_ids=payload.get("input_ids"),
        target_ids=payload.get("target_ids"),
        loss_mask=payload.get("loss_mask"),
        target_completion=payload.get("target_completion"),
        inference_logprobs=payload.get("inference_logprobs"),
        teacher_logprobs=payload.get("teacher_logprobs"),
        temperatures=payload.get("temperatures"),
        tools=payload.get("tools"),
        chat_template_kwargs=payload.get("chat_template_kwargs"),
        metadata=payload.get("metadata"),
        source=payload.get("source") or "dataset",
    )


def rl_examples_to_payload(records: list[RLExample]) -> list[dict[str, Any]]:
    """Convert rollout records to wire payloads."""
    return [rl_example_to_payload(record) for record in records]


def rl_examples_from_payload(payloads: list[dict[str, Any]]) -> list[RLExample]:
    """Construct rollout records from wire payloads."""
    return [rl_example_from_payload(payload) for payload in payloads]

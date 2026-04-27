from __future__ import annotations

from dataclasses import asdict
from typing import Any

from wavelet.data.rl_dataset import RLExample


def rl_example_to_payload(record: RLExample) -> dict[str, Any]:
    return asdict(record)


def rl_example_from_payload(payload: dict[str, Any]) -> RLExample:
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
    return [rl_example_to_payload(record) for record in records]


def rl_examples_from_payload(payloads: list[dict[str, Any]]) -> list[RLExample]:
    return [rl_example_from_payload(payload) for payload in payloads]

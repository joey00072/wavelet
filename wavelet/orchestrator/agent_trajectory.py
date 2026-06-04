from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenSegment:
    prompt_ids: list[int]
    output_ids: list[int]
    output_logprobs: list[float]
    prompt_loss_mask: list[bool] | None = None
    output_loss_mask: list[bool] | None = None
    turn_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        prompt_ids = [int(token_id) for token_id in self.prompt_ids]
        output_ids = [int(token_id) for token_id in self.output_ids]
        output_logprobs = [float(value) for value in self.output_logprobs]
        prompt_loss_mask = _coerce_mask(
            self.prompt_loss_mask,
            length=len(prompt_ids),
            default=False,
            field_name="prompt_loss_mask",
        )
        output_loss_mask = _coerce_mask(
            self.output_loss_mask,
            length=len(output_ids),
            default=True,
            field_name="output_loss_mask",
        )
        if len(output_logprobs) != len(output_ids):
            raise ValueError(
                "output_logprobs must align with output_ids "
                f"({len(output_logprobs)} != {len(output_ids)})."
            )
        object.__setattr__(self, "prompt_ids", prompt_ids)
        object.__setattr__(self, "output_ids", output_ids)
        object.__setattr__(self, "output_logprobs", output_logprobs)
        object.__setattr__(self, "prompt_loss_mask", prompt_loss_mask)
        object.__setattr__(self, "output_loss_mask", output_loss_mask)


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    input_ids: list[int]
    target_ids: list[int]
    loss_mask: list[bool]
    inference_logprobs: list[float]
    temperatures: list[float]
    turn_ids: list[str | None]

    def as_dict(self) -> dict[str, list[Any]]:
        return {
            "input_ids": self.input_ids,
            "target_ids": self.target_ids,
            "loss_mask": self.loss_mask,
            "inference_logprobs": self.inference_logprobs,
            "temperatures": self.temperatures,
            "turn_ids": self.turn_ids,
        }


@dataclass(slots=True)
class _ActiveSample:
    prefix_ids: list[int]
    sample: TrajectorySample


def merge_token_segments(
    segments: list[TokenSegment],
    *,
    temperature: float = 1.0,
    mask_outputs: bool = False,
) -> list[TrajectorySample]:
    """Merge tokenized agent turns when each prompt exactly extends prior tokens."""
    active: list[_ActiveSample] = []
    for segment in segments:
        for active_index, item in enumerate(active):
            prefix_len = len(item.prefix_ids)
            if segment.prompt_ids[:prefix_len] != item.prefix_ids:
                continue
            active[active_index] = _ActiveSample(
                prefix_ids=segment.prompt_ids + segment.output_ids,
                sample=_extend_sample(
                    item.sample,
                    segment,
                    prefix_len=prefix_len,
                    temperature=temperature,
                    mask_outputs=mask_outputs,
                ),
            )
            break
        else:
            active.append(
                _ActiveSample(
                    prefix_ids=segment.prompt_ids + segment.output_ids,
                    sample=_sample_from_segment(
                        segment,
                        temperature=temperature,
                        mask_outputs=mask_outputs,
                    ),
                )
            )
    return [item.sample for item in active if item.sample.input_ids]


def _sample_from_segment(
    segment: TokenSegment,
    *,
    temperature: float,
    mask_outputs: bool,
) -> TrajectorySample:
    output_mask = _output_mask(segment, mask_outputs=mask_outputs)
    token_ids = segment.prompt_ids + segment.output_ids
    token_mask = list(segment.prompt_loss_mask or []) + output_mask
    logprobs = [0.0] * len(segment.prompt_ids) + segment.output_logprobs
    turn_ids = [segment.turn_id] * len(token_ids)
    return _shift_tokens(
        token_ids,
        token_mask,
        logprobs,
        turn_ids,
        temperature=temperature,
    )


def _extend_sample(
    sample: TrajectorySample,
    segment: TokenSegment,
    *,
    prefix_len: int,
    temperature: float,
    mask_outputs: bool,
) -> TrajectorySample:
    prefix_ids = segment.prompt_ids[:prefix_len]
    new_prompt_ids = segment.prompt_ids[prefix_len:]
    output_mask = _output_mask(segment, mask_outputs=mask_outputs)
    extension_ids = new_prompt_ids + segment.output_ids
    extension_mask = [False] * len(new_prompt_ids) + output_mask
    extension_logprobs = [0.0] * len(new_prompt_ids) + segment.output_logprobs
    extension_turn_ids = [segment.turn_id] * len(extension_ids)

    input_ids = list(sample.input_ids)
    target_ids = list(sample.target_ids)
    loss_mask = list(sample.loss_mask)
    inference_logprobs = list(sample.inference_logprobs)
    temperatures = list(sample.temperatures)
    turn_ids = list(sample.turn_ids)

    if prefix_ids and extension_ids:
        input_ids.append(prefix_ids[-1])
        target_ids.append(extension_ids[0])
        loss_mask.append(extension_mask[0])
        inference_logprobs.append(extension_logprobs[0])
        temperatures.append(temperature)
        turn_ids.append(segment.turn_id)

    extension = _shift_tokens(
        extension_ids,
        extension_mask,
        extension_logprobs,
        extension_turn_ids,
        temperature=temperature,
    )
    input_ids.extend(extension.input_ids)
    target_ids.extend(extension.target_ids)
    loss_mask.extend(extension.loss_mask)
    inference_logprobs.extend(extension.inference_logprobs)
    temperatures.extend(extension.temperatures)
    turn_ids.extend(extension.turn_ids)
    return TrajectorySample(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        inference_logprobs=inference_logprobs,
        temperatures=temperatures,
        turn_ids=turn_ids,
    )


def _shift_tokens(
    token_ids: list[int],
    loss_mask: list[bool],
    logprobs: list[float],
    turn_ids: list[str | None],
    *,
    temperature: float,
) -> TrajectorySample:
    if not (
        len(token_ids) == len(loss_mask) == len(logprobs) == len(turn_ids)
    ):
        raise ValueError("Token ids, masks, logprobs, and turn ids must align.")
    if len(token_ids) < 2:
        return TrajectorySample([], [], [], [], [], [])
    return TrajectorySample(
        input_ids=token_ids[:-1],
        target_ids=token_ids[1:],
        loss_mask=loss_mask[1:],
        inference_logprobs=logprobs[1:],
        temperatures=[temperature] * (len(token_ids) - 1),
        turn_ids=turn_ids[1:],
    )


def _output_mask(segment: TokenSegment, *, mask_outputs: bool) -> list[bool]:
    if mask_outputs:
        return [False] * len(segment.output_ids)
    return list(segment.output_loss_mask or [])


def _coerce_mask(
    values: list[bool] | None,
    *,
    length: int,
    default: bool,
    field_name: str,
) -> list[bool]:
    if values is None:
        return [default] * length
    if len(values) != length:
        raise ValueError(f"{field_name} length must be {length}, got {len(values)}.")
    return [bool(value) for value in values]

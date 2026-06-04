from __future__ import annotations

import logging
from typing import Any, TypedDict

from transformers import PreTrainedTokenizerBase

from wavelet.configs.sft import LossMaskConfig
from wavelet.data.loading import Example


logger = logging.getLogger(__name__)


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


def _token_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if hasattr(value, "input_ids"):
        input_ids = getattr(value, "input_ids")
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(item) for item in input_ids[0]]
            return [int(item) for item in input_ids]
    if hasattr(value, "ids"):
        return [int(item) for item in value.ids]
    raise TypeError(f"Unsupported tokenized value: {type(value)!r}")


def _should_mask(role: str, loss_mask_config: LossMaskConfig) -> bool:
    match role:
        case "system":
            return loss_mask_config.system
        case "user":
            return loss_mask_config.user
        case "assistant":
            return loss_mask_config.assistant
        case "tool":
            return loss_mask_config.tool
        case _:
            raise ValueError(f"Unsupported message role: {role}")


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template_kwargs is not None:
        kwargs.update(chat_template_kwargs)
    return _token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _build_loss_mask_fast(
    tokenizer: PreTrainedTokenizerBase,
    record: Example,
    loss_mask_config: LossMaskConfig,
) -> tuple[list[int], list[bool]] | None:
    """Fast path using return_assistant_tokens_mask (one tokenizer call).

    Only applicable for the default config (assistant=True, all others False).
    Returns None if the chat template doesn't support it, falling back to the
    incremental approach.
    """
    if not (
        loss_mask_config.assistant
        and not loss_mask_config.system
        and not loss_mask_config.user
        and not loss_mask_config.tool
    ):
        return None

    messages = record.prompt + record.completion
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
    }
    if record.tools is not None:
        kwargs["tools"] = record.tools
    if record.chat_template_kwargs is not None:
        kwargs.update(record.chat_template_kwargs)

    try:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    except ValueError as exc:
        if "char_to_token() is not available" not in str(exc):
            raise
        return None
    raw_mask: list[int] = result.get("assistant_masks", [])  # type: ignore[union-attr]

    # If all zeros the template doesn't implement {% generation %} — fall back
    if not any(raw_mask):
        return None

    full_ids = _token_ids(result)
    loss_mask = [bool(m) for m in raw_mask]
    return full_ids, loss_mask


def _build_loss_mask(
    tokenizer: PreTrainedTokenizerBase,
    record: Example,
    loss_mask_config: LossMaskConfig,
) -> tuple[list[int], list[bool]]:
    """Incremental tokenization: apply the chat template to messages[0..i] at
    each step and measure the token delta since the previous step.

    This naturally handles Qwen3 and other models where a message's token
    representation depends on its position in the conversation (e.g. whether a
    system prompt is prepended, or whether thinking tokens are inserted). We
    avoid tokenizing turns in isolation so the rendered prefix remains exact."""
    messages = record.prompt + record.completion
    loss_mask: list[bool] = []
    prev_ids: list[int] = []
    prev_len = 0
    for index, message in enumerate(messages):
        # Parallel tool responses (e.g., qwen3 etc)
        if (
            message["role"] == "tool"
            and index + 1 < len(messages)
            and messages[index + 1]["role"] == "tool"
        ):
            continue
        add_generation_prompt = (
            message["role"] in {"user", "tool"}
            and index + 1 < len(messages)
            and messages[index + 1]["role"] == "assistant"
        )
        current_ids = _apply_chat_template(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=add_generation_prompt,
            tools=record.tools,
            chat_template_kwargs=record.chat_template_kwargs,
        )
        if prev_ids and prev_ids != current_ids[:prev_len]:
            prefilled = _maybe_append_assistant_prefill_delta(
                tokenizer,
                messages,
                index=index,
                prev_ids=prev_ids,
                current_ids=current_ids,
                loss_mask=loss_mask,
                loss_mask_config=loss_mask_config,
                tools=record.tools,
                chat_template_kwargs=record.chat_template_kwargs,
            )
            if prefilled is None:
                raise ValueError(
                    "Chat template incremental tokenization mismatch at "
                    f"message index {index}."
                )
            prev_ids, prev_len = prefilled
            continue
        # Per-turn override: if the message carries step_loss_mask=0, mask the
        # whole turn regardless of role. This is useful for curriculum learning
        # or filtering bad turns in multi-turn data.
        role_mask = _should_mask(message["role"], loss_mask_config)
        turn_mask = bool(message.get("step_loss_mask", 1)) and role_mask
        loss_mask.extend([turn_mask] * (len(current_ids) - prev_len))
        prev_ids = current_ids
        prev_len = len(current_ids)
    return prev_ids, loss_mask


def _maybe_append_assistant_prefill_delta(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    index: int,
    prev_ids: list[int],
    current_ids: list[int],
    loss_mask: list[bool],
    loss_mask_config: LossMaskConfig,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[list[int], int] | None:
    """Handle templates whose generation prompt pre-fills assistant text.

    Unsloth's GRPO template appends ``<start_working_out>`` when
    ``add_generation_prompt=True``. vLLM then returns only tokens after that
    prefill, so the full assistant message no longer has ``prev_ids`` as a
    prefix. In that case we keep the prefilled prompt tokens and append only the
    assistant-message delta from the non-prefilled prompt.
    """
    if messages[index]["role"] != "assistant":
        return None

    prompt_ids = _apply_chat_template(
        tokenizer,
        messages[:index],
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_len = len(prompt_ids)
    prompt_text = tokenizer.decode(prompt_ids)
    if not tokenizer.decode(prev_ids).startswith(prompt_text):
        return None

    if current_ids[:prompt_len] == prompt_ids:
        delta = current_ids[prompt_len:]
    else:
        current_text = tokenizer.decode(current_ids)
        if not current_text.startswith(prompt_text):
            return None
        delta = _token_ids(
            tokenizer(
                current_text[len(prompt_text) :],
                add_special_tokens=False,
            )
        )
    if not delta:
        return None
    turn_mask = bool(messages[index].get("step_loss_mask", 1)) and _should_mask(
        messages[index]["role"],
        loss_mask_config,
    )
    loss_mask.extend([turn_mask] * len(delta))
    full_ids = prev_ids + delta
    return full_ids, len(full_ids)


def build_sample(
    record: Example,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seq_len: int,
    loss_mask_config: LossMaskConfig,
) -> Sample | None:
    result = _build_loss_mask_fast(tokenizer, record, loss_mask_config)
    if result is None:
        result = _build_loss_mask(tokenizer, record, loss_mask_config)
    full_ids, loss_mask = result

    if tokenizer.eos_token_id not in full_ids:
        logger.warning(
            f"EOS token id {tokenizer.eos_token_id} not found in sample. "
            "Is something wrong with the chat template? Appending EOS token."
        )
        full_ids.append(tokenizer.eos_token_id)
        loss_mask.append(True)

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    loss_mask = loss_mask[1:]

    input_ids = input_ids[:seq_len]
    target_ids = target_ids[:seq_len]
    loss_mask = loss_mask[:seq_len]

    if sum(loss_mask) == 0:
        logger.warning(
            "Skipping sample: no trainable tokens found within the context "
            f"window ({seq_len}). This prevents NaN loss."
        )
        return None

    assert len(input_ids) == len(loss_mask) == len(target_ids), (
        f"Length mismatch: {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
    )
    assert tokenizer.eos_token_id in target_ids, (
        "EOS token id must be present in target_ids"
    )

    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "target_ids": target_ids,
        "loss_mask": loss_mask,
    }


def trainable_target_ids(sample: Sample) -> list[int]:
    return [
        int(target_id)
        for target_id, trainable in zip(
            sample["target_ids"],
            sample["loss_mask"],
            strict=True,
        )
        if trainable
    ]


def validate_token_logprob_alignment(
    sample: Sample,
    logprobs: list[float],
    *,
    field_name: str = "inference_logprobs",
) -> None:
    trainable_tokens = sum(bool(value) for value in sample["loss_mask"])
    if len(logprobs) != trainable_tokens:
        raise ValueError(
            f"{field_name} must align with trainable tokens "
            f"({len(logprobs)} != {trainable_tokens})."
        )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from datasets import Dataset, interleave_datasets, load_dataset

from wavelet.configs.sft import DataConfig


@dataclass
class Example:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    source: str = "dataset"


class Stats(TypedDict):
    samples: dict[str, int]
    tokens: dict[str, int]
    skipped: int


def _coerce_messages(value: Any, role: str | None) -> list[dict[str, str]]:
    if isinstance(value, str):
        if role is None:
            raise ValueError("String messages require an explicit role.")
        return [{"role": role, "content": value}]
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Message items must be objects.")
            if "role" not in item and role is None:
                raise ValueError("Message items must include a role.")
            messages.append(
                {
                    **item,
                    "role": str(item.get("role", role)),
                    "content": str(item["content"]),
                }
            )
        return messages
    raise ValueError("Prompt/completion must be either strings or message lists.")


def _deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            normalized.append(message)
            continue
        normalized_tool_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                normalized_tool_calls.append(tool_call)
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(
                function.get("arguments"), str
            ):
                try:
                    arguments: object = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    arguments = function["arguments"]
                normalized_tool_calls.append(
                    {
                        **tool_call,
                        "function": {
                            **function,
                            "arguments": arguments,
                        },
                    }
                )
            else:
                normalized_tool_calls.append(tool_call)
        normalized.append({**message, "tool_calls": normalized_tool_calls})
    return normalized


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            stripped.append({**message, "content": content.strip()})
        else:
            stripped.append(message)
    return stripped


def _merge_message_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        thinking = message.get("thinking")
        content = message.get("content")
        if isinstance(thinking, str) and thinking.strip() and isinstance(content, str):
            merged.append(
                {
                    **message,
                    "content": f"<think>\n{thinking.strip()}\n</think>\n{content}",
                }
            )
        else:
            merged.append(message)
    return merged


def _split_messages_for_sft(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("messages rows must include at least one assistant message.")
    split_at = assistant_indices[-1]
    prompt = messages[:split_at]
    completion = messages[split_at:]
    if not completion:
        raise ValueError("messages rows must produce a non-empty completion.")
    return prompt, completion


def _prepend_system_prompt(
    messages: list[dict[str, str]],
    system_prompt: str | None,
) -> list[dict[str, str]]:
    if not system_prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def _paths(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, list):
        return [Path(path) for path in value]
    return [Path(value)]


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected {path} to contain a JSON list.")
        return [{**dict(row), "__source": path.name} for row in payload]

    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payloads.append({**dict(json.loads(stripped)), "__source": path.name})
    return payloads


def _load_local_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    return [_load_payloads(path) for path in _paths(config.path)]


def _load_hf_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    subsets_and_splits = _hf_subsets_and_splits(config)
    payload_groups: list[list[dict[str, Any]]] = []
    for subset, split in subsets_and_splits:
        dataset = load_dataset(config.hf_name, subset, split=split)
        payload_groups.append([dict(row) for row in dataset])
    return payload_groups


def _load_fake_payload_groups(config: DataConfig) -> list[list[dict[str, Any]]]:
    prompt = " ".join(f"tok{i}" for i in range(max(1, config.seq_len // 2)))
    completion = " ".join(
        f"tok{i}"
        for i in range(
            max(1, config.seq_len // 2),
            max(2, config.seq_len),
        )
    )
    payloads = [
        {
            config.prompt_column: prompt,
            config.completion_column: completion,
            "__source": "fake",
        }
        for _ in range(config.max_examples or config.batch_size)
    ]
    return [payloads]


def _hf_subsets_and_splits(
    config: DataConfig,
) -> list[tuple[str | None, str]]:
    if config.hf_subsets is None and config.hf_splits is None:
        return [(None, "train")]
    if config.hf_subsets is not None and config.hf_splits is None:
        return [(subset, "train") for subset in config.hf_subsets]
    if config.hf_subsets is None and config.hf_splits is not None:
        return [(None, split) for split in config.hf_splits]
    assert config.hf_subsets is not None and config.hf_splits is not None
    return list(zip(config.hf_subsets, config.hf_splits, strict=True))


def _mix_payload_groups(
    payload_groups: list[list[dict[str, Any]]],
    *,
    probabilities: list[float] | None,
    stopping_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not payload_groups:
        return []
    if len(payload_groups) == 1:
        return payload_groups[0]
    if probabilities is not None and len(probabilities) != len(payload_groups):
        raise ValueError("probabilities must match the number of dataset groups")

    datasets_list = []
    for index, payloads in enumerate(payload_groups):
        if not payloads:
            continue
        dataset = Dataset.from_list(payloads)
        datasets_list.append(dataset)
    if not datasets_list:
        return []
    mixed = interleave_datasets(
        datasets_list,
        probabilities=probabilities,
        stopping_strategy=stopping_strategy,
        seed=seed,
    )
    return [dict(row) for row in mixed]


def _normalize_record(payload: dict[str, Any], config: DataConfig) -> Example:
    tools = payload.get(config.tools_column)
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be a list when provided.")

    chat_template_kwargs = payload.get(config.chat_template_kwargs_column)
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object when provided.")

    has_messages = (
        config.messages_column in payload
        and payload.get(config.messages_column) is not None
    )
    has_prompt_completion = (
        config.prompt_column in payload
        and config.completion_column in payload
        and payload.get(config.prompt_column) is not None
        and payload.get(config.completion_column) is not None
    )

    if has_messages and not has_prompt_completion:
        messages = _coerce_messages(payload[config.messages_column], None)
        messages = _strip_message_content(_deserialize_tool_calls(messages))
        if config.merge_messages_thinking:
            messages = _merge_message_thinking(messages)
        prompt, completion = _split_messages_for_sft(messages)
    else:
        if not has_prompt_completion:
            raise ValueError(
                "Each row must contain either prompt/completion columns or a "
                "messages column."
            )
        prompt = _strip_message_content(
            _deserialize_tool_calls(
                _coerce_messages(payload[config.prompt_column], "user")
            )
        )
        prompt = _prepend_system_prompt(prompt, config.system_prompt)
        completion = _strip_message_content(
            _deserialize_tool_calls(
                _coerce_messages(payload[config.completion_column], "assistant")
            )
        )

    return Example(
        prompt=prompt,
        completion=completion,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        source=str(
            payload.get("__subset")
            or payload.get("__split")
            or payload.get("__source")
            or config.source
        ),
    )


def load_records(config: DataConfig) -> list[Example]:
    if config.source == "hf":
        payload_groups = _load_hf_payload_groups(config)
    elif config.source == "fake":
        payload_groups = _load_fake_payload_groups(config)
    else:
        payload_groups = _load_local_payload_groups(config)
    payloads = _mix_payload_groups(
        payload_groups,
        probabilities=config.probabilities,
        stopping_strategy=config.stopping_strategy,
        seed=config.seed,
    )
    rows = [_normalize_record(payload, config) for payload in payloads]
    if config.max_examples is not None:
        rows = rows[: config.max_examples]
    if not rows:
        raise ValueError("No training rows found for the configured data source.")
    return rows

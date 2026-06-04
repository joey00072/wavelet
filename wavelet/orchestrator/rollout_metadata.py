from __future__ import annotations

from typing import Any


def rollout_task_harness_metadata(
    output: dict[str, Any],
    *,
    group_key: str,
    sample_index: int,
) -> dict[str, Any]:
    task_name = str(output.get("env_name") or output.get("task") or "rollout")
    example_id = output.get("example_id")
    harness_name = str(output.get("harness_name") or task_name)
    harness_type = str(output.get("harness_type") or "environment")
    rollout_key = f"{group_key}:{sample_index}"
    trajectory = output.get("trajectory")
    trajectory = trajectory if isinstance(trajectory, list) else []
    return {
        "task": {
            "name": task_name,
            "example_id": example_id,
        },
        "harness": {
            "name": harness_name,
            "type": harness_type,
            "version": output.get("harness_version"),
        },
        "rollout": {
            "group_key": group_key,
            "rollout_key": rollout_key,
            "sample_index": sample_index,
            "trajectory_id": output.get("trajectory_id") or rollout_key,
            "num_turns": len(trajectory),
            "tool_calls": _tool_call_count(trajectory),
            "elapsed_sec": _float_or_none(
                output.get("elapsed_sec", output.get("elapsed_seconds"))
            ),
            "stop_condition": output.get("stop_condition"),
            "is_truncated": output.get("is_truncated"),
            "error": output.get("error"),
            "reward_components": _reward_components(output),
        },
    }


def metadata_task_name(metadata: dict[str, Any]) -> str | None:
    task = metadata.get("task")
    if isinstance(task, dict) and task.get("name") is not None:
        return str(task["name"])
    return None


def metadata_harness_name(metadata: dict[str, Any]) -> str | None:
    harness = metadata.get("harness")
    if isinstance(harness, dict) and harness.get("name") is not None:
        return str(harness["name"])
    return None


def _tool_call_count(trajectory: list[Any]) -> int:
    count = 0
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        count += _message_tool_call_count(step.get("prompt"))
        count += _message_tool_call_count(step.get("completion"))
        count += _message_tool_call_count(step.get("messages"))
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _message_tool_call_count(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reward_components(output: dict[str, Any]) -> dict[str, Any] | None:
    value = output.get("reward_components")
    if isinstance(value, dict):
        return dict(value)
    metrics = output.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("reward_components")
    if isinstance(value, dict):
        return dict(value)
    return None

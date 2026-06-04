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
            "stop_condition": output.get("stop_condition"),
            "is_truncated": output.get("is_truncated"),
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

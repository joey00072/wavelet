"""Convert native Verifiers message graphs without retokenizing model output."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sample_digest(samples: list[dict[str, Any]]) -> str:
    """Fingerprint shifted training contexts, masks, and sampled logprobs."""
    encoded = []
    for sample in samples:
        mask = sample["loss_mask"]
        encoded.append(
            json.dumps(
                [
                    sample["input_ids"],
                    sample["target_ids"],
                    mask,
                    [
                        float(value)
                        for value, trainable in zip(
                            sample["inference_logprobs"], mask, strict=True
                        )
                        if trainable
                    ],
                ],
                separators=(",", ":"),
            )
        )
    return hashlib.sha256("\n".join(sorted(encoded)).encode()).hexdigest()


def trace_to_output(trace: dict[str, Any], *, reward: float) -> dict[str, Any]:
    """Keep each sampled node's exact physical prompt and completion spans."""
    nodes = trace["nodes"]
    trajectory = []
    completions = []
    for index, node in enumerate(nodes):
        if node.get("sampled"):
            completions.append(node["message"])
        if not node.get("sampled") or not any(node["mask"]):
            continue
        if len(node["token_ids"]) != len(node["mask"]):
            raise ValueError("Native node tokens and mask must align.")
        ancestors = []
        parent = node.get("parent")
        visited = {index}
        while parent is not None:
            if parent in visited or not 0 <= parent < len(nodes):
                raise ValueError("Invalid native message parent graph.")
            visited.add(parent)
            ancestor = nodes[parent]
            ancestors.append(ancestor)
            parent = ancestor.get("parent")
        ancestors.reverse()
        start = node["mask"].index(True)
        prompt_ids = [token for item in ancestors for token in item["token_ids"]]
        prompt_ids.extend(node["token_ids"][:start])
        sampled_logprobs = iter(node["logprobs"])
        if len(node["logprobs"]) != sum(node["mask"]):
            raise ValueError("Native sampled logprobs must align with the mask.")
        logprobs = [next(sampled_logprobs) if flag else 0.0 for flag in node["mask"]]
        completion = node["message"]
        trajectory.append(
            {
                "prompt": [item["message"] for item in ancestors],
                "completion": [completion],
                "turn_id": str(index),
                "tokens": {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": [False] * len(prompt_ids),
                    "completion_ids": node["token_ids"][start:],
                    "completion_mask": node["mask"][start:],
                    "completion_logprobs": logprobs[start:],
                },
            }
        )
    return {
        "reward": reward,
        "trajectory": trajectory,
        "completion": completions,
        "error": None if trace["ok"] else trace.get("errors") or "NativeTraceFailed",
        "stop_condition": trace.get("stop_condition"),
        "is_truncated": any(
            call.get("finish_reason") == "length" for call in trace["calls"]
        ),
        "trajectory_id": trace["id"],
    }

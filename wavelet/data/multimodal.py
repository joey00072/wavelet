"""Explicit processor tensor collation shared by SFT and RL."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

VISION_FIELDS = {
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_sizes",
    "video_sizes",
    "second_per_grid_ts",
}
PROCESSOR_FIELDS = VISION_FIELDS | {"mm_token_type_ids"}


def collate_multimodal_fields(
    samples: list[dict[str, Any]], max_len: int
) -> dict[str, Tensor]:
    keys = {key for sample in samples for key in (sample.get("mm_kwargs") or {})}
    unknown = keys - PROCESSOR_FIELDS
    if unknown:
        raise ValueError(f"Unsupported processor fields: {sorted(unknown)}")
    result = {}
    for key in keys:
        values = [(sample.get("mm_kwargs") or {}).get(key) for sample in samples]
        if any(value is None for value in values):
            raise ValueError(
                f"All samples in a batch must provide multimodal field {key!r}"
            )
        tensors = [torch.as_tensor(value) for value in values]
        if key == "mm_token_type_ids":
            if any(
                t.ndim != 1 or t.numel() != len(sample["input_ids"])
                for t, sample in zip(tensors, samples, strict=True)
            ):
                raise ValueError("mm_token_type_ids must align with sample input_ids")
            result[key] = torch.stack(
                [torch.nn.functional.pad(t, (0, max_len - t.numel())) for t in tensors]
            )
        else:
            try:
                result[key] = torch.cat(tensors, dim=0)
            except RuntimeError as exc:
                raise ValueError(
                    f"Multimodal field {key!r} has incompatible shapes"
                ) from exc
    return result

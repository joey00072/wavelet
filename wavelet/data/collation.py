from __future__ import annotations

import torch

from wavelet.data.batch import SFTBatch
from wavelet.data.tokenization import Sample

IGNORE_INDEX = -100


def collate_batch(
    batch: list[Sample],
    *,
    pad_token_id: int,
    include_attention_mask: bool = True,
) -> SFTBatch:
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_out = []
    attention_mask_out = []
    position_ids_out = []
    labels_out = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids_out.append(
            torch.tensor(
                item["input_ids"] + [pad_token_id] * pad_len,
                dtype=torch.long,
            )
        )
        if include_attention_mask:
            attention_mask_out.append(
                torch.tensor(
                    [1] * seq_len + [0] * pad_len,
                    dtype=torch.long,
                )
            )
        position_ids_out.append(
            torch.tensor(
                item["position_ids"] + list(range(seq_len, max_len)),
                dtype=torch.long,
            )
        )
        # Merge target_ids and loss_mask into labels with IGNORE_INDEX (-100).
        # Positions that don't contribute to the loss (role-masked or padding)
        # are set to -100 so CrossEntropyLoss(ignore_index=-100) skips them
        # automatically
        labels = [
            tid if mask else IGNORE_INDEX
            for tid, mask in zip(item["target_ids"], item["loss_mask"])
        ] + [IGNORE_INDEX] * pad_len
        labels_out.append(torch.tensor(labels, dtype=torch.long))

    return SFTBatch(
        input_ids=torch.stack(input_ids_out),
        position_ids=torch.stack(position_ids_out),
        labels=torch.stack(labels_out),
        attention_mask=(
            torch.stack(attention_mask_out) if include_attention_mask else None
        ),
    )

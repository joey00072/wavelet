from __future__ import annotations

from wavelet.data.rl_types import RLSample


def trainable_token_count(sample: RLSample) -> int:
    """Return the number of tokens that contribute to the RL loss."""
    return sum(bool(value) for value in sample["loss_mask"])


def trainable_sequence_count(sample: RLSample) -> int:
    """Return the number of packed sequences that contribute to the RL loss."""
    loss_mask = sample["loss_mask"]
    if not any(bool(value) for value in loss_mask):
        return 0

    starts = [
        index
        for index, position_id in enumerate(sample["position_ids"][: len(loss_mask)])
        if position_id == 0
    ]
    if not starts:
        return 1
    if starts[0] != 0:
        starts.insert(0, 0)
    ends = [*starts[1:], len(loss_mask)]
    return sum(
        any(bool(value) for value in loss_mask[start:end])
        for start, end in zip(starts, ends, strict=True)
    )


def pack_samples(
    samples: list[RLSample],
    *,
    seq_len: int,
    pad_to_multiple_of: int,
) -> list[RLSample]:
    """Pack samples with first-fit decreasing bin packing."""
    sorted_samples = sorted(samples, key=lambda sample: -len(sample["input_ids"]))
    bins: list[list[RLSample]] = []
    bin_lengths: list[int] = []
    for sample in sorted_samples:
        sample_len = len(sample["input_ids"])
        for index, current_len in enumerate(bin_lengths):
            if current_len + sample_len <= seq_len:
                bins[index].append(sample)
                bin_lengths[index] += sample_len
                break
        else:
            bins.append([sample])
            bin_lengths.append(sample_len)
    return [
        _merge_samples(items, pad_to_multiple_of=pad_to_multiple_of) for items in bins
    ]


def pad_bins_for_distribution(
    bins: list[RLSample],
    *,
    data_world_size: int,
) -> list[RLSample]:
    """Add zero-loss bins so every data rank receives the same bin count."""
    if data_world_size <= 1 or not bins:
        return bins
    pad_count = (-len(bins)) % data_world_size
    if pad_count == 0:
        return bins
    return [*bins, *(_zero_loss_copy(bins[0]) for _ in range(pad_count))]


def _merge_samples(
    samples: list[RLSample],
    *,
    pad_to_multiple_of: int,
) -> RLSample:
    input_ids: list[int] = []
    target_ids: list[int] = []
    position_ids: list[int] = []
    loss_mask: list[bool] = []
    advantages: list[float] = []
    inference_logprobs: list[float] = []
    teacher_logprobs: list[float] = []
    temperatures: list[float] = []
    rewards = [
        float(sample["reward"]) for sample in samples if sample["reward"] is not None
    ]
    sample_count = sum(int(sample.get("sample_count", 1)) for sample in samples)
    has_inference = all("inference_logprobs" in sample for sample in samples)
    has_teacher = all("teacher_logprobs" in sample for sample in samples)

    for sample in samples:
        input_ids.extend(sample["input_ids"])
        target_ids.extend(sample["target_ids"])
        position_ids.extend(sample["position_ids"])
        loss_mask.extend(sample["loss_mask"])
        advantages.extend(sample["advantages"])
        temperatures.extend(sample["temperatures"])
        if has_inference:
            inference_logprobs.extend(sample["inference_logprobs"])
        if has_teacher:
            teacher_logprobs.extend(sample["teacher_logprobs"])

    _pad_token_streams(
        input_ids,
        target_ids,
        position_ids,
        loss_mask,
        multiple=pad_to_multiple_of,
    )
    packed: RLSample = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "temperatures": temperatures,
        "reward": sum(rewards) / len(rewards) if rewards else None,
        "sample_count": sample_count,
    }
    if has_inference:
        packed["inference_logprobs"] = inference_logprobs
    if has_teacher:
        packed["teacher_logprobs"] = teacher_logprobs
    return packed


def _pad_token_streams(
    input_ids: list[int],
    target_ids: list[int],
    position_ids: list[int],
    loss_mask: list[bool],
    *,
    multiple: int,
) -> None:
    if multiple <= 1:
        return
    padding = (-len(input_ids)) % multiple
    if padding == 0:
        return
    input_ids.extend([1] * padding)
    target_ids.extend([0] * padding)
    position_ids.extend(range(padding))
    loss_mask.extend([False] * padding)


def _zero_loss_copy(source: RLSample) -> RLSample:
    dummy: RLSample = {
        "input_ids": list(source["input_ids"]),
        "position_ids": list(source["position_ids"]),
        "target_ids": list(source["target_ids"]),
        "loss_mask": [False] * len(source["loss_mask"]),
        "advantages": [],
        "temperatures": [],
        "reward": None,
        "sample_count": 0,
    }
    if "inference_logprobs" in source:
        dummy["inference_logprobs"] = []
    if "teacher_logprobs" in source:
        dummy["teacher_logprobs"] = []
    return dummy

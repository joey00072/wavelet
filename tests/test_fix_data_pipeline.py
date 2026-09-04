from __future__ import annotations

import itertools

from wavelet.configs.rl_config import RLDataConfig
from wavelet.configs.sft import LossMaskConfig
from wavelet.data.loading import Example
from wavelet.data.rl import (
    FakeRLDataset,
    PackedRLDataset,
    RLDataset,
    RLExample,
    pack_samples,
    prepare_rl_sample,
)
from wavelet.data.sft import _coerce_messages
from wavelet.data.tokenization import build_sample, trainable_target_ids
from wavelet.trainer.debug import build_debug_tokenizer


def _record(index: int, *, length: int = 4) -> RLExample:
    base = index * 100
    return RLExample(
        prompt=[],
        completion=[],
        advantage=float(index),
        reward=float(index),
        input_ids=list(range(base, base + length)),
        target_ids=list(range(base + 1, base + length + 1)),
        loss_mask=[True] * length,
        inference_logprobs=[-1.0] * length,
        temperatures=[1.0] * length,
    )


def _prompt_only_within_context(index: int, *, prompt_tokens: int) -> RLExample:
    """A row whose two trainable tokens fall beyond ``seq_len``."""
    record = _record(index, length=prompt_tokens + 2)
    record.loss_mask = [False] * prompt_tokens + [True, True]
    record.advantage = [1.0, 1.0]
    record.inference_logprobs = [-1.0, -1.0]
    record.temperatures = [1.0, 1.0]
    return record


# ── pretokenized rows never skip or retokenize ────────────────────────────────


def test_truncated_pretokenized_row_trains_as_zero_loss_instead_of_skipping() -> None:
    records = [_record(0), _prompt_only_within_context(1, prompt_tokens=8), _record(2)]
    dataset = RLDataset(
        records=records,
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(seq_len=8),
    )

    samples = list(itertools.islice(iter(dataset), 3))

    # Every record of the epoch is visited once; the next epoch does not leak in.
    assert [sample["input_ids"][0] for sample in samples] == [0, 100, 200]
    assert sum(samples[1]["loss_mask"]) == 0
    assert samples[1]["advantages"] == []
    assert samples[1]["inference_logprobs"] == []
    assert dataset.skipped == 0


def test_pretokenized_row_without_trainable_tokens_is_not_retokenized() -> None:
    class _Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("pretokenized rows must never be retokenized")

        def __call__(self, *args, **kwargs):
            raise AssertionError("pretokenized rows must never be retokenized")

    sample = prepare_rl_sample(
        _prompt_only_within_context(1, prompt_tokens=8),
        tokenizer=_Tokenizer(),  # type: ignore[arg-type]
        data_config=RLDataConfig(seq_len=8),
        seq_len=8,
    )

    assert sample is not None
    assert len(sample["input_ids"]) == 8
    assert sum(sample["loss_mask"]) == 0


# ── packing keeps optional streams intact ─────────────────────────────────────


def test_pack_samples_never_mixes_rows_with_and_without_logprobs() -> None:
    config = RLDataConfig(seq_len=16)
    with_logprobs = [
        prepare_rl_sample(_record(i), None, config, 16)  # type: ignore[arg-type]
        for i in range(2)
    ]
    without = _record(5)
    without.inference_logprobs = None
    without_logprobs = prepare_rl_sample(without, None, config, 16)  # type: ignore[arg-type]

    bins = pack_samples(
        [*with_logprobs, without_logprobs],  # type: ignore[list-item]
        seq_len=16,
        pad_to_multiple_of=1,
    )

    assert len(bins) == 2
    by_presence = {("inference_logprobs" in packed): packed for packed in bins}
    assert len(by_presence[True]["inference_logprobs"]) == 8
    assert len(by_presence[True]["input_ids"]) == 8
    assert len(by_presence[False]["input_ids"]) == 4


def test_packed_dataset_keeps_only_the_current_epoch_cached() -> None:
    dataset = PackedRLDataset(
        records=[_record(i) for i in range(3)],
        tokenizer=None,  # type: ignore[arg-type]
        seq_len=8,
        data_config=RLDataConfig(seq_len=8, pack_sequences=True),
    )

    list(itertools.islice(iter(dataset), 7))

    assert dataset.epoch >= 2
    assert len(dataset._epoch_bins) == 1
    assert len(dataset._epoch_global_bins) == 1


# ── fake data ─────────────────────────────────────────────────────────────────


def test_fake_rl_dataset_differs_across_data_ranks_at_shifted_steps() -> None:
    def _dataset(seed: int) -> FakeRLDataset:
        return FakeRLDataset(
            seq_len=8,
            vocab_size=1000,
            length_mode="fixed",
            input_mode="random",
            seed=seed,
        )

    rank0_step2 = list(itertools.islice(iter(_dataset(0)), 2))[1]
    rank1_step1 = next(iter(_dataset(1)))

    assert rank0_step2["input_ids"] != rank1_step1["input_ids"]


# ── chat message handling ─────────────────────────────────────────────────────


def test_coerce_messages_accepts_tool_calls_without_content_key() -> None:
    tool_calls = [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]

    messages = _coerce_messages(
        [{"role": "assistant", "tool_calls": tool_calls}],
        None,
    )

    assert messages == [{"role": "assistant", "tool_calls": tool_calls, "content": ""}]


def test_assistant_header_is_never_trainable_regardless_of_previous_role() -> None:
    tokenizer = build_debug_tokenizer(model_max_length=256)

    def _trainable_text(prompt, completion) -> str:
        sample = build_sample(
            Example(prompt=prompt, completion=completion, source="test"),
            tokenizer,
            seq_len=128,
            loss_mask_config=LossMaskConfig(),
        )
        assert sample is not None
        return tokenizer.decode(trainable_target_ids(sample))

    after_system = _trainable_text(
        [{"role": "system", "content": "rules"}],
        [{"role": "assistant", "content": "ok"}],
    )
    after_assistant = _trainable_text(
        [{"role": "user", "content": "hi"}],
        [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
    )

    assert after_system == "ok<eos>"
    assert "<|assistant|>" not in after_assistant
    assert after_assistant.startswith("a<eos>")
    assert after_assistant.endswith("b<eos>")

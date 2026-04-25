from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase

from wavelet.configs.sft import DataConfig, LossMaskConfig
from wavelet.data.loading import Example
from wavelet.data.tokenization import Sample, build_sample


class Batch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor


@dataclass
class SFTDataset(IterableDataset[Sample]):
    def __init__(
        self,
        records: list[Example],
        tokenizer: PreTrainedTokenizerBase,
        *,
        seq_len: int,
        loss_mask_config: LossMaskConfig,
        shuffle: bool = False,
        seed: int = 0,
        data_rank: int = 0,
        data_world_size: int = 1,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.shuffle = shuffle
        self.seed = seed
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.step = 0
        self.epoch = 0
        self.num_samples: dict[str, int] = defaultdict(int)
        self.num_tokens: dict[str, int] = defaultdict(int)
        self.skipped = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "num_samples": dict(self.num_samples),
            "num_tokens": dict(self.num_tokens),
            "skipped": self.skipped,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = int(state_dict["step"])
        self.epoch = int(state_dict["epoch"])
        self.num_samples = defaultdict(int, state_dict.get("num_samples", {}))
        self.num_tokens = defaultdict(int, state_dict.get("num_tokens", {}))
        self.skipped = int(state_dict.get("skipped", 0))

    def stats(self) -> dict[str, Any]:
        return {
            "samples": dict(self.num_samples),
            "tokens": dict(self.num_tokens),
            "skipped": self.skipped,
        }

    def __iter__(self) -> Iterator[Sample]:
        num_examples = len(self.records)
        if num_examples == 0:
            return
        data_rank, data_world_size = self._effective_data_partition()
        while True:
            next_step = self.step + 1
            epoch = (next_step - 1) // num_examples
            if epoch != self.epoch:
                self.epoch = epoch
            sample_index = (next_step - 1) % num_examples
            self.step = next_step
            if (next_step - 1) % data_world_size != data_rank:
                continue

            record_index = self._order_for_epoch(epoch)[sample_index]
            record = self.records[record_index]
            sample = build_sample(
                record,
                self.tokenizer,
                seq_len=self.seq_len,
                loss_mask_config=self.loss_mask_config,
            )

            if sample is None:
                self.skipped += 1
                continue

            source = getattr(record, "source", None) or "dataset"
            self.num_samples[source] += 1
            self.num_tokens[source] += len(sample["input_ids"])
            yield sample

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(order)
        return order

    def _effective_data_partition(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data_rank, self.data_world_size
        return (
            self.data_rank * worker_info.num_workers + worker_info.id,
            self.data_world_size * worker_info.num_workers,
        )


class CatDataset(IterableDataset[Sample]):
    """Concatenative packing: fills seq_len exactly by concatenating samples.

    Zero padding waste. Each yielded chunk uses sequential position IDs
    [0, 1, ..., seq_len-1] so RoPE embeddings are monotonically increasing
    within the context window. This matches TRL's packing behavior.

    For true document-aware attention (no cross-doc leakage), use
    micro_batch_size=1 with Flash Attention 2 — FA2 detects reset
    position IDs and switches to varlen mode automatically.
    """

    def __init__(self, base: SFTDataset, seq_len: int) -> None:
        self.base = base
        self.seq_len = seq_len

    def state_dict(self) -> dict[str, Any]:
        return self.base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base.load_state_dict(state_dict)

    def stats(self) -> dict[str, Any]:
        return self.base.stats()

    def __iter__(self) -> Iterator[Sample]:
        buf_input: list[int] = []
        buf_target: list[int] = []
        buf_mask: list[bool] = []

        for sample in self.base:
            buf_input.extend(sample["input_ids"])
            buf_target.extend(sample["target_ids"])
            buf_mask.extend(sample["loss_mask"])

            while len(buf_input) >= self.seq_len:
                yield {
                    "input_ids": buf_input[: self.seq_len],
                    "target_ids": buf_target[: self.seq_len],
                    "loss_mask": buf_mask[: self.seq_len],
                    "position_ids": list(range(self.seq_len)),
                }
                buf_input = buf_input[self.seq_len :]
                buf_target = buf_target[self.seq_len :]
                buf_mask = buf_mask[self.seq_len :]


def setup_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: DataConfig,
    *,
    data_rank: int,
    data_world_size: int,
) -> SFTDataset | CatDataset:
    from wavelet.data.loading import load_records

    base = SFTDataset(
        load_records(config),
        tokenizer,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        shuffle=config.shuffle,
        seed=config.seed,
        data_rank=data_rank,
        data_world_size=data_world_size,
    )
    if config.pack_function == "cat":
        return CatDataset(base, config.seq_len)
    return base

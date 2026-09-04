from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from torch.utils.data import get_worker_info

RecordT = TypeVar("RecordT")


class StatefulDatasetMixin(Generic[RecordT]):
    records: list[RecordT]
    shuffle: bool
    seed: int
    data_rank: int
    data_world_size: int
    step: int
    epoch: int
    num_samples: defaultdict[str, int]
    num_tokens: defaultdict[str, int]
    skipped: int
    _order_cache: tuple[int, list[int]] | None

    def _initialize_iteration_state(self) -> None:
        self.step = 0
        self.epoch = 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.skipped = 0
        self._order_cache = None

    def state_dict(self) -> dict[str, Any]:
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

    def _record_sample(self, source: str | None, token_count: int) -> None:
        source_name = source or "dataset"
        self.num_samples[source_name] += 1
        self.num_tokens[source_name] += token_count

    def _order_for_epoch(self, epoch: int) -> list[int]:
        # Shuffling is O(N); cache the current epoch so per-sample lookups stay
        # O(1). The cache is derived state and is intentionally not checkpointed.
        cached = getattr(self, "_order_cache", None)
        if cached is not None and cached[0] == epoch:
            return cached[1]
        order = list(range(len(self.records)))
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(order)
        self._order_cache = (epoch, order)
        return order

    def _local_record_indexes(self) -> Iterator[int]:
        record_count = len(self.records)
        if record_count == 0:
            return

        data_rank, data_world_size = self._effective_data_partition()
        while True:
            self.step += 1
            self.epoch = (self.step - 1) // record_count
            if (self.step - 1) % data_world_size != data_rank:
                continue
            position = (self.step - 1) % record_count
            yield self._order_for_epoch(self.epoch)[position]

    def _effective_data_partition(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data_rank, self.data_world_size
        return (
            self.data_rank * worker_info.num_workers + worker_info.id,
            self.data_world_size * worker_info.num_workers,
        )

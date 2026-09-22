"""Small backend-independent weight-transfer interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Protocol

from torch import Tensor

from wavelet.contracts.queue_records import PolicySnapshot


class WeightSender(Protocol):
    supports_lora: ClassVar[bool]
    requires_inference_world_size: ClassVar[bool]

    def connect(self) -> None: ...

    def send(
        self,
        step: int,
        tensors: Iterator[tuple[str, Tensor]],
        *,
        metadata: dict[str, Any],
    ) -> Path: ...

    def close(self) -> None: ...


class WeightReceiver(Protocol):
    def wait_for_step(self, step: int) -> PolicySnapshot: ...

    def available_steps(self) -> list[int]: ...

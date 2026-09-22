"""Backend-neutral inference interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from wavelet.contracts.queue_records import PolicySnapshot
from wavelet.data.rl import RLExample


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    sampling_mask: bool = False
    prompt_logprobs: bool = False
    prefill_score: bool = False
    lora_hot_reload: bool = False
    full_model_reload: bool = False
    nccl_weight_update: bool = False
    sleep_wake: bool = False


class EngineAdmin(Protocol):
    def health(self) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def load_policy(
        self,
        snapshot: PolicySnapshot,
        *,
        adapter_name: str | None = None,
    ) -> dict[str, Any]: ...

    def init_weight_receiver(self, init_info: dict[str, Any]) -> None: ...

    def sleep(self, *, level: int = 1) -> None: ...

    def wake(self, *, tags: list[str] | None = None) -> None: ...


class InferenceEngine(Protocol):
    admin: list[EngineAdmin]
    capabilities: EngineCapabilities

    def setup(self) -> None: ...

    def annotate(self, records: list[RLExample]) -> list[RLExample]: ...

    def base_urls(self) -> list[str]: ...

    def close(self) -> None: ...

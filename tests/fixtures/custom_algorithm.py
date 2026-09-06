from __future__ import annotations

from dataclasses import dataclass, replace

from wavelet.data.rl import RLExample
from wavelet.orchestrator.algorithms import BaseAlgorithm, register_algorithm


@register_algorithm("multiplier")
@dataclass(frozen=True, slots=True)
class MultiplierAlgorithm(BaseAlgorithm):
    multiplier: float

    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        return [
            replace(record, advantage=float(record.reward or 0.0) * self.multiplier)
            for record in records
        ]


@register_algorithm("reward_plus_one")
class RewardPlusOneAlgorithm(BaseAlgorithm):
    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        return [
            replace(record, advantage=float(record.reward or 0.0) + 1.0)
            for record in records
        ]


@register_algorithm("missing_group_hook")
class MissingGroupHook:
    def score_rollout(self, record: RLExample) -> RLExample:
        return record


class UndecoratedAlgorithm(BaseAlgorithm):
    pass


NOT_CALLABLE = 42


@register_algorithm("duplicate")
class FirstDuplicateAlgorithm(BaseAlgorithm):
    pass


@register_algorithm("duplicate")
class SecondDuplicateAlgorithm(BaseAlgorithm):
    pass


@register_algorithm("invalid_rollout_return")
class InvalidRolloutReturnAlgorithm(BaseAlgorithm):
    def score_rollout(self, record: RLExample) -> object:
        return {"record": record}


@register_algorithm("short_group_return")
class ShortGroupReturnAlgorithm(BaseAlgorithm):
    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        return records[:-1]


@register_algorithm("both_hooks")
class BothHooksAlgorithm(BaseAlgorithm):
    def score_rollout(self, record: RLExample) -> RLExample:
        return replace(record, advantage=float(record.reward or 0.0))

    def score_group(self, records: list[RLExample]) -> list[RLExample]:
        return [
            replace(record, advantage=float(record.advantage or 0.0) + 1.0)
            for record in records
        ]


@register_algorithm("offset_factory")
def build_offset_algorithm(*, offset: float) -> BaseAlgorithm:
    @dataclass(frozen=True, slots=True)
    class OffsetAlgorithm(BaseAlgorithm):
        value: float

        def score_rollout(self, record: RLExample) -> RLExample:
            return replace(record, advantage=float(record.reward or 0.0) + self.value)

    return OffsetAlgorithm(value=offset)

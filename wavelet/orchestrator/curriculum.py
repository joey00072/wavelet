"""Stateful task sampling and rollout-group admission for Verifiers RL."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any, Protocol

from wavelet.configs.rl_config import (
    RLAdvRangeGateConfig,
    RLCurriculumConfig,
    RLDifficultyPoolSamplerConfig,
    RLStandardCurriculumSamplerConfig,
)


class Gate(Protocol):
    """Admission policy applied after an entire rollout group is scored."""

    def admit(self, outputs: list[dict[str, Any]]) -> bool: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    def metrics(self) -> dict[str, float]: ...


class Sampler(Protocol):
    """Stateful selector over a finite environment dataset."""

    def next_index(self) -> int: ...

    def observe(self, task_key: str, outputs: list[dict[str, Any]]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    def metrics(self) -> dict[str, float]: ...


class StandardSampler:
    """Cycle a finite dataset with the scheduler's deterministic epoch shuffle."""

    def __init__(
        self,
        task_count: int,
        *,
        seed: int,
        shuffle: bool,
        cursor: int = 0,
    ) -> None:
        if task_count < 1:
            raise ValueError("Standard curriculum sampler requires at least one task.")
        if cursor < 0:
            raise ValueError("Curriculum sampler cursor cannot be negative.")
        self.task_count = task_count
        self.seed = seed
        self.shuffle = shuffle
        self.cursor = cursor
        self._order_epoch: int | None = None
        self._order: list[int] = []

    def next_index(self) -> int:
        epoch, offset = divmod(self.cursor, self.task_count)
        if self._order_epoch != epoch:
            self._order = list(range(self.task_count))
            if self.shuffle:
                rng = random.Random(self.seed + epoch * 1_000_003)
                rng.shuffle(self._order)
            self._order_epoch = epoch
        index = self._order[offset]
        self.cursor += 1
        return index

    def observe(self, task_key: str, outputs: list[dict[str, Any]]) -> None:
        del task_key, outputs

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "task_count": self.task_count}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _require_fields(state_dict, {"cursor", "task_count"}, owner="standard sampler")
        task_count = _nonnegative_int(state_dict["task_count"], name="task_count")
        if task_count != self.task_count:
            raise ValueError(
                "Curriculum task count changed across resume: "
                f"expected {self.task_count}, found {task_count}."
            )
        self.cursor = _nonnegative_int(state_dict["cursor"], name="cursor")
        self._order_epoch = None
        self._order = []

    def metrics(self) -> dict[str, float]:
        return {"cursor": float(self.cursor)}


class DifficultyPoolSampler:
    """Sample tasks by reward-EMA difficulty pools."""

    def __init__(
        self,
        config: RLDifficultyPoolSamplerConfig,
        task_count: int,
    ) -> None:
        if task_count < 1:
            raise ValueError(
                "Difficulty-pool curriculum sampler requires at least one task."
            )
        self.config = config
        self.task_count = task_count
        self.task_keys = tuple(f"record:{index}" for index in range(task_count))
        self.rng = random.Random(config.seed)
        self._ordered_pools = tuple(
            sorted(config.pools.items(), key=lambda item: item[1].threshold)
        )
        self.task_rewards: dict[str, float] = {}
        self.selections = 0

    def task_pool(self, task_key: str) -> str | None:
        score = self.task_rewards.get(task_key)
        if score is None:
            return None
        for name, pool in self._ordered_pools:
            if score <= pool.threshold:
                return name
        return self._ordered_pools[-1][0]

    def next_index(self) -> int:
        weights = [
            1.0
            if (pool := self.task_pool(task_key)) is None
            else self.config.pools[pool].weight
            for task_key in self.task_keys
        ]
        if not any(weights):
            raise RuntimeError(
                "Difficulty-pool curriculum has no task with positive sampling weight."
            )
        index = self.rng.choices(range(self.task_count), weights=weights, k=1)[0]
        self.selections += 1
        return index

    def observe(self, task_key: str, outputs: list[dict[str, Any]]) -> None:
        if task_key not in self.task_keys:
            raise ValueError(f"Unknown curriculum task key: {task_key!r}.")
        rewards = [
            float(output["reward"])
            for output in outputs
            if isinstance(output.get("reward"), int | float)
            and not isinstance(output.get("reward"), bool)
            and math.isfinite(float(output["reward"]))
        ]
        if not rewards:
            return
        group_mean = sum(rewards) / len(rewards)
        previous = self.task_rewards.get(task_key)
        self.task_rewards[task_key] = (
            group_mean
            if previous is None
            else self.config.ema_alpha * group_mean
            + (1.0 - self.config.ema_alpha) * previous
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "selections": self.selections,
            "task_count": self.task_count,
            "task_rewards": dict(self.task_rewards),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _require_fields(
            state_dict,
            {"rng", "selections", "task_count", "task_rewards"},
            owner="difficulty-pool sampler",
        )
        task_count = _nonnegative_int(state_dict["task_count"], name="task_count")
        if task_count != self.task_count:
            raise ValueError(
                "Curriculum task count changed across resume: "
                f"expected {self.task_count}, found {task_count}."
            )
        raw_rewards = state_dict["task_rewards"]
        if not isinstance(raw_rewards, Mapping):
            raise ValueError("Curriculum task_rewards state must be a mapping.")
        task_rewards = {str(key): float(value) for key, value in raw_rewards.items()}
        invalid_keys = set(task_rewards) - set(self.task_keys)
        if invalid_keys or not all(
            math.isfinite(value) for value in task_rewards.values()
        ):
            raise ValueError(
                "Curriculum task_rewards state is incompatible or non-finite."
            )
        self.rng.setstate(_tuple_tree(state_dict["rng"]))
        self.selections = _nonnegative_int(
            state_dict["selections"],
            name="selections",
        )
        self.task_rewards = task_rewards

    def metrics(self) -> dict[str, float]:
        occupancy = dict.fromkeys(self.config.pools, 0)
        for task_key in self.task_rewards:
            pool = self.task_pool(task_key)
            if pool is not None:
                occupancy[pool] += 1
        metrics = {
            "pool/unseen": float(self.task_count - len(self.task_rewards)),
            "reward_ema/mean": (
                sum(self.task_rewards.values()) / len(self.task_rewards)
                if self.task_rewards
                else 0.0
            ),
            "selections": float(self.selections),
        }
        metrics.update(
            {f"pool/{name}": float(count) for name, count in occupancy.items()}
        )
        return metrics


class AdvRangeGate:
    """Reject a group when every available advantage lies in one interval."""

    def __init__(self, config: RLAdvRangeGateConfig) -> None:
        self.config = config
        self.admitted = 0
        self.rejected = 0

    def admit(self, outputs: list[dict[str, Any]]) -> bool:
        advantages = [
            advantage for output in outputs for advantage in _output_advantages(output)
        ]
        decision = not advantages or not all(
            self.config.reject_min <= advantage <= self.config.reject_max
            for advantage in advantages
        )
        if decision:
            self.admitted += 1
        else:
            self.rejected += 1
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "rejected": self.rejected}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _require_fields(state_dict, {"admitted", "rejected"}, owner="advantage gate")
        self.admitted = _nonnegative_int(state_dict["admitted"], name="admitted")
        self.rejected = _nonnegative_int(state_dict["rejected"], name="rejected")

    def metrics(self) -> dict[str, float]:
        total = self.admitted + self.rejected
        return {
            "admitted": float(self.admitted),
            "rejected": float(self.rejected),
            "admission_rate": self.admitted / total if total else 0.0,
        }


class Curriculum:
    """Compose one finite-task sampler with named admission gates."""

    def __init__(
        self,
        config: RLCurriculumConfig,
        *,
        task_count: int,
        data_seed: int,
        shuffle: bool,
        start_cursor: int = 0,
    ) -> None:
        self.config = config
        if isinstance(config.sampler, RLStandardCurriculumSamplerConfig):
            self.sampler: Sampler = StandardSampler(
                task_count,
                seed=data_seed,
                shuffle=shuffle,
                cursor=start_cursor,
            )
        elif isinstance(config.sampler, RLDifficultyPoolSamplerConfig):
            if start_cursor:
                raise ValueError(
                    "Difficulty-pool curriculum resume requires persisted sampler state."
                )
            self.sampler = DifficultyPoolSampler(config.sampler, task_count)
        else:  # pragma: no cover - Pydantic owns the closed config union.
            raise TypeError(
                f"Unsupported curriculum sampler: {type(config.sampler).__name__}."
            )
        self.gates: dict[str, Gate] = {}
        for name, gate_config in config.gates.items():
            if isinstance(gate_config, RLAdvRangeGateConfig):
                self.gates[name] = AdvRangeGate(gate_config)
            else:  # pragma: no cover - Pydantic owns the closed config union.
                raise TypeError(
                    f"Unsupported curriculum gate: {type(gate_config).__name__}."
                )

    def next_record_index(self) -> tuple[int, str]:
        index = self.sampler.next_index()
        return index, f"record:{index}"

    def on_result(self, task_key: str, outputs: list[dict[str, Any]]) -> bool:
        if not outputs:
            raise ValueError("Cannot report an empty rollout group to curriculum.")
        self.sampler.observe(task_key, outputs)
        decisions = [gate.admit(outputs) for gate in self.gates.values()]
        return all(decisions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "gates": {name: gate.state_dict() for name, gate in self.gates.items()},
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        _require_fields(state_dict, {"gates", "sampler"}, owner="curriculum")
        gate_states = state_dict["gates"]
        sampler_state = state_dict["sampler"]
        if not isinstance(gate_states, Mapping) or not isinstance(
            sampler_state,
            Mapping,
        ):
            raise ValueError("Curriculum checkpoint state must contain mappings.")
        if set(gate_states) != set(self.gates):
            raise ValueError(
                "Curriculum checkpoint gates differ from configuration: "
                f"expected {sorted(self.gates)}, found {sorted(gate_states)}."
            )
        self.sampler.load_state_dict(sampler_state)
        for name, gate in self.gates.items():
            gate_state = gate_states[name]
            if not isinstance(gate_state, Mapping):
                raise ValueError(f"Curriculum gate state {name!r} must be a mapping.")
            gate.load_state_dict(gate_state)

    def metrics(self) -> dict[str, float]:
        metrics = {
            f"sampler/{name}": value for name, value in self.sampler.metrics().items()
        }
        for gate_name, gate in self.gates.items():
            metrics.update(
                {
                    f"gate/{gate_name}/{name}": value
                    for name, value in gate.metrics().items()
                }
            )
        return metrics


def _output_advantages(output: dict[str, Any]) -> list[float]:
    raw = output.get("advantage")
    values = raw if isinstance(raw, list) else [raw]
    return [
        float(value)
        for value in values
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]


def _require_fields(
    state_dict: Mapping[str, Any],
    expected: set[str],
    *,
    owner: str,
) -> None:
    if set(state_dict) != expected:
        raise ValueError(f"Curriculum {owner} state fields must be {sorted(expected)}.")


def _nonnegative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Curriculum {name} must be a non-negative integer.")
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_tuple_tree(item) for item in value)
    return value

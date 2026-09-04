from __future__ import annotations

import json

import pytest

from wavelet.configs.rl_config import (
    RLAdvRangeGateConfig,
    RLCurriculumConfig,
    RLDifficultyPoolConfig,
    RLDifficultyPoolSamplerConfig,
)
from wavelet.orchestrator.curriculum import (
    AdvRangeGate,
    Curriculum,
    DifficultyPoolSampler,
    StandardSampler,
)


def _output(reward: float, advantage: float | list[float]) -> dict[str, object]:
    return {"reward": reward, "advantage": advantage}


def test_advantage_range_gate_rejects_only_groups_inside_interval() -> None:
    gate = AdvRangeGate(
        RLAdvRangeGateConfig(reject_min=-0.1, reject_max=0.1),
    )

    assert gate.admit([_output(1.0, [0.0, 0.05])]) is False
    assert gate.admit([_output(1.0, [-0.2, 0.0])]) is True
    assert gate.admit([{"reward": 1.0}]) is True
    assert gate.metrics() == {
        "admitted": 2.0,
        "rejected": 1.0,
        "admission_rate": pytest.approx(2 / 3),
    }


def test_difficulty_pool_sampler_tracks_reward_ema_and_pool_weights() -> None:
    config = RLDifficultyPoolSamplerConfig(
        seed=7,
        ema_alpha=0.5,
        pools={
            "hard": RLDifficultyPoolConfig(threshold=0.25, weight=0.0),
            "normal": RLDifficultyPoolConfig(threshold=0.75, weight=1.0),
            "easy": RLDifficultyPoolConfig(threshold=1.0, weight=0.0),
        },
    )
    sampler = DifficultyPoolSampler(config, task_count=3)
    sampler.observe("record:0", [_output(0.0, 0.0)])
    sampler.observe("record:0", [_output(1.0, 1.0)])
    sampler.observe("record:1", [_output(0.5, 0.5)])
    sampler.observe("record:2", [_output(1.0, 1.0)])

    assert sampler.task_rewards["record:0"] == pytest.approx(0.5)
    assert sampler.task_pool("record:0") == "normal"
    assert sampler.task_pool("record:1") == "normal"
    assert sampler.task_pool("record:2") == "easy"
    assert {sampler.next_index() for _ in range(20)} <= {0, 1}
    assert sampler.metrics()["pool/normal"] == 2.0


def test_difficulty_pool_sampler_json_state_restores_exact_rng_sequence() -> None:
    config = RLDifficultyPoolSamplerConfig(seed=11, ema_alpha=1.0)
    first = DifficultyPoolSampler(config, task_count=4)
    first.observe("record:0", [_output(0.1, 0.0)])
    first.observe("record:1", [_output(0.5, 0.0)])
    first.observe("record:2", [_output(0.9, 0.0)])
    first.next_index()
    state = json.loads(json.dumps(first.state_dict()))

    restored = DifficultyPoolSampler(config, task_count=4)
    restored.load_state_dict(state)

    assert [first.next_index() for _ in range(20)] == [
        restored.next_index() for _ in range(20)
    ]
    assert restored.task_rewards == first.task_rewards


def test_standard_sampler_state_resumes_epoch_shuffle() -> None:
    first = StandardSampler(5, seed=3, shuffle=True)
    prefix = [first.next_index() for _ in range(7)]
    state = first.state_dict()
    restored = StandardSampler(5, seed=3, shuffle=True)
    restored.load_state_dict(state)

    assert len(prefix) == 7
    assert [first.next_index() for _ in range(10)] == [
        restored.next_index() for _ in range(10)
    ]


def test_curriculum_composes_sampler_gate_and_checkpoint_state() -> None:
    config = RLCurriculumConfig(
        sampler={"type": "difficulty_pool", "seed": 5, "ema_alpha": 1.0},
        gates={"zero_signal": {"type": "advantage_range"}},
    )
    curriculum = Curriculum(
        config,
        task_count=2,
        data_seed=0,
        shuffle=False,
    )
    _, task_key = curriculum.next_record_index()

    assert curriculum.on_result(task_key, [_output(0.5, 0.0)]) is False
    state = json.loads(json.dumps(curriculum.state_dict()))

    restored = Curriculum(
        config,
        task_count=2,
        data_seed=0,
        shuffle=False,
    )
    restored.load_state_dict(state)

    assert restored.state_dict()["sampler"]["task_rewards"] == {
        task_key: 0.5,
    }
    assert restored.metrics()["gate/zero_signal/rejected"] == 1.0

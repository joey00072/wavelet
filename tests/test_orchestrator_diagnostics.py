from __future__ import annotations

import json
from dataclasses import replace

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.entrypoints.rl_orchestrator_debug import main as orchestrator_debug_main
from wavelet.orchestrator.diagnostics import (
    orchestrator_debug_state,
    probe_orchestrator,
    sample_orchestrator_records,
    with_orchestrator_limits,
)


def _example(index: int) -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": f"prompt {index}"}],
        completion=[{"role": "assistant", "content": "ok"}],
        advantage=None,
        reward=1.0,
        input_ids=[1, 2, 3],
        target_ids=[2, 3, 4],
        loss_mask=[False, True, True],
        temperatures=[1.0, 1.0, 1.0],
        metadata={"example_id": index},
        source="test",
    )


class _FakeInference:
    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        return [
            replace(
                record,
                completion=[{"role": "assistant", "content": "ok"}],
                input_ids=[1, 2, 3],
                target_ids=[2, 3, 4],
                loss_mask=[False, True, True],
                inference_logprobs=[-0.1, -0.2],
                temperatures=[1.0, 1.0, 1.0],
            )
            for record in records
        ]


def test_orchestrator_debug_state_exposes_schedule() -> None:
    config = RLConfig(
        orchestrator={"examples_per_step": 8, "rollouts_per_example": 2},
    )

    state = orchestrator_debug_state(config)

    assert state["schedule"]["target_steps"] == 1
    assert state["schedule"]["examples_per_step"] == 8
    assert state["schedule"]["rollouts_per_example"] == 2
    assert state["schedule"]["chunks_per_step"] >= 1


def test_orchestrator_probe_times_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        "wavelet.orchestrator.diagnostics.load_rl_records",
        lambda _config: [_example(index) for index in range(4)],
    )
    config = RLConfig(
        reward={"mode": "passthrough"},
        orchestrator={
            "examples_per_step": 2,
            "rollouts_per_example": 2,
            "advantage_mode": "reward",
            "filter_zero_advantage": False,
        },
    )

    probe = probe_orchestrator(
        config,
        step=0,
        inference_engine=_FakeInference(),
    )

    assert probe.records_available == 4
    assert probe.records_selected == 2
    assert probe.records_scored == 4
    assert probe.records_trainable == 4
    assert "load_records" in probe.timings
    assert "generate_score" in probe.timings
    assert probe.metrics["progress/samples"] == 4.0


def test_sample_orchestrator_records_reports_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "wavelet.orchestrator.diagnostics.load_rl_records",
        lambda _config: [_example(index) for index in range(3)],
    )
    config = RLConfig(orchestrator={"examples_per_step": 2})

    sample = sample_orchestrator_records(config, step=0)

    assert sample["records_available"] == 3
    assert sample["records"] == 2
    assert sample["sample"][0]["source"] == "test"


def test_with_orchestrator_limits_overrides_probe_size() -> None:
    config = with_orchestrator_limits(
        RLConfig(),
        examples=3,
        rollouts=5,
    )

    assert config.orchestrator.examples_per_step == 3
    assert config.orchestrator.rollouts_per_example == 5


def test_orchestrator_debug_inspect_outputs_json(capsys) -> None:
    assert orchestrator_debug_main(["inspect", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schedule"]["target_steps"] == 1
    assert report["orchestrator"]["enabled"] is True

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.orchestrator.rollouts import RLOrchestrator


def _example() -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": "What is 40 + 2?"}],
        completion=[{"role": "assistant", "content": "42"}],
        target_completion=[{"role": "assistant", "content": "42"}],
        advantage=None,
        reward=None,
        source="test",
    )


class _FlakyEmptyEngine:
    def __init__(self) -> None:
        self.calls = 0

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        self.calls += 1
        if self.calls == 1:
            return [
                replace(
                    record,
                    completion=[{"role": "assistant", "content": ""}],
                    loss_mask=[True],
                    temperatures=[1.0],
                )
                for record in records
            ]
        return [
            replace(
                record,
                completion=[{"role": "assistant", "content": "42"}],
                loss_mask=[True],
                temperatures=[1.0],
            )
            for record in records
        ]


class _FlakyIncompleteGroupEngine:
    def __init__(self) -> None:
        self.calls = 0

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        self.calls += 1
        annotated: list[RLExample] = []
        for index, record in enumerate(records):
            if self.calls == 1 and index == 0:
                content = ""
            else:
                content = "42"
            annotated.append(
                replace(
                    record,
                    completion=[{"role": "assistant", "content": content}],
                    loss_mask=[True],
                    temperatures=[1.0],
                )
            )
        return annotated


def test_native_rollout_materialization_retries_empty_completions(
    tmp_path,
    monkeypatch,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        inference={"enabled": True},
        reward={"mode": "reference_match"},
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "advantage_mode": "reward",
            "zero_advantage_max_retries": 1,
        },
    )
    orchestrator = RLOrchestrator(config)
    monkeypatch.setattr(
        "wavelet.orchestrator.rollouts.load_rl_records",
        lambda _config: [_example()],
    )

    engine = _FlakyEmptyEngine()
    path = orchestrator.materialize_native_chunk(
        optimizer_step=0,
        chunk_index=0,
        queue_step=0,
        chunk_examples=1,
        inference_engine=engine,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert engine.calls == 2
    assert len(rows) == 1
    assert rows[0]["completion"] == [{"role": "assistant", "content": "42"}]
    assert rows[0]["reward"] == pytest.approx(1.0)


def test_native_rollout_materialization_retries_incomplete_groups(
    tmp_path,
    monkeypatch,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        inference={"enabled": True},
        reward={"mode": "reference_match"},
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 2,
            "advantage_mode": "group_reward",
            "zero_advantage_max_retries": 1,
        },
    )
    orchestrator = RLOrchestrator(config)
    monkeypatch.setattr(
        "wavelet.orchestrator.rollouts.load_rl_records",
        lambda _config: [_example()],
    )

    engine = _FlakyIncompleteGroupEngine()
    path = orchestrator.materialize_native_chunk(
        optimizer_step=0,
        chunk_index=0,
        queue_step=0,
        chunk_examples=1,
        inference_engine=engine,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert engine.calls == 2
    assert len(rows) == 2
    assert {row["reward"] for row in rows} == {1.0}


def test_native_rollout_materialization_fails_on_repeated_empty_completions(
    tmp_path,
    monkeypatch,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        inference={"enabled": True},
        reward={"mode": "reference_match"},
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "zero_advantage_max_retries": 1,
        },
    )
    orchestrator = RLOrchestrator(config)
    monkeypatch.setattr(
        "wavelet.orchestrator.rollouts.load_rl_records",
        lambda _config: [_example()],
    )

    class EmptyEngine:
        def annotate(self, records: list[RLExample]) -> list[RLExample]:
            return [
                replace(
                    record,
                    completion=[{"role": "assistant", "content": ""}],
                    loss_mask=[True],
                    temperatures=[1.0],
                )
                for record in records
            ]

    with pytest.raises(RuntimeError, match="All rollout groups"):
        orchestrator.materialize_native_chunk(
            optimizer_step=0,
            chunk_index=0,
            queue_step=0,
            chunk_examples=1,
            inference_engine=EmptyEngine(),
        )

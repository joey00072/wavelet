from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any
from unittest.mock import Mock

import pytest

import wavelet.orchestrator.envs as verifier_envs
from wavelet.configs.rl_config import GRPOAlgorithmConfig, RLConfig
from wavelet.data.rl_dataset import RLExample, _pretokenized_sample
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.verifiers import (
    _PendingVerifierRequest,
    _VerifierBatchStats,
    _VerifierGroupState,
    VerifierRolloutScheduler,
    _assign_rollout_advantages,
    _completed_group_outputs,
    _is_usable_training_group,
    _load_cached_env,
    _records_from_output,
    _rollout_records_policy_step,
    _run_all,
    _run_group,
    _sampling_args,
    _successful_rollout_outputs,
    _verifier_extra_env_kwargs,
)


def _trainable_trajectory() -> list[dict[str, Any]]:
    return [
        {
            "tokens": {
                "prompt_ids": [1],
                "prompt_mask": [False],
                "completion_ids": [2],
                "completion_mask": [True],
                "completion_logprobs": [-0.1],
            }
        }
    ]


CUSTOM_ALGORITHM_FILE = Path(__file__).parent / "fixtures" / "custom_algorithm.py"


def test_verifier_step_converts_to_trainable_record() -> None:
    output = {
        "example_id": 7,
        "task": "alphabet-sort",
        "reward": 1.0,
        "sampling_args": {"temperature": 0.7},
        "stop_condition": "done",
        "is_truncated": False,
        "error": None,
        "trajectory": [
            {
                "prompt": [
                    {"role": "system", "content": "sort"},
                    {"role": "assistant", "content": "old"},
                    {"role": "user", "content": "next"},
                ],
                "completion": [{"role": "assistant", "content": "new"}],
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [0, 0, 0],
                    "completion_ids": [4, 5],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.1, -0.2],
                },
            }
        ],
    }

    record = _records_from_output(output)[0]

    assert record.reward == 1.0
    assert record.input_ids == [1, 2, 3, 4]
    assert record.target_ids == [2, 3, 4, 5]
    assert record.loss_mask == [False, False, True, True]
    assert record.inference_logprobs == [-0.1, -0.2]
    assert record.temperatures == [0.7, 0.7]
    assert record.metadata == {
        "group_key": '{"env_name":"alphabet-sort","example_id":"7"}',
        "rollout_key": '{"env_name":"alphabet-sort","example_id":"7"}:0',
        "stop_condition": "done",
        "is_truncated": False,
        "completion_token_count": 2,
        "tool_response_token_count": 0,
        "turn_count": 1,
        "_wavelet_rollout_count": 1,
        "task": {"name": "alphabet-sort", "example_id": 7},
        "harness": {
            "name": "alphabet-sort",
            "type": "environment",
            "version": None,
        },
        "rollout": {
            "group_key": '{"env_name":"alphabet-sort","example_id":"7"}',
            "rollout_key": '{"env_name":"alphabet-sort","example_id":"7"}:0',
            "sample_index": 0,
            "trajectory_id": '{"env_name":"alphabet-sort","example_id":"7"}:0',
            "num_turns": 1,
            "tool_calls": 0,
            "elapsed_sec": None,
            "stop_condition": "done",
            "is_truncated": False,
            "error": None,
            "reward_components": None,
        },
    }
    assert record.prompt[1]["step_loss_mask"] == 0
    assert record.source == "alphabet-sort"


def test_verifier_record_preserves_actual_generation_policy_step() -> None:
    output = {
        "example_id": 7,
        "reward": 1.0,
        "advantage": 1.0,
        "_wavelet_policy_step": 3,
        "trajectory": _trainable_trajectory(),
    }

    record = _records_from_output(output)[0]

    assert record.metadata["policy_step"] == 3
    assert _rollout_records_policy_step([record], fallback=9) == 3


def test_rollout_batch_policy_step_uses_oldest_actual_policy() -> None:
    records = [
        RLExample([], [], 1.0, 1.0, metadata={"policy_step": step})
        for step in (5, 3, 4)
    ]

    assert _rollout_records_policy_step(records, fallback=9) == 3


def test_custom_verifier_rollout_function_loads_without_env_import() -> None:
    orchestrator = RLOrchestrator(
        RLConfig(
            orchestrator={
                "custom_rollout_function": (
                    "wavelet.orchestrator.verifiers:generate_rollouts"
                )
            }
        )
    )

    function = orchestrator._load_custom_rollout_function(
        "wavelet.orchestrator.verifiers:generate_rollouts"
    )

    assert function.__name__ == "generate_rollouts"


def test_verifier_sampling_args_preserve_extra_body() -> None:
    config = RLConfig(
        orchestrator={"enabled": False},
        inference={
            "sampling": {
                "top_k": -1,
                "min_p": 0.0,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "custom": "value",
                },
                "min_tokens": 2,
            }
        }
    )

    args = _sampling_args(config)

    assert args["extra_body"]["return_token_ids"] is True
    assert args["extra_body"]["top_k"] == -1
    assert args["extra_body"]["min_p"] == 0.0
    assert args["extra_body"]["min_tokens"] == 2
    assert args["extra_body"]["custom"] == "value"
    assert args["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_verifier_sampling_args_include_policy_cache_salt() -> None:
    config = RLConfig()

    args = _sampling_args(config, cache_salt="7")

    assert args["extra_body"]["cache_salt"] == "7"
    assert args["extra_body"]["return_token_ids"] is True


def test_verifier_env_gets_sequence_budget_kwargs() -> None:
    config = RLConfig(data={"seq_len": 2048})

    assert _verifier_extra_env_kwargs(config) == {
        "max_seq_len": 2048,
        "max_total_completion_tokens": -1,
    }


def test_verifier_env_sequence_budget_uses_rollout_context_when_smaller() -> None:
    config = RLConfig(
        data={"seq_len": 16384},
        inference={"vllm": {"max_model_len": 8192}},
    )

    assert _verifier_extra_env_kwargs(config)["max_seq_len"] == 8192


def test_verifier_env_extra_kwargs_include_timeout_when_configured() -> None:
    config = RLConfig(
        orchestrator={
            "verifier_timeout_seconds": 123.0,
            "verifier_max_total_completion_tokens": 4096,
        }
    )

    assert _verifier_extra_env_kwargs(config) == {
        "max_seq_len": 128,
        "max_total_completion_tokens": 4096,
        "timeout_seconds": 123.0,
    }


def test_run_all_uses_target_group_scheduler_when_dataset_is_larger() -> None:
    class DummyRolloutInput:
        def __init__(self, **kwargs: Any) -> None:
            self.payload = kwargs

    class DummyVerifierModule:
        RolloutInput = DummyRolloutInput

    class DummyEnv:
        requires_group_scoring = False

        async def run_rollout(
            self,
            rollout_input: DummyRolloutInput,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            example_id = rollout_input.payload["example_id"]
            reward = float(example_id % 2)
            return {
                "example_id": example_id,
                "reward": reward,
                "error": None,
                "trajectory": _trainable_trajectory(),
            }

    async def run() -> list[dict[str, Any]]:
        records = [
            RLExample(
                prompt=[],
                completion=[],
                advantage=None,
                reward=None,
                metadata={"example_id": index},
            )
            for index in range(4)
        ]
        return await _run_all(
            DummyVerifierModule(),
            DummyEnv(),
            records,
            clients=[object()],
            model="debug",
            sampling_args={},
            rollout_count=2,
            max_retries=0,
            target_groups=1,
            filter_zero_advantage=False,
            advantage_epsilon=1e-6,
            algorithm_config=GRPOAlgorithmConfig(),
        )

    outputs = asyncio.run(run())

    assert len(outputs) == 2
    assert len({output["example_id"] for output in outputs}) == 1
    assert {output["env_name"] for output in outputs} == {"verifier"}
    assert [output["advantage"] for output in outputs] == [0.0, 0.0]


def test_run_group_uses_env_group_scoring_when_required() -> None:
    class DummyRolloutInput:
        def __init__(self, **kwargs: Any) -> None:
            self.payload = kwargs

    class DummyVerifierModule:
        RolloutInput = DummyRolloutInput

    class DummyEnv:
        requires_group_scoring = True

        def __init__(self) -> None:
            self.rollout_calls = 0
            self.group_calls = 0

        async def run_rollout(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.rollout_calls += 1
            raise AssertionError("group-scoring envs must use run_group")

        async def run_group(
            self,
            group_inputs: list[DummyRolloutInput],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            self.group_calls += 1
            return [
                {
                    "example_id": rollout_input.payload["example_id"],
                    "reward": float(index),
                    "error": None,
                    "trajectory": _trainable_trajectory(),
                }
                for index, rollout_input in enumerate(group_inputs)
            ]

    async def run() -> tuple[DummyEnv, list[dict[str, Any]]]:
        env = DummyEnv()
        outputs = await _run_group(
            DummyVerifierModule(),
            env,
            {"example_id": 3, "prompt": []},
            client=object(),
            model="debug",
            sampling_args={},
            rollout_count=2,
            max_retries=0,
            algorithm_config=GRPOAlgorithmConfig(),
        )
        return env, outputs

    env, outputs = asyncio.run(run())

    assert env.group_calls == 1
    assert env.rollout_calls == 0
    assert [output["reward"] for output in outputs] == [0.0, 1.0]
    assert [output["advantage"] for output in outputs] == [-0.5, 0.5]


def test_run_all_uses_group_scoring_without_target_scheduler() -> None:
    class DummyRolloutInput:
        def __init__(self, **kwargs: Any) -> None:
            self.payload = kwargs

    class DummyVerifierModule:
        RolloutInput = DummyRolloutInput

    class DummyEnv:
        requires_group_scoring = True

        def __init__(self) -> None:
            self.rollout_calls = 0
            self.group_calls = 0

        async def run_rollout(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.rollout_calls += 1
            raise AssertionError("group-scoring envs must use run_group")

        async def run_group(
            self,
            group_inputs: list[DummyRolloutInput],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            self.group_calls += 1
            return [
                {
                    "example_id": rollout_input.payload["example_id"],
                    "reward": float(index),
                    "error": None,
                    "trajectory": _trainable_trajectory(),
                }
                for index, rollout_input in enumerate(group_inputs)
            ]

    async def run() -> tuple[DummyEnv, list[dict[str, Any]]]:
        env = DummyEnv()
        records = [
            RLExample(
                prompt=[],
                completion=[],
                advantage=None,
                reward=None,
                metadata={"example_id": 9},
            )
        ]
        outputs = await _run_all(
            DummyVerifierModule(),
            env,
            records,
            clients=[object()],
            model="debug",
            sampling_args={},
            rollout_count=2,
            max_retries=0,
            target_groups=1,
            filter_zero_advantage=False,
            advantage_epsilon=1e-6,
            algorithm_config=GRPOAlgorithmConfig(),
            env_name="group-env",
        )
        return env, outputs

    env, outputs = asyncio.run(run())

    assert env.group_calls == 1
    assert env.rollout_calls == 0
    assert len(outputs) == 2
    assert {output["env_name"] for output in outputs} == {"group-env"}
    assert [output["advantage"] for output in outputs] == [-0.5, 0.5]


def test_run_all_keeps_duplicate_example_ids_in_separate_groups() -> None:
    class DummyRolloutInput:
        def __init__(self, **kwargs: Any) -> None:
            self.payload = kwargs

    class DummyVerifierModule:
        RolloutInput = DummyRolloutInput

    class DummyEnv:
        requires_group_scoring = False

        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        async def run_rollout(
            self,
            rollout_input: DummyRolloutInput,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            base = int(rollout_input.payload["base"])
            rollout_index = self.calls.get(base, 0)
            self.calls[base] = rollout_index + 1
            return {
                "example_id": "duplicate",
                "reward": float(base + rollout_index),
                "error": None,
                "trajectory": _trainable_trajectory(),
            }

    async def run() -> list[dict[str, Any]]:
        records = [
            RLExample(
                prompt=[],
                completion=[],
                advantage=None,
                reward=None,
                metadata={
                    "verifier_example": {
                        "example_id": "duplicate",
                        "base": base,
                    }
                },
            )
            for base in (0, 10)
        ]
        return await _run_all(
            DummyVerifierModule(),
            DummyEnv(),
            records,
            clients=[object()],
            model="debug",
            sampling_args={},
            rollout_count=2,
            max_retries=0,
            target_groups=2,
            filter_zero_advantage=False,
            advantage_epsilon=1e-6,
            algorithm_config=GRPOAlgorithmConfig(),
        )

    outputs = asyncio.run(run())

    assert [output["reward"] for output in outputs] == [0.0, 1.0, 10.0, 11.0]
    assert [output["advantage"] for output in outputs] == [-0.5, 0.5] * 2
    assert [output["_wavelet_group_id"] for output in outputs] == [
        "complete:0",
        "complete:0",
        "complete:1",
        "complete:1",
    ]
    record_group_keys = {
        _records_from_output(output)[0].metadata["group_key"] for output in outputs
    }
    assert len(record_group_keys) == 2


def test_verifier_scheduler_uses_oversampling_without_async_multiplier() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 32,
            "rollouts_per_example": 16,
            "oversampling_factor": 2.0,
            "max_async_level": 8,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 32
    scheduler.clients = [object()] * 6

    assert scheduler.max_inflight_groups == 64


def test_verifier_scheduler_cycles_every_record_before_repeating() -> None:
    records = [
        RLExample(prompt=[], completion=[], advantage=0.0, reward=0.0, source=str(i))
        for i in range(5)
    ]
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.records = records
    scheduler.config = RLConfig(data={"shuffle": False})
    scheduler.record_cursor = 0
    scheduler._record_order_epoch = None  # noqa: SLF001
    scheduler._record_order = []  # noqa: SLF001

    selected = [scheduler._next_record().source for _ in range(7)]  # noqa: SLF001

    assert selected == ["0", "1", "2", "3", "4", "0", "1"]


def test_verifier_scheduler_shuffles_deterministically_per_epoch() -> None:
    records = [
        RLExample(prompt=[], completion=[], advantage=0.0, reward=0.0, source=str(i))
        for i in range(5)
    ]

    def sample() -> list[str]:
        scheduler = object.__new__(VerifierRolloutScheduler)
        scheduler.records = records
        scheduler.config = RLConfig(data={"shuffle": True, "seed": 123})
        scheduler.record_cursor = 0
        scheduler._record_order_epoch = None  # noqa: SLF001
        scheduler._record_order = []  # noqa: SLF001
        return [scheduler._next_record().source for _ in range(10)]  # noqa: SLF001

    selected = sample()

    assert sample() == selected
    assert set(selected[:5]) == {"0", "1", "2", "3", "4"}
    assert set(selected[5:]) == {"0", "1", "2", "3", "4"}


def test_verifier_scheduler_uses_explicit_pending_chunk_limit() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 64,
            "rollouts_per_example": 8,
            "rollout_chunk_examples": 8,
            "max_async_level": 8,
            "max_pending_rollout_chunks": 16,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 64
    scheduler.clients = [object()] * 4

    assert scheduler.max_inflight_groups == 128


def test_verifier_scheduler_rollout_capacity_uses_pending_chunk_limit() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 64,
            "rollouts_per_example": 8,
            "rollout_chunk_examples": 8,
            "max_pending_rollout_chunks": 16,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 64
    scheduler.rollout_count = 8
    scheduler.clients = [object()] * 4

    assert scheduler.max_inflight_groups == 128
    assert scheduler.max_inflight_rollouts == 1024


def test_pending_chunk_limit_is_not_expanded_by_oversampling() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 32,
            "rollouts_per_example": 8,
            "rollout_chunk_examples": 8,
            "max_pending_rollout_chunks": 8,
            "oversampling_factor": 3.0,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 32
    scheduler.rollout_count = 8
    scheduler.clients = [object()] * 4

    assert scheduler.max_inflight_groups == 64
    assert scheduler.max_inflight_rollouts == 512


def test_verifier_scheduler_uses_explicit_max_inflight_rollouts() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 64,
            "rollouts_per_example": 8,
            "oversampling_factor": 4.0,
            "max_inflight_rollouts": 128,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 64
    scheduler.rollout_count = 8
    scheduler.clients = [object()] * 4

    assert scheduler.max_inflight_groups == 16
    assert scheduler.max_inflight_rollouts == 128


def test_explicit_rollout_limit_is_hard_with_many_clients() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 64,
            "rollouts_per_example": 8,
            "max_inflight_rollouts": 130,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 64
    scheduler.rollout_count = 8
    scheduler.clients = [object()] * 256

    assert scheduler.max_inflight_groups == 16
    assert scheduler.max_inflight_rollouts == 130


def test_pending_chunk_limit_is_hard_with_many_clients() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 64,
            "rollouts_per_example": 8,
            "rollout_chunk_examples": 2,
            "max_pending_rollout_chunks": 2,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.target_groups = 64
    scheduler.rollout_count = 8
    scheduler.clients = [object()] * 256

    assert scheduler.max_inflight_groups == 4
    assert scheduler.max_inflight_rollouts == 32


def test_verifier_executors_scale_to_high_water_concurrency(
    monkeypatch,
) -> None:
    scale_executors = Mock()
    thread_utils = type(
        "ThreadUtils",
        (),
        {"scale_executors": scale_executors},
    )()
    monkeypatch.setitem(
        sys.modules,
        "verifiers.utils.thread_utils",
        thread_utils,
    )
    monkeypatch.setattr(verifier_envs, "_VERIFIER_EXECUTOR_CONCURRENCY", 0)

    assert verifier_envs._scale_verifier_executors(256) == 256
    assert verifier_envs._scale_verifier_executors(128) == 256

    scale_executors.assert_called_once_with(256)


def test_cached_verifier_environments_are_torn_down_once(monkeypatch) -> None:
    teardown = Mock(return_value=None)
    env = type("Env", (), {"teardown": teardown})()
    shutdown_executors = Mock()
    thread_utils = type(
        "ThreadUtils",
        (),
        {"shutdown_executors": shutdown_executors},
    )()
    monkeypatch.setitem(
        sys.modules,
        "verifiers.utils.thread_utils",
        thread_utils,
    )
    monkeypatch.setattr(
        verifier_envs,
        "_ENV_CACHE",
        {("train", "", ""): env, ("eval", "", ""): env},
    )
    monkeypatch.setattr(verifier_envs, "_VERIFIER_EXECUTOR_CONCURRENCY", 256)

    asyncio.run(verifier_envs._teardown_cached_verifier_envs())

    teardown.assert_called_once_with()
    shutdown_executors.assert_called_once_with()
    assert verifier_envs._ENV_CACHE == {}
    assert verifier_envs._VERIFIER_EXECUTOR_CONCURRENCY == 0


def test_verifier_scheduler_drains_done_tasks_and_buffers_extra_groups() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "filter_zero_advantage": False,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 1
    scheduler.target_groups = 1
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = []

    def output(example_id: int) -> dict[str, Any]:
        return {
            "example_id": example_id,
            "reward": float(example_id),
            "error": None,
            "trajectory": _trainable_trajectory(),
        }

    async def run() -> list[RLExample]:
        loop = asyncio.get_running_loop()
        for group_id in (1, 2):
            task = loop.create_future()
            task.set_result([output(group_id)])
            scheduler.pending[task] = _PendingVerifierRequest(
                group_id=group_id,
                client_index=0,
                rollout_count=1,
                policy_step=3,
            )
            scheduler.pending_clients[task] = 0
            scheduler.groups[group_id] = _VerifierGroupState(
                example={"example_id": group_id},
                rollouts_to_schedule=0,
                policy_step=3,
            )
        scheduler._fill_inflight = lambda: None  # type: ignore[method-assign]
        return await scheduler.generate_batch(target_groups=1)

    records = asyncio.run(run())

    assert len(records) == 1
    assert not scheduler.pending
    assert len(scheduler.ready_groups) == 1
    ready_example_id = scheduler.ready_groups[0][0]["example_id"]
    record_example_id = json.loads(records[0].metadata["group_key"])["example_id"]
    assert {ready_example_id, int(record_example_id)} == {1, 2}
    assert scheduler.ready_groups[0][0]["env_name"] == "verifier"
    assert scheduler.ready_groups[0][0]["_wavelet_policy_step"] == 3
    assert records[0].metadata["policy_step"] == 3


def test_verifier_scheduler_resamples_zero_advantage_groups() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "examples_per_step": 2,
            "rollouts_per_example": 2,
            "filter_zero_advantage": True,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 2
    scheduler.target_groups = 2
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = []
    scheduler.ready_group_off_policy_steps = []
    scheduler._fill_inflight = lambda: None  # type: ignore[method-assign]

    def output(example_id: int, reward: float) -> dict[str, Any]:
        return {
            "example_id": example_id,
            "reward": reward,
            "error": None,
            "sampling_args": {"temperature": 1.0},
            "trajectory": _trainable_trajectory(),
        }

    async def run() -> tuple[
        list[RLExample], tuple[int, int, int], tuple[int, int, int]
    ]:
        loop = asyncio.get_running_loop()
        for group_id, rewards in enumerate(([0.0, 1.0], [1.0, 1.0])):
            task = loop.create_future()
            task.set_result([output(group_id, reward) for reward in rewards])
            scheduler.pending[task] = _PendingVerifierRequest(
                group_id=group_id,
                client_index=0,
                rollout_count=2,
            )
            scheduler.pending_clients[task] = 0
            scheduler.groups[group_id] = _VerifierGroupState(
                example={"example_id": group_id},
                rollouts_to_schedule=0,
            )
        tasks = list(scheduler.pending)
        outputs: list[dict[str, Any]] = []
        accepted = scheduler._consume_completed_task(  # noqa: SLF001
            tasks[0],
            target_groups=1,
            outputs=outputs,
            accepted_groups=0,
        )
        rejected = scheduler._consume_completed_task(  # noqa: SLF001
            tasks[1],
            target_groups=1,
            outputs=outputs,
            accepted_groups=accepted[0],
        )
        records = [
            record
            for output_row in outputs
            for record in _records_from_output(output_row)
        ]
        return records, accepted, rejected

    records, accepted, rejected = asyncio.run(run())

    assert accepted == (1, 1, 0)
    assert rejected == (0, 1, 1)
    assert len(records) == 2
    assert [record.reward for record in records] == [0.0, 1.0]
    assert all(record.loss_mask == [True] for record in records)


def test_verifier_batch_stats_report_unfiltered_generation_reward() -> None:
    stats = _VerifierBatchStats()
    stats.observe([{"reward": 0.0}, {"reward": 1.0}], admitted=True)
    stats.observe([{"reward": 1.0}, {"reward": 1.0}], admitted=False)

    metrics = stats.metrics(rollouts_per_group=2)

    assert metrics["generation/groups/completed"] == 2.0
    assert metrics["generation/groups/admitted"] == 1.0
    assert metrics["generation/groups/rejected"] == 1.0
    assert metrics["generation/groups/admission_rate"] == 0.5
    assert metrics["generation/rollouts/scored"] == 4.0
    assert metrics["generation/reward/mean"] == 0.75
    assert metrics["generation/solve_none/rate"] == 0.0
    assert metrics["generation/solve_all/rate"] == 0.5
    assert metrics["generation/effective_groups/rate"] == 0.5

    empty_metrics = _VerifierBatchStats().metrics(rollouts_per_group=2)
    assert "generation/reward/mean" not in empty_metrics


def test_successful_rollout_outputs_raises_on_rate_limit_exception() -> None:
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        _successful_rollout_outputs(
            [
                RuntimeError(
                    "Error code: 429 - GoUsageLimitError: 5-hour usage limit reached"
                )
            ]
        )


def test_successful_rollout_outputs_raises_on_rate_limit_payload() -> None:
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        _successful_rollout_outputs(
            [
                {
                    "error": {
                        "type": "GoUsageLimitError",
                        "message": "5-hour usage limit reached",
                    },
                    "reward": 0.0,
                }
            ]
        )


def test_rl_config_filters_zero_advantages_by_default() -> None:
    config = RLConfig()

    assert config.orchestrator.filter_zero_advantage is True


def test_load_cached_env_applies_extra_env_kwargs() -> None:
    class DummyEnv:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def set_kwargs(self, **kwargs: Any) -> None:
            self.kwargs.update(kwargs)

    class DummyVerifierModule:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def load_environment(self, env_id: str, **env_args: Any) -> DummyEnv:
            assert env_id == "alphabet-sort"
            assert env_args == {"min_turns": 3}
            return self.env

    vf = DummyVerifierModule()
    env, cache_hit = _load_cached_env(
        vf,
        "alphabet-sort",
        {"min_turns": 3},
        {"max_seq_len": 2048, "max_total_completion_tokens": -1},
    )

    assert not cache_hit
    assert env.kwargs == {
        "max_seq_len": 2048,
        "max_total_completion_tokens": -1,
    }


def test_pretokenized_rows_preserve_sequence_window() -> None:
    sample = _pretokenized_sample(
        RLExample(
            prompt=[],
            completion=[],
            advantage=1.0,
            reward=1.0,
            input_ids=[10, 11, 12, 13],
            target_ids=[11, 12, 13, 14],
            loss_mask=[False, True, True, True],
        ),
        seq_len=4,
    )

    assert sample is not None
    assert sample["input_ids"] == [10, 11, 12, 13]
    assert sample["target_ids"] == [11, 12, 13, 14]
    assert sample["loss_mask"] == [False, True, True, True]


def test_verifier_multiturn_interleave_preserves_turn_boundary() -> None:
    output = {
        "example_id": 7,
        "task": "alphabet-sort",
        "reward": 1.0,
        "advantage": 0.5,
        "sampling_args": {"temperature": 1.0},
        "stop_condition": "done",
        "is_truncated": False,
        "error": None,
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "a"}],
                "completion": [{"role": "assistant", "content": "b"}],
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3, 4],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.3, -0.4],
                },
            },
            {
                "prompt": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ],
                "completion": [{"role": "assistant", "content": "d"}],
                "tokens": {
                    "prompt_ids": [1, 2, 3, 4, 5],
                    "prompt_mask": [0, 0, 0, 0, 0],
                    "completion_ids": [6],
                    "completion_mask": [1],
                    "completion_logprobs": [-0.6],
                },
            },
        ],
    }

    record = _records_from_output(output)[0]

    assert record.input_ids == [1, 2, 3, 4, 5]
    assert record.target_ids == [2, 3, 4, 5, 6]
    assert record.loss_mask == [False, True, True, False, True]
    assert record.inference_logprobs == [-0.3, -0.4, -0.6]


def test_verifier_interleave_trains_first_token_when_extension_has_no_new_prompt() -> (
    None
):
    output = {
        "example_id": 7,
        "task": "alphabet-sort",
        "reward": 1.0,
        "advantage": 0.5,
        "sampling_args": {"temperature": 1.0},
        "error": None,
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "a"}],
                "completion": [{"role": "assistant", "content": "b"}],
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3, 4],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.3, -0.4],
                },
            },
            {
                "prompt": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
                "completion": [{"role": "assistant", "content": "c"}],
                "tokens": {
                    "prompt_ids": [1, 2, 3, 4],
                    "prompt_mask": [0, 0, 0, 0],
                    "completion_ids": [5, 6],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.5, -0.6],
                },
            },
        ],
    }

    record = _records_from_output(output)[0]

    assert record.input_ids == [1, 2, 3, 4, 5]
    assert record.target_ids == [2, 3, 4, 5, 6]
    assert record.loss_mask == [False, True, True, True, True]
    assert record.inference_logprobs == [-0.3, -0.4, -0.5, -0.6]


def test_verifier_records_include_length_metadata() -> None:
    output = {
        "example_id": 7,
        "task": "tool-env",
        "reward": 1.0,
        "advantage": 0.5,
        "sampling_args": {"temperature": 1.0},
        "metrics": {"rlm_total_tool_response_tokens": 12},
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "a"}],
                "completion": [{"role": "assistant", "content": "b"}],
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3, 4, 5],
                    "completion_mask": [1, 1, 1],
                    "completion_logprobs": [-0.3, -0.4, -0.5],
                },
            }
        ],
    }

    record = _records_from_output(output)[0]

    assert record.metadata["completion_token_count"] == 3
    assert record.metadata["tool_response_token_count"] == 12
    assert record.metadata["turn_count"] == 1


def test_verifier_records_keep_task_and_harness_metadata_separate() -> None:
    output = {
        "example_id": "case-7",
        "task": "repair",
        "harness_name": "local-runner",
        "harness_type": "agent",
        "harness_version": "1",
        "trajectory_id": "traj-7",
        "elapsed_seconds": 1.25,
        "reward_components": {"exact": 1.0},
        "is_truncated": True,
        "error": None,
        "reward": 1.0,
        "advantage": 0.5,
        "sampling_args": {"temperature": 1.0},
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "fix"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "patch",
                        "tool_calls": [{"id": "call-1"}],
                    }
                ],
                "tokens": {
                    "prompt_ids": [1],
                    "prompt_mask": [0],
                    "completion_ids": [2],
                    "completion_mask": [1],
                    "completion_logprobs": [-0.2],
                },
            }
        ],
    }

    record = _records_from_output(output)[0]

    assert record.source == "repair"
    assert record.metadata["task"] == {"name": "repair", "example_id": "case-7"}
    assert record.metadata["harness"] == {
        "name": "local-runner",
        "type": "agent",
        "version": "1",
    }
    assert record.metadata["rollout"] == {
        "group_key": '{"env_name":"repair","example_id":"case-7"}',
        "rollout_key": '{"env_name":"repair","example_id":"case-7"}:0',
        "sample_index": 0,
        "trajectory_id": "traj-7",
        "num_turns": 1,
        "tool_calls": 1,
        "elapsed_sec": 1.25,
        "stop_condition": None,
        "is_truncated": True,
        "error": None,
        "reward_components": {"exact": 1.0},
    }


def test_verifier_tool_response_length_penalty_shapes_advantages() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "length_penalty": {
                "type": "tokens",
                "completion_weight": 0.0,
                "tool_response_weight": 1.0,
            },
        }
    )
    outputs = [
        {
            "example_id": 1,
            "reward": 1.0,
            "trajectory": [{"tokens": {"completion_ids": [1]}}],
            "metrics": {"rlm_total_tool_response_tokens": 100},
        },
        {
            "example_id": 1,
            "reward": 1.0,
            "trajectory": [{"tokens": {"completion_ids": [1]}}],
            "metrics": {"rlm_total_tool_response_tokens": 0},
        },
        {
            "example_id": 1,
            "reward": 0.0,
            "trajectory": [{"tokens": {"completion_ids": [1]}}],
            "metrics": {"rlm_total_tool_response_tokens": 50},
        },
    ]

    _assign_rollout_advantages(outputs, config)

    assert outputs[1]["advantage"] > outputs[0]["advantage"]
    assert outputs[0]["advantage"] > outputs[2]["advantage"]
    assert sum(float(output["advantage"]) for output in outputs) == pytest.approx(0.0)


def test_successful_rollout_outputs_skip_exceptions_errors_and_missing_rewards() -> (
    None
):
    trainable_trajectory = _trainable_trajectory()
    outputs = _successful_rollout_outputs(
        [
            RuntimeError("boom"),
            {"example_id": 0, "error": "timeout", "reward": 0.0},
            {"example_id": 1, "completion": []},
            object(),
            {
                "example_id": 2,
                "reward": 1.0,
                "error": None,
                "trajectory": trainable_trajectory,
            },
        ]
    )

    assert outputs == [
        {
            "example_id": 2,
            "reward": 1.0,
            "error": None,
            "trajectory": trainable_trajectory,
        }
    ]


def test_successful_rollout_outputs_skip_empty_or_untrainable_trajectories() -> None:
    assert (
        _successful_rollout_outputs(
            [
                {"example_id": 0, "reward": 1.0, "error": None, "trajectory": []},
                {
                    "example_id": 1,
                    "reward": 1.0,
                    "error": None,
                    "trajectory": [
                        {
                            "tokens": {
                                "prompt_ids": [1],
                                "prompt_mask": [False],
                                "completion_ids": [2],
                                "completion_mask": [False],
                                "completion_logprobs": [-0.1],
                            }
                        }
                    ],
                },
            ]
        )
        == []
    )


def test_successful_rollout_outputs_can_keep_eval_only_rewards() -> None:
    assert _successful_rollout_outputs(
        [{"example_id": 0, "reward": 1.0, "error": None, "trajectory": []}],
        require_trainable=False,
    ) == [{"example_id": 0, "reward": 1.0, "error": None, "trajectory": []}]


def test_completed_group_outputs_treat_task_exception_as_empty_group() -> None:
    async def fail() -> list[dict[str, float]]:
        raise RuntimeError("boom")

    async def run() -> list[dict[str, float]]:
        task = asyncio.create_task(fail())
        await asyncio.gather(task, return_exceptions=True)
        return _completed_group_outputs(task)

    assert asyncio.run(run()) == []


def test_verifier_scheduler_reschedules_failed_single_rollout() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 2,
            "filter_zero_advantage": False,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 2
    scheduler.target_groups = 1
    scheduler.clients = [object()]
    scheduler.requires_group_scoring = False
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0, "prompt": "x"},
            rollouts_to_schedule=2,
        )
    }
    scheduler._scheduled = 0

    async def fail() -> list[dict[str, float]]:
        return []

    async def succeed() -> list[dict[str, float]]:
        return [
            {
                "example_id": 0,
                "reward": 1.0,
                "error": None,
                "trajectory": _trainable_trajectory(),
            }
        ]

    def fill_inflight(self) -> None:
        group = self.groups.get(0)
        while group is not None and group.rollouts_to_schedule > 0:
            group.rollouts_to_schedule -= 1
            task = asyncio.create_task(fail() if self._scheduled == 0 else succeed())
            self.pending[task] = _PendingVerifierRequest(
                group_id=0,
                client_index=0,
                rollout_count=1,
            )
            self.pending_clients[task] = 0
            self._scheduled += 1

    scheduler._fill_inflight = MethodType(fill_inflight, scheduler)

    asyncio.run(scheduler.generate_batch(target_groups=1))

    assert scheduler._scheduled == 3
    assert scheduler.groups == {}


def test_verifier_scheduler_drops_incomplete_group_after_policy_change() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 2,
            "filter_zero_advantage": False,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.rollout_count = 2
    scheduler.requires_group_scoring = False
    scheduler.policy_step = 3
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0, "prompt": "x"},
            rollouts_to_schedule=0,
            policy_step=2,
        )
    }

    async def run() -> tuple[int, int, int]:
        task = asyncio.get_running_loop().create_future()
        task.set_result([])
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
            policy_step=2,
        )
        return scheduler._consume_completed_task(  # noqa: SLF001
            task,
            target_groups=1,
            outputs=[],
            accepted_groups=0,
        )

    assert asyncio.run(run()) == (0, 1, 1)
    assert scheduler.groups == {}


def test_verifier_scheduler_cancels_stale_off_policy_groups() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = RLConfig(orchestrator={"max_off_policy_steps": 1})
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0, "prompt": "x"},
            rollouts_to_schedule=1,
        )
    }
    scheduler.cancelled_rollouts_count = 0

    async def pending_rollout() -> list[dict[str, float]]:
        await asyncio.sleep(60)
        return [{"example_id": 0, "reward": 1.0, "error": None}]

    async def run() -> None:
        task = asyncio.create_task(pending_rollout())
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
        )
        scheduler.pending_clients[task] = 0

        assert await scheduler.mark_policy_update() == 0
        assert scheduler.pending[task].off_policy_steps == 1
        assert await scheduler.mark_policy_update() == 1
        assert task.cancelled()

    asyncio.run(run())

    assert scheduler.pending == {}
    assert scheduler.groups == {}
    assert scheduler.cancelled_rollouts_count == 1


def test_verifier_scheduler_uses_actual_policy_version_lag() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = RLConfig(orchestrator={"max_off_policy_steps": 1})
    scheduler.policy_step = 5
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0, "prompt": "x"},
            rollouts_to_schedule=1,
            policy_step=2,
        )
    }
    scheduler.cancelled_rollouts_count = 0

    async def pending_rollout() -> list[dict[str, float]]:
        await asyncio.sleep(60)
        return []

    async def run() -> int:
        task = asyncio.create_task(pending_rollout())
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
            policy_step=2,
        )
        scheduler.pending_clients[task] = 0
        return await scheduler.mark_policy_update()

    assert asyncio.run(run()) == 1
    assert scheduler.pending == {}
    assert scheduler.groups == {}


def test_verifier_scheduler_ages_ready_groups_on_policy_update() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = RLConfig(orchestrator={"max_off_policy_steps": 1})
    scheduler.pending = {}
    scheduler.ready_groups = [[{"example_id": 0}], [{"example_id": 1}]]
    scheduler.ready_group_off_policy_steps = [0, 1]
    scheduler.cancelled_rollouts_count = 0

    cancelled = asyncio.run(scheduler.mark_policy_update())

    assert cancelled == 1
    assert scheduler.ready_groups == [[{"example_id": 0}]]
    assert scheduler.ready_group_off_policy_steps == [1]
    assert scheduler.cancelled_rollouts_count == 1


def test_verifier_scheduler_zero_off_policy_cancels_immediately() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = RLConfig(orchestrator={"max_off_policy_steps": 0})
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: _VerifierGroupState(
            example={"example_id": 0, "prompt": "x"},
            rollouts_to_schedule=1,
        )
    }
    scheduler.cancelled_rollouts_count = 0

    async def pending_rollout() -> list[dict[str, float]]:
        await asyncio.sleep(60)
        return [{"example_id": 0, "reward": 1.0, "error": None}]

    async def run() -> None:
        task = asyncio.create_task(pending_rollout())
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
        )
        scheduler.pending_clients[task] = 0

        assert await scheduler.mark_policy_update() == 1
        assert task.cancelled()

    asyncio.run(run())

    assert scheduler.pending == {}
    assert scheduler.groups == {}
    assert scheduler.cancelled_rollouts_count == 1


def test_verifier_scheduler_pops_ready_group_age_when_consumed() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "filter_zero_advantage": False,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 1
    scheduler.target_groups = 1
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = [
        [
            {
                "example_id": 0,
                "reward": 1.0,
                "advantage": 1.0,
                "trajectory": _trainable_trajectory(),
            }
        ]
    ]
    scheduler.ready_group_off_policy_steps = [3]
    scheduler._fill_inflight = lambda: None  # type: ignore[method-assign]

    records = asyncio.run(scheduler.generate_batch(target_groups=1))

    assert len(records) == 1
    assert scheduler.ready_groups == []
    assert scheduler.ready_group_off_policy_steps == []


def test_verifier_scheduler_salts_new_requests_with_loaded_policy() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = RLConfig()
    scheduler.model = "base"
    scheduler.policy_step = None

    assert (
        "cache_salt" not in scheduler._sampling_args_for_current_policy()["extra_body"]
    )

    scheduler.set_policy_step(3, model_name="policy")

    assert (
        scheduler._sampling_args_for_current_policy()["extra_body"]["cache_salt"] == "3"
    )
    assert scheduler.model == "policy"


def test_incomplete_verifier_groups_are_not_trainable() -> None:
    trajectory = _trainable_trajectory()
    outputs = [
        {"example_id": 0, "reward": 0.0, "advantage": -0.5, "trajectory": trajectory},
        {"example_id": 0, "reward": 1.0, "advantage": 0.5, "trajectory": trajectory},
    ]

    assert _is_usable_training_group(
        outputs,
        expected_rollouts=2,
        filter_zero_advantage=True,
        advantage_epsilon=1e-6,
    )
    assert not _is_usable_training_group(
        outputs[:1],
        expected_rollouts=2,
        filter_zero_advantage=True,
        advantage_epsilon=1e-6,
    )


def test_verifier_advantages_group_by_environment_and_example_id() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "rollouts_per_example": 2,
        }
    )
    outputs = [
        {"env_name": "a", "example_id": 1, "reward": 1.0},
        {"env_name": "a", "example_id": 1, "reward": 0.0},
        {"env_name": "b", "example_id": 1, "reward": 1.0},
        {"env_name": "b", "example_id": 1, "reward": 1.0},
    ]

    _assign_rollout_advantages(outputs, config)

    assert outputs[0]["advantage"] == pytest.approx(0.5)
    assert outputs[1]["advantage"] == pytest.approx(-0.5)
    assert outputs[2]["advantage"] == pytest.approx(0.0)
    assert outputs[3]["advantage"] == pytest.approx(0.0)


def test_verifier_group_advantages_preserve_normalization() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "normalize_group_advantages": True,
        }
    )
    outputs = [
        {"example_id": 1, "reward": 1.0},
        {"example_id": 1, "reward": 0.0},
        {"example_id": 1, "reward": 0.0},
    ]

    _assign_rollout_advantages(outputs, config)

    assert [output["advantage"] for output in outputs] == pytest.approx(
        [2**0.5, -(0.5**0.5), -(0.5**0.5)]
    )


def test_verifier_group_advantages_dispatch_max_rl() -> None:
    config = RLConfig(algo={"type": "max_rl"})
    outputs = [
        {"example_id": 1, "reward": 1.0},
        {"example_id": 1, "reward": 0.0},
    ]

    _assign_rollout_advantages(outputs, config)

    assert [output["advantage"] for output in outputs] == pytest.approx([1.0, -1.0])


def test_verifier_group_advantages_dispatch_external_algorithm() -> None:
    config = RLConfig(
        algo={
            "file": CUSTOM_ALGORITHM_FILE,
            "algorithm": "reward_plus_one",
            "scope": "group",
        }
    )
    outputs = [
        {"example_id": 1, "reward": 1.0},
        {"example_id": 1, "reward": 0.0},
    ]

    _assign_rollout_advantages(outputs, config)

    assert [output["advantage"] for output in outputs] == pytest.approx([2.0, 1.0])


def test_verifier_scheduler_bounds_zero_advantage_retries() -> None:
    config = RLConfig(
        orchestrator={
            "advantage_mode": "group_reward",
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "filter_zero_advantage": True,
            "zero_advantage_max_retries": 1,
        }
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 1
    scheduler.target_groups = 1
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {
        0: type(
            "Group",
            (),
            {
                "completed_outputs": [],
            },
        )()
    }
    scheduler._scheduled = 0

    async def zero_advantage_group() -> list[dict[str, float]]:
        return [
            {
                "reward": 1.0,
                "advantage": 0.0,
                "trajectory": _trainable_trajectory(),
            }
        ]

    def fill_inflight(self) -> None:
        if self.pending:
            return
        self.groups[0] = type(
            "Group",
            (),
            {
                "completed_outputs": [],
            },
        )()
        task = asyncio.create_task(zero_advantage_group())
        self.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
        )
        self.pending_clients[task] = 0
        self._scheduled += 1

    scheduler._fill_inflight = MethodType(fill_inflight, scheduler)

    with pytest.raises(RuntimeError, match="could not produce enough trainable"):
        asyncio.run(scheduler.generate_batch(target_groups=1))


def test_verifier_scheduler_rejects_partial_batch_at_retry_limit() -> None:
    with pytest.raises(RuntimeError, match="accepted 1, rejected 1"):
        VerifierRolloutScheduler._raise_if_retries_exhausted(  # noqa: SLF001
            completed_groups=2,
            max_completed_groups=2,
            accepted_groups=1,
            target_groups=2,
            rejected_groups=1,
        )


def test_policy_update_gate_blocks_new_rollout_submission() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler._policy_update_ready = asyncio.Event()
    scheduler._policy_update_ready.set()
    scheduled = []
    scheduler._schedule_next_rollout = lambda: scheduled.append(True) or False

    scheduler.begin_policy_update()
    scheduler._fill_inflight()

    assert scheduler.policy_update_in_progress is True
    assert scheduled == []

    scheduler.finish_policy_update()

    assert scheduler.policy_update_in_progress is False

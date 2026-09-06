from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import Mock

import pytest

import wavelet.orchestrator.envs as verifier_envs
from wavelet.configs.rl_config import (
    GRPOAlgorithmConfig,
    RewardAlgorithmConfig,
    RLConfig,
    RLCurriculumConfig,
)
from wavelet.data.rl_dataset import RLExample, _pretokenized_sample
from wavelet.orchestrator.queue import FileSystemRolloutSender
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.verifiers import (
    VerifierRolloutScheduler,
    _assign_rollout_advantages,
    _completed_group_outputs,
    _is_usable_training_group,
    _load_cached_env,
    _PendingVerifierRequest,
    _records_from_output,
    _resume_curriculum_state,
    _resume_environment_cursor_state,
    _rollout_environment_record_cursors,
    _rollout_records_policy_step,
    _run_all,
    _run_group,
    _sampling_args,
    _successful_rollout_outputs,
    _verifier_extra_env_kwargs,
    _VerifierBatchStats,
    _VerifierEnvRuntime,
    _VerifierFailureStats,
    _VerifierGroupState,
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


def _mixed_environment_config() -> RLConfig:
    return RLConfig(
        algo={"type": "reward"},
        data={"shuffle": False},
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "examples_per_step": 2,
            "rollouts_per_example": 1,
            "max_async_level": 1,
            "filter_zero_advantage": False,
            "envs": [
                {"id": "math", "ratio": 1.0},
                {
                    "id": "code",
                    "ratio": 3.0,
                    "group_size": 2,
                    "algo": {"type": "grpo"},
                },
            ],
        },
    )


CUSTOM_ALGORITHM_FILE = Path(__file__).parent / "fixtures" / "custom_algorithm.py"


def test_verifier_example_moves_legacy_task_route_to_info() -> None:
    record = RLExample(
        prompt=[{"role": "user", "content": "abc"}],
        completion=[],
        advantage=None,
        reward=None,
        metadata={
            "verifier_example": {
                "example_id": 7,
                "prompt": [{"role": "user", "content": "abc"}],
                "task": "reverse-text",
                "info": {"split": "train"},
            }
        },
    )

    example = verifier_envs._verifier_example(record)

    assert "task" not in example
    assert example["info"] == {"split": "train", "env_id": "reverse-text"}


def test_verifier_example_preserves_structured_task_payload() -> None:
    task = {"name": "solve", "arguments": {"value": 3}}
    record = RLExample(
        prompt=[],
        completion=[],
        advantage=None,
        reward=None,
        metadata={"verifier_example": {"task": task}},
    )

    assert verifier_envs._verifier_example(record)["task"] == task


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
        "input_token_count": 3,
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
        "_wavelet_policy_end_step": 4,
        "trajectory": _trainable_trajectory(),
    }

    record = _records_from_output(output)[0]

    assert record.metadata["policy_step"] == 3
    assert record.metadata["policy_end_step"] == 4
    assert _rollout_records_policy_step([record], fallback=9) == 3


def test_verifier_scheduler_snapshots_policy_when_request_completes() -> None:
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.policy_step = 4

    async def run() -> _PendingVerifierRequest:
        task = asyncio.create_task(asyncio.sleep(0, result=[]))
        request = _PendingVerifierRequest(
            group_id=1,
            client_index=0,
            rollout_count=1,
            policy_step=2,
        )
        scheduler.pending = {task: request}
        task.add_done_callback(scheduler._record_request_completion)
        await task
        await asyncio.sleep(0)
        return request

    request = asyncio.run(run())

    assert request.completed_policy_step == 4


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
        },
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
    scheduler._record_order_epoch = None
    scheduler._record_order = []

    selected = [scheduler._next_record().source for _ in range(7)]

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
        scheduler._record_order_epoch = None
        scheduler._record_order = []
        return [scheduler._next_record().source for _ in range(10)]

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


def test_adaptive_concurrency_cancels_youngest_complete_group() -> None:
    async def run() -> None:
        scheduler = object.__new__(VerifierRolloutScheduler)
        scheduler.pending = {}
        scheduler.pending_clients = {}
        scheduler.groups = {}
        scheduler.cancelled_rollouts_count = 0
        scheduler.adaptive_cancelled_rollouts = 0
        tasks: list[asyncio.Task[list[dict[str, Any]]]] = []
        gate = asyncio.Event()
        for group_id in range(3):
            task = asyncio.create_task(gate.wait())  # type: ignore[arg-type]
            tasks.append(task)
            scheduler.pending[task] = _PendingVerifierRequest(
                group_id=group_id,
                client_index=0,
                rollout_count=1,
            )
            scheduler.pending_clients[task] = 0
            scheduler.groups[group_id] = _VerifierGroupState(
                example={"id": group_id},
                rollouts_to_schedule=0,
            )

        cancelled = scheduler._cancel_youngest_requests(1)
        await asyncio.sleep(0)

        assert cancelled == 1
        assert tasks[2].cancelled()
        assert set(scheduler.groups) == {0, 1}
        assert {request.group_id for request in scheduler.pending.values()} == {0, 1}
        assert scheduler.adaptive_cancelled_rollouts == 1

        for task in scheduler.pending:
            task.cancel()
        await asyncio.gather(*scheduler.pending, return_exceptions=True)

    asyncio.run(run())


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
        accepted = scheduler._consume_completed_task(
            tasks[0],
            target_groups=1,
            outputs=outputs,
            accepted_groups=0,
        )
        rejected = scheduler._consume_completed_task(
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
    assert record.metadata["input_token_count"] == 2
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
        "timing": {
            "generation_ms": 800.0,
            "scoring_ms": 200.0,
            "total_ms": 1100.0,
        },
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
        "timing_seconds": {
            "generation": 0.8,
            "scoring": 0.2,
            "total": 1.1,
        },
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
    failure_stats = _VerifierFailureStats()
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
        ],
        failure_stats=failure_stats,
    )

    assert outputs == [
        {
            "example_id": 2,
            "reward": 1.0,
            "error": None,
            "trajectory": trainable_trajectory,
        }
    ]
    assert failure_stats.consume_metrics() == {
        "fate/errors/invalid_result": 1.0,
        "fate/errors/missing_reward": 1.0,
        "fate/errors/runtime_error": 1.0,
        "fate/errors/timeout": 1.0,
    }
    assert failure_stats.consume_metrics() == {}


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


def test_verifier_scheduler_cancels_incomplete_group_after_policy_change() -> None:
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
        return scheduler._consume_completed_task(
            task,
            target_groups=1,
            outputs=[],
            accepted_groups=0,
        )

    # The swap made the group unfinishable; it is cancelled work, not a
    # reward-filter rejection, so it must not consume the retry budget.
    assert asyncio.run(run()) == (0, 0, 0)
    assert scheduler.groups == {}
    assert scheduler.cancelled_rollouts_count == 0


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


def test_verifier_scheduler_stops_after_token_budget_and_buffers_extra_group() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "token_batch_size": 3,
            "rollouts_per_example": 1,
            "max_async_level": 1,
            "filter_zero_advantage": False,
        },
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 1
    scheduler.target_groups = None
    scheduler.target_tokens = 3
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = [
        [
            {
                "example_id": example_id,
                "reward": 1.0,
                "advantage": 1.0,
                "trajectory": _trainable_trajectory(),
            }
        ]
        for example_id in range(4)
    ]
    scheduler.ready_group_off_policy_steps = [0, 0, 0, 0]
    scheduler._fill_inflight = lambda: None  # type: ignore[method-assign]

    records = asyncio.run(scheduler.generate_batch())

    assert len(records) == 3
    assert sum(len(record.input_ids or []) for record in records) == 3
    assert len(scheduler.ready_groups) == 1
    assert scheduler.last_batch_metrics["generation/batch/tokens"] == 3.0


def test_token_batch_covers_one_distributed_micro_batch() -> None:
    config = RLConfig(
        launcher={"mode": "process", "trainer_num_processes": 4},
        data={"micro_batch_size": 1},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "token_batch_size": 1,
            "rollouts_per_example": 2,
            "max_async_level": 1,
            "filter_zero_advantage": False,
        },
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 2
    scheduler.target_groups = None
    scheduler.target_tokens = 1
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = [
        [
            {
                "example_id": example_id,
                "reward": 1.0,
                "advantage": 1.0,
                "trajectory": _trainable_trajectory(),
            }
            for _ in range(2)
        ]
        for example_id in range(3)
    ]
    scheduler.ready_group_off_policy_steps = [0, 0, 0]
    scheduler._fill_inflight = lambda: None  # type: ignore[method-assign]

    records = asyncio.run(scheduler.generate_batch())

    assert len(records) == 4
    assert len(scheduler.ready_groups) == 1


def test_token_batch_retry_limit_counts_rejected_groups_not_short_groups() -> None:
    VerifierRolloutScheduler._raise_if_retries_exhausted(
        completed_groups=4,
        max_completed_groups=2,
        accepted_groups=4,
        target_groups=None,
        accepted_tokens=4,
        target_tokens=8,
        rejected_groups=0,
    )
    with pytest.raises(RuntimeError, match=r"4 token\(s\) across 4 group"):
        VerifierRolloutScheduler._raise_if_retries_exhausted(
            completed_groups=6,
            max_completed_groups=2,
            accepted_groups=4,
            target_groups=None,
            accepted_tokens=4,
            target_tokens=8,
            rejected_groups=2,
        )


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
        VerifierRolloutScheduler._raise_if_retries_exhausted(
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


def _environment_runtime(
    config: RLConfig,
    index: int,
    records: list[RLExample],
) -> _VerifierEnvRuntime:
    env_config = config.orchestrator.envs[index]
    return _VerifierEnvRuntime(
        config=env_config,
        env=object(),
        env_name=env_config.resolved_name,
        records=records,
        sampling=config.resolved_train_sampling(env_config),
        algorithm_config=env_config.algo or config.algo,
        rollout_count=(
            env_config.group_size or config.orchestrator.rollouts_per_example or 1
        ),
        requires_group_scoring=False,
    )


def _source_record(prompt: str) -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": prompt}],
        completion=[],
        advantage=None,
        reward=None,
    )


def _bounded_group_scheduler(
    *,
    oversampling_factor: float = 1.0,
    max_inflight_rollouts: int = 128,
    max_async_level: int = 1,
    max_off_policy_steps: int = 0,
) -> VerifierRolloutScheduler:
    config = RLConfig(
        data={"shuffle": False},
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "examples_per_step": 8,
            "rollouts_per_example": 16,
            "oversampling_factor": oversampling_factor,
            "max_inflight_rollouts": max_inflight_rollouts,
            "max_async_level": max_async_level,
            "max_off_policy_steps": max_off_policy_steps,
            "filter_zero_advantage": False,
            "envs": [{"id": "verifier", "group_size": 16}],
        },
    )
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.rollout_count = 16
    scheduler._minimum_rollout_count = 16
    scheduler.target_groups = 8
    scheduler.target_tokens = None
    scheduler.clients = [object()]
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = []
    scheduler.policy_step = 0
    scheduler.next_group_id = 0
    scheduler.env_selection_cursor = 0
    scheduler.concurrency_controller = None
    scheduler.cancelled_rollouts_count = 0
    scheduler.requires_group_scoring = False
    record = _source_record("sort")
    runtime = _environment_runtime(config, 0, [record])
    scheduler.env_runtimes = [runtime]
    scheduler._next_environment_record = lambda: (runtime, record, 0, None)

    def schedule_group_rollout(
        self: VerifierRolloutScheduler,
        group_id: int,
        group: _VerifierGroupState,
    ) -> None:
        group.rollouts_to_schedule = 0
        request = object()
        self.pending[request] = _PendingVerifierRequest(  # type: ignore[index]
            group_id=group_id,
            client_index=0,
            rollout_count=16,
            policy_step=0,
        )
        self.pending_clients[request] = 0  # type: ignore[index]

    scheduler._schedule_group_rollout = MethodType(
        schedule_group_rollout,
        scheduler,
    )
    scheduler._set_group_admission_target(8, accepted_groups=0)
    return scheduler


def _remove_bounded_candidate(
    scheduler: VerifierRolloutScheduler,
    group_id: int,
) -> None:
    request = next(
        request
        for request, info in scheduler.pending.items()
        if info.group_id == group_id
    )
    scheduler.pending.pop(request)
    scheduler.pending_clients.pop(request)
    scheduler.groups.pop(group_id)


def test_group_admission_does_not_refill_accepted_stragglers() -> None:
    scheduler = _bounded_group_scheduler()

    scheduler._fill_inflight()

    assert len(scheduler.groups) == 8
    assert scheduler.inflight_rollout_count == 128
    assert scheduler.next_group_id == 8

    # A completed group buffered for this batch is still a candidate, so the
    # newly available request capacity must not admit a ninth group.
    _remove_bounded_candidate(scheduler, 0)
    scheduler.ready_groups.append([{"example_id": 0}])
    scheduler._fill_inflight()
    assert scheduler.next_group_id == 8

    # Once that group is accepted, it leaves the active candidate set but also
    # reduces the remaining batch target. Repeated straggler completions cannot
    # refill the freed slots with unrelated groups.
    scheduler.ready_groups.clear()
    scheduler._set_group_admission_target(8, accepted_groups=1)
    for accepted_groups, group_id in enumerate(range(1, 8), start=2):
        _remove_bounded_candidate(scheduler, group_id)
        scheduler._set_group_admission_target(
            8,
            accepted_groups=accepted_groups,
        )
        scheduler._fill_inflight()
        assert scheduler.next_group_id == 8


def test_group_admission_replaces_rejected_candidate() -> None:
    scheduler = _bounded_group_scheduler()
    scheduler._fill_inflight()

    _remove_bounded_candidate(scheduler, 0)
    scheduler._fill_inflight()

    assert len(scheduler.groups) == 8
    assert scheduler.inflight_rollout_count == 128
    assert scheduler.next_group_id == 9


def test_group_admission_preserves_configured_oversampling() -> None:
    scheduler = _bounded_group_scheduler(
        oversampling_factor=2.0,
        max_inflight_rollouts=256,
    )

    scheduler._fill_inflight()

    assert len(scheduler.groups) == 16
    assert scheduler.inflight_rollout_count == 256
    assert scheduler.next_group_id == 16


def test_group_prewarm_skips_policy_outside_next_rollout_window() -> None:
    scheduler = _bounded_group_scheduler(
        max_async_level=3,
        max_off_policy_steps=8,
    )
    scheduler.policy_step = 4

    prewarmed = scheduler._prewarm_next_batch(8, rollout_step=7)

    assert prewarmed is False
    assert scheduler.groups == {}
    assert scheduler.pending == {}
    assert scheduler.next_group_id == 0


def test_group_prewarm_keeps_policy_at_next_rollout_window_boundary() -> None:
    scheduler = _bounded_group_scheduler(
        max_async_level=3,
        max_off_policy_steps=8,
    )
    scheduler.policy_step = 4

    prewarmed = scheduler._prewarm_next_batch(8, rollout_step=6)

    assert prewarmed is True
    assert scheduler.rollout_step == 6
    assert len(scheduler.groups) == 8
    assert scheduler.inflight_rollout_count == 128


def test_verifier_scheduler_loads_each_environment_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _mixed_environment_config()
    envs = [
        env.model_copy(
            update={
                "data_path": tmp_path / f"{env.resolved_name}.jsonl",
                "curriculum": (
                    RLCurriculumConfig(gates={"signal": {"type": "advantage_range"}})
                    if index == 0
                    else None
                ),
            }
        )
        for index, env in enumerate(base.orchestrator.envs)
    ]
    config = base.model_copy(
        update={
            "data": base.data.model_copy(update={"path": tmp_path / "shared.jsonl"}),
            "orchestrator": base.orchestrator.model_copy(update={"envs": envs}),
        }
    )
    loaded_envs: list[tuple[str, dict[str, Any]]] = []

    def load_env(_vf, env_id, env_args, _extra):
        loaded_envs.append((env_id, env_args))
        env = type("Env", (), {"requires_group_scoring": env_id == "code"})()
        return env, False

    def load_records(data_config):
        return [_source_record(Path(data_config.path).stem)]

    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._load_verifiers", lambda _: object()
    )
    monkeypatch.setattr("wavelet.orchestrator.scheduler._load_cached_env", load_env)
    monkeypatch.setattr("wavelet.orchestrator.scheduler.load_rl_records", load_records)
    monkeypatch.setattr(
        "wavelet.orchestrator.scheduler._verifier_clients",
        lambda _vf, _config: [object()],
    )

    scheduler = VerifierRolloutScheduler(
        RLOrchestrator(config),
        start_environment_record_cursors={"math": 3, "code": 5},
        start_environment_selection_cursor=8,
    )

    assert loaded_envs == [("math", {}), ("code", {})]
    assert [
        runtime.records[0].prompt[0]["content"] for runtime in scheduler.env_runtimes
    ] == ["math", "code"]
    assert [runtime.record_cursor for runtime in scheduler.env_runtimes] == [3, 5]
    assert [runtime.rollout_count for runtime in scheduler.env_runtimes] == [1, 2]
    assert [runtime.requires_group_scoring for runtime in scheduler.env_runtimes] == [
        False,
        True,
    ]
    assert scheduler.env_selection_cursor == 8
    assert scheduler.env_runtimes[0].curriculum is not None
    assert scheduler.curriculum_state_snapshot()["math"]["sampler"] == {
        "cursor": 3,
        "task_count": 1,
    }


def test_verifier_scheduler_selects_weighted_environments_with_independent_cursors() -> (
    None
):
    config = _mixed_environment_config()
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.env_selection_cursor = 0
    scheduler.record_cursor = 0
    scheduler.env_runtimes = [
        _environment_runtime(
            config,
            0,
            [_source_record("math-0"), _source_record("math-1")],
        ),
        _environment_runtime(
            config,
            1,
            [_source_record("code-0"), _source_record("code-1")],
        ),
    ]

    selected = [scheduler._next_environment_record()[0].env_name for _ in range(400)]

    assert 70 <= selected.count("math") <= 130
    assert 270 <= selected.count("code") <= 330
    assert sum(runtime.record_cursor for runtime in scheduler.env_runtimes) == 400


def test_verifier_scheduler_tracks_absolute_environment_cursors_across_epochs() -> None:
    config = _mixed_environment_config()
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    runtime = _environment_runtime(
        config,
        0,
        [_source_record("a"), _source_record("b")],
    )
    runtime.record_cursor = 1

    first_cursor, first, _ = scheduler._next_runtime_record(runtime)
    second_cursor, second, _ = scheduler._next_runtime_record(runtime)

    assert (first_cursor, first.prompt[0]["content"]) == (1, "b")
    assert (second_cursor, second.prompt[0]["content"]) == (2, "a")


def test_verifier_scheduler_applies_group_size_and_algorithm_per_environment() -> None:
    config = _mixed_environment_config()
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 2
    scheduler.requires_group_scoring = False
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.groups = {}
    scheduler.ready_groups = []
    scheduler.ready_group_off_policy_steps = []

    def output(reward: float) -> dict[str, Any]:
        return {
            "reward": reward,
            "error": None,
            "sampling_args": {"temperature": 1.0},
            "trajectory": _trainable_trajectory(),
        }

    async def run() -> list[RLExample]:
        loop = asyncio.get_running_loop()
        reward_task = loop.create_future()
        reward_task.set_result([output(2.0)])
        grpo_task = loop.create_future()
        grpo_task.set_result([output(0.0), output(2.0)])
        scheduler.pending[reward_task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
        )
        scheduler.pending[grpo_task] = _PendingVerifierRequest(
            group_id=1,
            client_index=0,
            rollout_count=2,
        )
        scheduler.groups[0] = _VerifierGroupState(
            example={"example_id": "math"},
            rollouts_to_schedule=0,
            env_name="math",
            rollout_count=1,
            algorithm_config=RewardAlgorithmConfig(),
            record_cursor=5,
        )
        scheduler.groups[1] = _VerifierGroupState(
            example={"example_id": "code"},
            rollouts_to_schedule=0,
            env_name="code",
            rollout_count=2,
            algorithm_config=GRPOAlgorithmConfig(),
            record_cursor=7,
        )
        outputs: list[dict[str, Any]] = []
        accepted, _, _ = scheduler._consume_completed_task(
            reward_task,
            target_groups=2,
            outputs=outputs,
            accepted_groups=0,
        )
        scheduler._consume_completed_task(
            grpo_task,
            target_groups=2,
            outputs=outputs,
            accepted_groups=accepted,
        )
        return [
            record
            for completed_output in outputs
            for record in _records_from_output(completed_output)
        ]

    records = asyncio.run(run())

    assert [record.source for record in records] == ["math", "code", "code"]
    assert [record.advantage for record in records] == [2.0, -1.0, 1.0]
    assert [record.metadata["_wavelet_group_size"] for record in records] == [1, 2, 2]
    assert [record.metadata["verifier_record_cursor"] for record in records] == [
        5,
        7,
        7,
    ]


def test_verifier_scheduler_observes_curriculum_and_applies_gate() -> None:
    config = RLConfig(
        algo={"type": "reward"},
        orchestrator={
            "examples_per_step": 1,
            "rollouts_per_example": 1,
            "filter_zero_advantage": False,
        },
    )
    curriculum = Mock()
    curriculum.on_result.return_value = False
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.config = config
    scheduler.orchestrator = RLOrchestrator(config)
    scheduler.rollout_count = 1
    scheduler.requires_group_scoring = False
    scheduler.pending = {}
    scheduler.pending_clients = {}
    scheduler.ready_groups = []
    scheduler.ready_group_off_policy_steps = []

    async def run() -> tuple[tuple[int, int, int], list[dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        task = loop.create_future()
        result = {
            "reward": 1.0,
            "trajectory": _trainable_trajectory(),
        }
        task.set_result([result])
        scheduler.pending[task] = _PendingVerifierRequest(
            group_id=0,
            client_index=0,
            rollout_count=1,
        )
        scheduler.groups = {
            0: _VerifierGroupState(
                example={"example_id": 0},
                rollouts_to_schedule=0,
                curriculum=curriculum,
                curriculum_task_key="record:0",
            )
        }
        outputs: list[dict[str, Any]] = []
        decision = scheduler._consume_completed_task(
            task,
            target_groups=1,
            outputs=outputs,
            accepted_groups=0,
        )
        return decision, outputs

    decision, outputs = asyncio.run(run())

    assert decision == (0, 1, 1)
    assert outputs == []
    curriculum.on_result.assert_called_once()
    assert curriculum.on_result.call_args.args[0] == "record:0"


def test_environment_record_cursors_round_trip_through_queue_manifests(
    tmp_path: Path,
) -> None:
    config = _mixed_environment_config().model_copy(update={"output_dir": tmp_path})
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    source = tmp_path / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    sender.publish(
        source,
        step=0,
        optimizer_step=0,
        environment_record_cursors={"code": [1, 3], "math": [0]},
        environment_next_record_cursors={"code": 4, "math": 1},
        environment_selection_cursor=5,
    )
    sender.publish(
        source,
        step=1,
        optimizer_step=0,
        environment_record_cursors={"math": [2]},
        environment_next_record_cursors={"code": 4, "math": 3},
        environment_selection_cursor=7,
    )

    cursors, selection_cursor = _resume_environment_cursor_state(
        config,
        before_queue_step=2,
    )

    assert cursors == {"code": 4, "math": 3}
    assert selection_cursor == 7


def test_environment_cursor_resume_requires_preceding_manifest(tmp_path: Path) -> None:
    config = _mixed_environment_config().model_copy(update={"output_dir": tmp_path})

    with pytest.raises(ValueError, match="queue step 0 is unavailable"):
        _resume_environment_cursor_state(config, before_queue_step=1)


def test_curriculum_state_round_trips_through_queue_manifest(tmp_path: Path) -> None:
    base = _mixed_environment_config()
    envs = [
        base.orchestrator.envs[0].model_copy(
            update={
                "curriculum": RLCurriculumConfig(sampler={"type": "difficulty_pool"})
            }
        ),
        base.orchestrator.envs[1],
    ]
    config = base.model_copy(
        update={
            "output_dir": tmp_path,
            "orchestrator": base.orchestrator.model_copy(update={"envs": envs}),
        }
    )
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    source = tmp_path / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    state = {
        "math": {
            "gates": {},
            "sampler": {
                "rng": [3, [1, 2, 3], None],
                "selections": 4,
                "task_count": 2,
                "task_rewards": {"record:0": 0.5},
            },
        }
    }
    sender.publish(
        source,
        step=0,
        optimizer_step=0,
        curriculum_state=state,
    )

    assert _resume_curriculum_state(config, before_queue_step=1) == state


def test_rollout_environment_record_cursors_deduplicate_trajectory_rows() -> None:
    records = [
        RLExample(
            prompt=[],
            completion=[],
            advantage=1.0,
            reward=1.0,
            source="math",
            metadata={"verifier_record_cursor": 4},
        ),
        RLExample(
            prompt=[],
            completion=[],
            advantage=1.0,
            reward=1.0,
            source="math",
            metadata={"verifier_record_cursor": 4},
        ),
        RLExample(
            prompt=[],
            completion=[],
            advantage=1.0,
            reward=1.0,
            source="code",
            metadata={"verifier_record_cursor": 9},
        ),
    ]

    assert _rollout_environment_record_cursors(records) == {
        "code": [9],
        "math": [4],
    }

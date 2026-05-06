from __future__ import annotations

import asyncio
import json
import random
from types import MethodType
from typing import Any

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample, _pretokenized_sample
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.verifiers import (
    _PendingVerifierRequest,
    _VerifierGroupState,
    VerifierRolloutScheduler,
    _assign_rollout_advantages,
    _completed_group_outputs,
    _is_usable_training_group,
    _load_cached_env,
    _records_from_output,
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
    }
    assert record.prompt[1]["step_loss_mask"] == 0
    assert record.source == "alphabet-sort"


def test_custom_verifier_rollout_function_loads_without_env_import() -> None:
    orchestrator = RLOrchestrator(
        RLConfig(
            orchestrator={
                "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts"
            }
        )
    )

    function = orchestrator._load_custom_rollout_function(
        "wavelet.orchestrator.verifiers:generate_rollouts"
    )

    assert function.__name__ == "generate_rollouts"


def test_verifier_sampling_args_preserve_extra_body() -> None:
    config = RLConfig(
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
            normalize_group_advantages=False,
            length_penalty=None,
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
            normalize_group_advantages=False,
            advantage_epsilon=1e-6,
            length_penalty=None,
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
            normalize_group_advantages=False,
            length_penalty=None,
            env_name="group-env",
        )
        return env, outputs

    env, outputs = asyncio.run(run())

    assert env.group_calls == 1
    assert env.rollout_calls == 0
    assert len(outputs) == 2
    assert {output["env_name"] for output in outputs} == {"group-env"}
    assert [output["advantage"] for output in outputs] == [-0.5, 0.5]


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


def test_verifier_scheduler_samples_records_randomly_with_replacement() -> None:
    records = [
        RLExample(prompt=[], completion=[], advantage=0.0, reward=0.0, source=str(i))
        for i in range(5)
    ]
    scheduler = object.__new__(VerifierRolloutScheduler)
    scheduler.records = records
    scheduler.rng = random.Random(123)

    selected = [scheduler._next_record().source for _ in range(8)]  # noqa: SLF001

    assert selected == ["0", "2", "0", "3", "2", "0", "0", "3"]


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
            )
            scheduler.pending_clients[task] = 0
            scheduler.groups[group_id] = _VerifierGroupState(
                example={"example_id": group_id},
                rollouts_to_schedule=0,
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


def test_verifier_scheduler_keeps_filtered_rollouts_for_metrics() -> None:
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

    async def run() -> list[RLExample]:
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
        return await scheduler.generate_batch(target_groups=2)

    records = asyncio.run(run())

    assert len(records) == 4
    filtered = [
        record
        for record in records
        if (record.metadata or {}).get("_wavelet_filtered_rollout")
    ]
    trainable = [record for record in records if record not in filtered]
    assert len(filtered) == 2
    assert len(trainable) == 2
    assert all(record.loss_mask == [False] for record in filtered)
    assert [record.reward for record in filtered] == [1.0, 1.0]


def test_successful_rollout_outputs_raises_on_rate_limit_exception() -> None:
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        _successful_rollout_outputs(
            [
                RuntimeError(
                    "Error code: 429 - GoUsageLimitError: 5-hour usage limit "
                    "reached"
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


def test_pretokenized_rows_match_prime_sequence_window() -> None:
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

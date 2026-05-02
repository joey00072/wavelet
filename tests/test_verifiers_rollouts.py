from __future__ import annotations

import asyncio
from types import MethodType

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample, _pretokenized_sample
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.verifiers import (
    VerifierRolloutScheduler,
    _completed_group_outputs,
    _is_usable_training_group,
    _records_from_output,
    _sampling_args,
    _successful_rollout_outputs,
)


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
        "group_key": "7",
        "rollout_key": "7:0",
        "stop_condition": "done",
        "is_truncated": False,
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
            }
        }
    )

    args = _sampling_args(config)

    assert args["extra_body"]["return_token_ids"] is True
    assert args["extra_body"]["top_k"] == -1
    assert args["extra_body"]["min_p"] == 0.0
    assert args["extra_body"]["custom"] == "value"
    assert args["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


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
    assert sample["input_ids"] == [10, 11, 12]
    assert sample["target_ids"] == [11, 12, 13]
    assert sample["loss_mask"] == [False, True, True]


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


def test_successful_rollout_outputs_skip_exceptions_errors_and_missing_rewards() -> None:
    outputs = _successful_rollout_outputs(
        [
            RuntimeError("boom"),
            {"example_id": 0, "error": "timeout", "reward": 0.0},
            {"example_id": 1, "completion": []},
            object(),
            {"example_id": 2, "reward": 1.0, "error": None},
        ]
    )

    assert outputs == [{"example_id": 2, "reward": 1.0, "error": None}]


def test_completed_group_outputs_treat_task_exception_as_empty_group() -> None:
    async def fail() -> list[dict[str, float]]:
        raise RuntimeError("boom")

    async def run() -> list[dict[str, float]]:
        task = asyncio.create_task(fail())
        await asyncio.gather(task, return_exceptions=True)
        return _completed_group_outputs(task)

    assert asyncio.run(run()) == []


def test_incomplete_verifier_groups_are_not_trainable() -> None:
    outputs = [
        {"example_id": 0, "reward": 0.0, "advantage": -0.5},
        {"example_id": 0, "reward": 1.0, "advantage": 0.5},
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


def test_verifier_scheduler_bounds_zero_advantage_retries() -> None:
    config = RLConfig(
        orchestrator={
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
    scheduler._scheduled = 0

    async def zero_advantage_group() -> list[dict[str, float]]:
        return [{"reward": 1.0, "advantage": 0.0}]

    def fill_inflight(self) -> None:
        if self.pending:
            return
        task = asyncio.create_task(zero_advantage_group())
        self.pending[task] = self._scheduled
        self.pending_clients[task] = 0
        self._scheduled += 1

    scheduler._fill_inflight = MethodType(fill_inflight, scheduler)

    with pytest.raises(RuntimeError, match="could not produce enough trainable"):
        asyncio.run(scheduler.generate_batch(target_groups=1))

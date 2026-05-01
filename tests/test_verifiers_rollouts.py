from __future__ import annotations

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample, _pretokenized_sample
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.verifiers import _records_from_output, _sampling_args


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

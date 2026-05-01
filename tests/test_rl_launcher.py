from __future__ import annotations

from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.entrypoints.rl_inference import (
    _colocated_trainer_device_ids,
    _use_streaming_native_scheduler,
    _wait_for_colocated_training_memory,
)
from wavelet.entrypoints.rl_trainer import _use_streaming_rollout_chunks
from wavelet.entrypoints.rl_launcher import (
    _sleep_vllm_http_servers,
    _trainer_device_group,
)
from wavelet.entrypoints.rl_vllm_openai_server import _serve_args
from wavelet.inference.http import HTTPPolicyInferenceEngine, _shift_completion_sample
from wavelet.orchestrator.rollouts import RLOrchestrator


def test_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_sleep_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_colocate_launcher_allows_explicit_trainer_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
            "trainer_cuda_visible_devices": "1",
        }
    )

    assert _trainer_device_group(config) == "1"


def test_colocate_launcher_requires_trainer_devices_for_multi_replica() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_num_replicas": 2,
            "inference_cuda_visible_devices": ["0", "1"],
        }
    )

    with pytest.raises(ValueError, match="multiple inference replicas"):
        _trainer_device_group(config)


def test_sleep_colocate_requires_synchronous_rollouts() -> None:
    with pytest.raises(ValueError, match="requires synchronous rollouts"):
        RLConfig(
            launcher={"mode": "colocate_sleep"},
            orchestrator={"max_async_level": 1},
        )


def test_sleep_colocate_allows_multi_process_trainer() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "trainer_num_processes": 2,
            "inference_cuda_visible_devices": "0,1",
        }
    )

    assert _trainer_device_group(config) == "0,1"


def test_sleep_colocate_allows_multi_replica_inference() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "inference_num_replicas": 4,
            "inference_cuda_visible_devices": ["0,1", "2,3", "4,5", "6,7"],
            "trainer_num_processes": 8,
            "trainer_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        }
    )

    assert config.launcher.inference_num_replicas == 4
    assert _trainer_device_group(config) == "0,1,2,3,4,5,6,7"


def test_sleep_colocate_enables_vllm_sleep_allocator() -> None:
    config = RLConfig(launcher={"mode": "colocate_sleep"})

    args = _serve_args(config)

    assert args.enable_sleep_mode is True


def test_sleep_colocate_resolves_memory_wait_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "inference_cuda_visible_devices": "0,1",
            "trainer_cuda_visible_devices": "2,3",
        }
    )

    assert _colocated_trainer_device_ids(config) == {"2", "3"}


def test_sleep_colocate_memory_wait_can_be_disabled(monkeypatch) -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate_sleep",
            "colocate_memory_wait_timeout_seconds": 0,
        }
    )

    def fail_query(_devices: set[str]) -> dict[str, tuple[int, int]]:
        raise AssertionError("memory query should be skipped")

    monkeypatch.setattr(
        "wavelet.entrypoints.rl_inference._query_gpu_memory_mib",
        fail_query,
    )

    _wait_for_colocated_training_memory(config)


def test_sleep_colocate_initial_sleep_targets_all_vllm_servers(monkeypatch) -> None:
    config = RLConfig(launcher={"mode": "colocate_sleep"})
    calls = []

    def fake_sleep(_config: RLConfig, *, port: int | None = None) -> None:
        calls.append(port)

    monkeypatch.setattr(
        "wavelet.entrypoints.rl_launcher._sleep_vllm_http_server",
        fake_sleep,
    )

    _sleep_vllm_http_servers(config, ports=[8000, 8001, 8002, 8003])

    assert sorted(calls) == [8000, 8001, 8002, 8003]


def test_http_inference_sleep_discards_vllm_gpu_allocations() -> None:
    config = RLConfig()
    engine = HTTPPolicyInferenceEngine(config)
    calls = []

    def fake_request_all(method: str, path: str, payload: dict) -> None:
        calls.append((method, path, payload))

    engine._request_all = fake_request_all  # type: ignore[method-assign]

    engine.sleep()

    assert calls == [("POST", "/sleep", {"level": 1})]


def test_http_openai_load_policy_uses_step_scoped_adapter_name() -> None:
    config = RLConfig(
        inference={"vllm": {"server_backend": "openai"}},
        lora={"rank": 4, "target_modules": ["q_proj"]},
    )
    engine = HTTPPolicyInferenceEngine(config)
    calls = []

    def fake_request(
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        base_url: str | None = None,
    ) -> dict:
        calls.append((method, path, payload, base_url))
        return {}

    engine._request = fake_request  # type: ignore[method-assign]

    engine.load_policy(Path("policy"), step=7)

    assert calls == [
        (
            "POST",
            "/load_policy",
            {
                "policy_dir": "policy",
                "step": 7,
                "adapter_name": "policy-000007",
            },
            "http://127.0.0.1:8000",
        )
    ]
    assert engine.policy_model_name == "policy-000007"


def test_http_openai_response_converts_to_pretokenized_rollout() -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig())
    record = RLExample(
        prompt=[{"role": "user", "content": "x"}],
        completion=[{"role": "assistant", "content": "expected"}],
        reward=None,
        advantage=0.5,
    )

    converted = engine._record_from_openai_response(
        record,
        prompt_ids=[10, 11],
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "token_ids": [12, 13],
                    "logprobs": {
                        "content": [
                            {"logprob": -0.1},
                            {"logprob": -0.2},
                        ]
                    },
                }
            ]
        },
    )

    assert converted.completion == [{"role": "assistant", "content": "answer"}]
    assert converted.target_completion == [{"role": "assistant", "content": "expected"}]
    assert converted.input_ids == [10, 11, 12]
    assert converted.target_ids == [11, 12, 13]
    assert converted.loss_mask == [False, True, True]
    assert converted.inference_logprobs == [-0.1, -0.2]


def test_http_openai_payload_sets_vllm_request_fields() -> None:
    config = RLConfig(
        inference={
            "sampling": {
                "top_k": 20,
                "min_p": 0.05,
                "extra_body": {"return_token_ids": False, "allowed_token_ids": [1, 2]},
            },
            "vllm": {"server_backend": "openai"},
        }
    )
    engine = HTTPPolicyInferenceEngine(config)
    record = RLExample(
        prompt=[{"role": "user", "content": "x"}],
        completion=[],
        reward=None,
        advantage=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )

    payload = engine._openai_payload(
        record,
        [10, 11],
        policy_model_name="policy-000123",
    )

    assert payload["model"] == "policy-000123"
    assert payload["return_token_ids"] is True
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert payload["allowed_token_ids"] == [1, 2]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in payload


def test_shift_completion_sample_masks_only_completion_tokens() -> None:
    sample = _shift_completion_sample(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        completion_logprobs=[-0.3, -0.4],
        temperature=0.7,
    )

    assert sample["input_ids"] == [1, 2, 3]
    assert sample["target_ids"] == [2, 3, 4]
    assert sample["loss_mask"] == [False, True, True]
    assert sample["inference_logprobs"] == [0.0, -0.3, -0.4]
    assert sample["temperatures"] == [0.7, 0.7, 0.7]


def test_native_process_rollouts_use_streaming_chunks() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": None,
            "examples_per_step": 4,
            "max_async_level": 4,
        },
    )

    assert _use_streaming_native_scheduler(config) is True
    assert _use_streaming_rollout_chunks(config) is True


def test_native_rollout_chunks_partition_selected_step_records(monkeypatch) -> None:
    records = [
        RLExample(
            prompt=[{"role": "user", "content": f"p{i}"}],
            completion=[],
            advantage=None,
            reward=None,
        )
        for i in range(8)
    ]
    config = RLConfig(
        data={"seed": 123},
        orchestrator={"examples_per_step": 4, "rollout_chunk_examples": 1},
    )
    orchestrator = RLOrchestrator(config)
    monkeypatch.setattr(
        "wavelet.orchestrator.rollouts.load_rl_records",
        lambda _config: records,
    )

    selected = orchestrator._select_step_records(records, seed=123, limit=4)
    chunks = [
        orchestrator._load_native_chunk_records(
            optimizer_step=0,
            chunk_index=index,
            chunk_examples=1,
            retry=0,
        )
        for index in range(4)
    ]

    assert [chunk[0] for chunk in chunks] == selected

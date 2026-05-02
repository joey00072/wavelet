from __future__ import annotations

import os
from pathlib import Path

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.entrypoints.rl_inference import (
    _colocated_trainer_device_ids,
    _use_streaming_native_scheduler,
    _wait_for_colocated_training_memory,
)
from wavelet.entrypoints.rl_launcher import (
    _sleep_vllm_http_servers,
    _trainer_device_group,
)
from wavelet.entrypoints.rl_trainer import (
    _chunks_per_step,
    _dummy_rollout_row,
    _use_streaming_rollout_chunks,
)
from wavelet.entrypoints.rl_vllm_openai_server import _serve_args
from wavelet.inference.http import HTTPPolicyInferenceEngine, _shift_completion_sample
from wavelet.inference.vllm import VLLMPolicyInferenceEngine
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


def test_streaming_chunk_resume_uses_chunk_index_space() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        orchestrator={
            "examples_per_step": 16,
            "rollout_chunk_examples": 4,
            "max_async_level": 8,
        },
    )

    assert _use_streaming_rollout_chunks(config) is True
    assert _chunks_per_step(config) == 4


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


def test_http_openai_load_policy_reuses_stable_adapter_name() -> None:
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
                "adapter_name": "policy",
                "load_inplace": True,
            },
            "http://127.0.0.1:8000",
        )
    ]
    assert engine.policy_model_name == "policy"


def test_http_openai_load_policy_stages_adapter_in_tmpfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("WAVELET_POLICY_CACHE_DIR", str(cache_root))
    policy_dir = tmp_path / "policy"
    adapter_dir = policy_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    config = RLConfig(
        inference={"vllm": {"server_backend": "openai"}},
        lora={"rank": 4, "target_modules": ["q_proj"]},
        output_dir=tmp_path / "out",
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

    engine.load_policy(policy_dir, step=7)

    payload = calls[0][2]
    assert payload is not None
    cached_policy_dir = Path(payload["policy_dir"])
    assert cached_policy_dir != policy_dir
    assert cached_policy_dir.parent.parent == cache_root / f"wavelet-policy-cache-{os.getuid()}"
    assert (cached_policy_dir / "adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert payload["adapter_name"] == "policy"
    assert payload["load_inplace"] is True


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


def test_openai_sampling_kwargs_passes_stop_strings() -> None:
    config = RLConfig(
        inference={
            "sampling": {
                "extra_body": {
                    "stop": ["</SOLUTION>"],
                    "include_stop_str_in_output": True,
                }
            },
            "vllm": {"server_backend": "openai"},
        }
    )
    engine = VLLMPolicyInferenceEngine(config)

    kwargs = engine._openai_sampling_kwargs(  # noqa: SLF001
        {
            "temperature": 1.0,
            "top_p": 1.0,
            "extra_body": config.inference.sampling.extra_body,
        }
    )

    assert kwargs["stop"] == ["</SOLUTION>"]
    assert kwargs["include_stop_str_in_output"] is True


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


def test_dummy_rollout_row_does_not_add_missing_teacher_logprobs() -> None:
    row = {
        "loss_mask": [False, True],
        "advantage": 0.5,
        "reward": 1.0,
        "inference_logprobs": [-0.1],
        "temperature": [1.0],
        "metadata": {"source": "test"},
    }

    dummy = _dummy_rollout_row(RLConfig(), row)

    assert dummy["loss_mask"] == [False, False]
    assert dummy["inference_logprobs"] == []
    assert "teacher_logprobs" not in dummy
    assert dummy["metadata"] == {
        "source": "test",
        "_wavelet_dummy_rollout": True,
    }


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

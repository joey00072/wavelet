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
    _config_path_for_role,
    _config_with_nccl_inference_world_size,
    _is_main_trainer_process as _launcher_is_main_trainer_process,
    _role_specs,
    _sleep_vllm_http_servers,
    _trainer_device_group,
    _wait_for_vllm_http_server,
)
from wavelet.entrypoints.rl_trainer import (
    _StreamingChunkAccumulator,
    _dummy_rollout_row,
    _is_main_trainer_process as _entrypoint_is_main_trainer_process,
    _should_step_streaming_rollouts,
    _use_streaming_rollout_chunks,
)
from wavelet.entrypoints.inference_server import _serve_args
from wavelet.entrypoints.inference_server import _fit_chat_request_to_context
from wavelet.inference.http import HTTPPolicyInferenceEngine, _shift_completion_sample
from wavelet.inference.vllm import VLLMPolicyInferenceEngine
from wavelet.orchestrator.queue import publish_adapter_policy_snapshot
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.schedule import chunks_per_step, rollout_chunk_examples
from wavelet.utils.policy_transfer import NCCL_READY_MARKER


class _FakeWorld:
    def __init__(self, *, is_main: bool) -> None:
        self.is_main = is_main


class _FakeTrainer:
    def __init__(self, *, is_main: bool | None) -> None:
        self.world = None if is_main is None else _FakeWorld(is_main=is_main)


def test_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_queue_lifecycle_records_are_rank_zero_only() -> None:
    assert _entrypoint_is_main_trainer_process(_FakeTrainer(is_main=None))
    assert _entrypoint_is_main_trainer_process(_FakeTrainer(is_main=True))
    assert not _entrypoint_is_main_trainer_process(_FakeTrainer(is_main=False))
    assert _launcher_is_main_trainer_process(_FakeTrainer(is_main=True))
    assert not _launcher_is_main_trainer_process(_FakeTrainer(is_main=False))


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


def test_wait_for_vllm_http_server_fails_when_service_exits(tmp_path) -> None:
    class ExitedHandle:
        log_path = tmp_path / "vllm.log"

        def poll(self) -> int:
            return 1

    config = RLConfig(
        inference={
            "http": {
                "port": 65530,
                "startup_timeout_seconds": 10.0,
            }
        },
    )

    with pytest.raises(RuntimeError, match="exited with code 1"):
        _wait_for_vllm_http_server(config, handle=ExitedHandle())  # type: ignore[arg-type]


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
    assert rollout_chunk_examples(config) == 4
    assert chunks_per_step(config) == 4


def test_rollout_chunk_examples_defaults_to_async_split() -> None:
    config = RLConfig(
        orchestrator={
            "examples_per_step": 17,
            "max_async_level": 8,
        },
    )

    assert rollout_chunk_examples(config) == 3
    assert chunks_per_step(config) == 6


def test_streaming_rollout_steps_on_chunk_boundary_with_variable_rows() -> None:
    counts = [16] * 800
    counts[32] = 17
    counts[389] = 17
    counts[584] = 14
    counts[639] = 18

    steps = 0
    accumulated_rows = 0
    accumulated_chunks = 0
    for row_count in counts:
        accumulated_rows += row_count
        accumulated_chunks += 1
        if _should_step_streaming_rollouts(
            accumulated_rows=accumulated_rows,
            accumulated_chunks=accumulated_chunks,
            target_rollout_rows=128,
            chunks_per_step=8,
        ):
            steps += 1
            accumulated_rows = 0
            accumulated_chunks = 0

    assert steps == 100
    assert accumulated_rows == 0
    assert accumulated_chunks == 0


def test_streaming_rollout_waits_for_full_chunk_group_when_rows_overshoot() -> None:
    assert not _should_step_streaming_rollouts(
        accumulated_rows=1200,
        accumulated_chunks=7,
        target_rollout_rows=1024,
        chunks_per_step=8,
    )
    assert _should_step_streaming_rollouts(
        accumulated_rows=1300,
        accumulated_chunks=8,
        target_rollout_rows=1024,
        chunks_per_step=8,
    )


def test_streaming_chunk_accumulator_preserves_chunk_step_boundary(tmp_path) -> None:
    accumulator = _StreamingChunkAccumulator()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    accumulator.buffer(first, 2)
    assert not accumulator.should_load(min_rows=4)
    accumulator.buffer(second, 3)
    assert accumulator.should_load(min_rows=4)

    paths, chunks = accumulator.drain_pending_paths()
    accumulator.mark_loaded(rows=5, chunks=chunks, loss_scale=7.0)

    assert paths == [first, second]
    assert accumulator.pending_rows == 0
    assert accumulator.accumulated_rows == 5
    assert accumulator.accumulated_chunks == 2
    assert accumulator.accumulated_loss_scale == 7.0
    assert accumulator.should_step(chunks_per_step=2)

    accumulator.reset_after_optimizer_step()
    assert accumulator.accumulated_rows == 0
    assert accumulator.accumulated_chunks == 0
    assert accumulator.accumulated_loss_scale == 0.0


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


def test_inference_server_enables_fully_sharded_loras() -> None:
    config = RLConfig(
        inference={"vllm": {"fully_sharded_loras": True}},
        lora={"rank": 32, "target_modules": ["q_proj"]},
    )

    args = _serve_args(config)

    assert args.enable_lora is True
    assert args.max_loras == 1
    assert args.max_cpu_loras == 1
    assert args.fully_sharded_loras is True
    assert args.max_lora_rank == 32


def test_inference_server_passes_quantized_load_args() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
            }
        }
    )

    args = _serve_args(config)

    assert args.quantization == "bitsandbytes"
    assert args.load_format == "bitsandbytes"


def test_inference_server_uses_nccl_worker_for_nccl_transfer() -> None:
    config = RLConfig(
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        lora=None,
        reward={"mode": "reference_match"},
        policy_transfer={"type": "nccl"},
    )

    args = _serve_args(config)

    assert (
        args.worker_extension_cls
        == "wavelet.inference.vllm_weight_update.NCCLWeightUpdateWorker"
    )


def test_process_eval_only_launcher_skips_trainer_role(tmp_path: Path) -> None:
    config = RLConfig(
        max_steps=0,
        output_dir=tmp_path,
        launcher={"mode": "process", "inference_num_replicas": 1},
        inference={"mode": "vllm_http"},
        orchestrator={
            "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts",
        },
        policy_transfer={"export_initial": True},
    )

    roles = _role_specs(
        config,
        trainer_config_path=_config_path_for_role(config, "trainer", config),
        inference_config_path=_config_path_for_role(config, "inference", config),
        inference_ports=[8100],
    )

    assert [role.name for role in roles] == ["inference_server_0", "inference"]
    assert roles[0].command == "inference-server"


def test_launcher_uses_native_inference_server_only_when_requested(tmp_path: Path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        launcher={"mode": "process", "inference_num_replicas": 1},
        inference={"mode": "vllm_http", "vllm": {"server_backend": "offline"}},
        orchestrator={
            "custom_rollout_function": "wavelet.orchestrator.verifiers:generate_rollouts",
        },
    )

    roles = _role_specs(
        config,
        trainer_config_path=_config_path_for_role(config, "trainer", config),
        inference_config_path=_config_path_for_role(config, "inference", config),
        inference_ports=[8100],
    )

    assert roles[0].command == "native-inference-server"


def test_publish_adapter_policy_snapshot_copies_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_source"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    config = RLConfig().policy_transfer

    step_dir = publish_adapter_policy_snapshot(output_dir, config, adapter, step=0)

    assert (step_dir / "STABLE").exists()
    assert (
        step_dir / "adapter" / "adapter_model.safetensors"
    ).read_bytes() == b"weights"
    assert (step_dir / "policy.json").exists()


def test_inference_server_auto_enables_qwen_tool_parser() -> None:
    config = RLConfig(model={"name": "Qwen/Qwen3-4B-Instruct-2507"})

    args = _serve_args(config)

    assert args.tool_call_parser == "hermes"
    assert args.enable_auto_tool_choice is True


def test_inference_server_auto_detects_qwen35_tool_parser() -> None:
    config = RLConfig(model={"name": "Qwen/Qwen3.5-397B-A17B"})

    args = _serve_args(config)

    assert args.tool_call_parser == "qwen3_coder"
    assert args.enable_auto_tool_choice is True


def test_inference_server_unknown_auto_tool_parser_disabled() -> None:
    config = RLConfig(model={"name": "some/unknown-model"})

    args = _serve_args(config)

    assert args.tool_call_parser is None
    assert args.enable_auto_tool_choice is False


def test_inference_server_allows_disabling_tool_parser() -> None:
    config = RLConfig(
        model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
        inference={"vllm": {"tool_call_parser": None}},
    )

    args = _serve_args(config)

    assert args.tool_call_parser is None
    assert args.enable_auto_tool_choice is False


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


def test_http_openai_nccl_load_policy_signals_ready(tmp_path: Path) -> None:
    config = RLConfig(
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        lora=None,
        reward={"mode": "reference_match"},
        policy_transfer={"type": "nccl"},
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

    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    engine.load_policy(policy_dir, step=3)

    assert (policy_dir / NCCL_READY_MARKER).is_file()
    assert calls == [
        (
            "POST",
            "/load_policy",
            {"policy_dir": str(policy_dir), "step": 3},
            "http://127.0.0.1:8000",
        )
    ]


def test_launcher_derives_nccl_inference_world_size() -> None:
    config = RLConfig(
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        lora=None,
        reward={"mode": "reference_match"},
        policy_transfer={"type": "nccl"},
        launcher={
            "inference_num_replicas": 2,
            "inference_cuda_visible_devices": ["0,1", "2,3"],
        },
    )

    resolved = _config_with_nccl_inference_world_size(
        config,
        inference_replicas=2,
    )

    assert resolved.policy_transfer.nccl_inference_world_size == 4
    assert resolved.policy_transfer.nccl_rank_offset == 1


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
    assert (
        cached_policy_dir.parent.parent
        == cache_root / f"wavelet-policy-cache-{os.getuid()}"
    )
    assert (
        cached_policy_dir / "adapter" / "adapter_model.safetensors"
    ).read_bytes() == b"weights"
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
    engine.policy_step = 123
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
        policy_model_name="policy",
    )

    assert payload["model"] == "policy"
    assert payload["return_token_ids"] is True
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert payload["allowed_token_ids"] == [1, 2]
    assert payload["cache_salt"] == "123"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in payload


def test_inference_server_fits_overlong_chat_completion_request() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.exceptions import VLLMValidationError

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        model="test-model",
        max_completion_tokens=1024,
    )
    error = VLLMValidationError(
        "This model's maximum context length is 8192 tokens. However, you "
        "requested 1024 output tokens and your prompt contains at least 7169 "
        "input tokens, for a total of at least 8193 tokens.",
        parameter="input_tokens",
        value=7169,
    )

    fitted = _fit_chat_request_to_context(
        request,
        max_model_len=8192,
        error=error,
    )

    assert fitted is not request
    assert fitted.max_completion_tokens == 1007


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
    assert "max_tokens" not in kwargs


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

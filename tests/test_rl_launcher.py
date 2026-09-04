from __future__ import annotations

import asyncio
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import wavelet.orchestrator.envs as verifier_envs
from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference import server as inference_server
from wavelet.inference.http import HTTPPolicyInferenceEngine, _shift_completion_sample
from wavelet.inference.server import _fit_chat_request_to_context, _serve_argv
from wavelet.inference.vllm import VLLMPolicyInferenceEngine
from wavelet.orchestrator import runtime
from wavelet.orchestrator.queue import (
    FileSystemRolloutSender,
    publish_adapter_policy_snapshot,
)
from wavelet.orchestrator.rollout_worker import (
    _colocated_trainer_device_ids,
    _wait_for_colocated_training_memory,
)
from wavelet.orchestrator.rollouts import RLOrchestrator
from wavelet.orchestrator.runtime import (
    _config_path_for_role,
    _config_with_nccl_inference_world_size,
    _publish_rollout_timed,
    _role_specs,
    _rollout_client_config,
    _run_integrated_launcher,
    _shared_wandb_environment,
    _sleep_vllm_http_servers,
    _trainer_device_group,
    _wait_for_vllm_http_server,
)
from wavelet.orchestrator.schedule import (
    chunks_per_step,
    rollout_chunk_examples,
    rollout_groups_for_chunk,
)
from wavelet.orchestrator.scheduler import PublishMode, resolve_rollout_schedule
from wavelet.trainer.rl_trainer import RLTrainer
from wavelet.trainer.rl_worker import (
    _dummy_rollout_row,
    _run_streaming_rollout_training,
    _StreamingChunkAccumulator,
    _use_streaming_rollout_chunks,
    _validate_rollout_batch,
    _validate_streaming_rollout_batch,
)
from wavelet.utils.policy_transfer import NCCL_READY_MARKER


class _FakeWorld:
    def __init__(self, *, is_main: bool) -> None:
        self.is_main = is_main


class _FakeTrainer:
    def __init__(self, *, is_main: bool | None) -> None:
        self.world = None if is_main is None else _FakeWorld(is_main=is_main)


def test_integrated_rollout_publish_records_loaded_policy_step() -> None:
    published: dict[str, object] = {}

    def publish(**kwargs):
        published.update(kwargs)
        return object()

    orchestrator = type("Orchestrator", (), {"publish": staticmethod(publish)})()
    inference_engine = type("InferenceEngine", (), {"policy_step": 7})()

    batch, elapsed = _publish_rollout_timed(
        orchestrator,  # type: ignore[arg-type]
        step=9,
        inference_engine=inference_engine,
    )

    assert batch is not None
    assert elapsed >= 0.0
    assert published == {
        "step": 9,
        "inference_engine": inference_engine,
        "policy_step": 7,
    }


@pytest.mark.parametrize("fails", [False, True])
def test_integrated_launcher_closes_resources(tmp_path, monkeypatch, fails) -> None:
    config = RLConfig(output_dir=tmp_path / "run")
    trainer = Mock(step=0, resume_checkpoint_dir=None)
    inference_engine = Mock()
    teardown = AsyncMock()
    rollout_loop = Mock(side_effect=RuntimeError("rollout failed") if fails else None)
    monkeypatch.setattr(runtime, "RLTrainer", Mock(return_value=trainer))
    monkeypatch.setattr(runtime, "FileSystemRolloutReceiver", Mock())
    monkeypatch.setattr(runtime, "FileSystemPolicyReceiver", Mock())
    monkeypatch.setattr(
        runtime,
        "create_policy_inference_engine",
        Mock(return_value=inference_engine),
    )
    monkeypatch.setattr(runtime, "_run_rollout_loop", rollout_loop)
    monkeypatch.setattr(
        verifier_envs,
        "_teardown_cached_verifier_envs",
        teardown,
    )

    if fails:
        with pytest.raises(RuntimeError, match="rollout failed"):
            _run_integrated_launcher(config)
    else:
        assert _run_integrated_launcher(config) == 0

    teardown.assert_awaited_once_with()
    inference_engine.close.assert_called_once_with()
    trainer.finalize.assert_called_once_with(status="failed" if fails else "completed")


def _argv_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_colocate_launcher_defaults_trainer_to_inference_devices() -> None:
    config = RLConfig(
        launcher={
            "mode": "colocate",
            "inference_cuda_visible_devices": "0",
        }
    )

    assert _trainer_device_group(config) == "0"


def test_queue_lifecycle_records_are_rank_zero_only() -> None:
    assert RLTrainer.is_main_process(_FakeTrainer(is_main=None))
    assert RLTrainer.is_main_process(_FakeTrainer(is_main=True))
    assert not RLTrainer.is_main_process(_FakeTrainer(is_main=False))


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


def test_wait_for_vllm_http_server_checks_model_listing(monkeypatch) -> None:
    config = RLConfig(model={"name": "expected/model"})
    urls: list[str] = []

    def urlopen(url: str, *, timeout: float):
        assert timeout == 5.0
        urls.append(url)
        if url.endswith("/v1/models"):
            payload = json.dumps({"data": [{"id": "expected/model"}]}).encode()
            return contextlib.nullcontext(io.BytesIO(payload))
        return contextlib.nullcontext()

    monkeypatch.setattr(runtime.urllib.request, "urlopen", urlopen)

    _wait_for_vllm_http_server(config)

    assert urls == [
        "http://127.0.0.1:8000/liveness",
        "http://127.0.0.1:8000/v1/models",
    ]


def test_vllm_liveness_route_probes_workers(monkeypatch) -> None:
    engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[None]))
    monkeypatch.setattr(inference_server, "_CONFIG", RLConfig())
    monkeypatch.setattr(inference_server, "_engine_client", lambda _request: engine)

    response = asyncio.run(inference_server.liveness(SimpleNamespace()))

    assert response == {"status": "ok"}
    engine.collective_rpc.assert_awaited_once_with("liveness_probe")


def test_vllm_liveness_route_times_out_unresponsive_workers(monkeypatch) -> None:
    async def never_responds(_method: str) -> None:
        await asyncio.Future()

    engine = SimpleNamespace(collective_rpc=Mock(side_effect=never_responds))
    config = RLConfig(inference={"http": {"liveness_timeout_seconds": 0.001}})
    monkeypatch.setattr(inference_server, "_CONFIG", config)
    monkeypatch.setattr(inference_server, "_engine_client", lambda _request: engine)

    response = asyncio.run(inference_server.liveness(SimpleNamespace()))

    assert response.status_code == 503
    assert response.body == b'{"status":"engine_unresponsive"}'


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
    assert [rollout_groups_for_chunk(config, index) for index in range(6)] == [
        3,
        3,
        3,
        3,
        3,
        2,
    ]


def test_token_batch_uses_one_dynamic_chunk_per_optimizer_step() -> None:
    config = RLConfig(
        launcher={"mode": "process"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "token_batch_size": 4096,
            "max_async_level": 2,
        },
    )

    assert _use_streaming_rollout_chunks(config) is True
    assert rollout_chunk_examples(config) == 1
    assert chunks_per_step(config) == 1
    with pytest.raises(ValueError, match="dynamic group count"):
        rollout_groups_for_chunk(config, 0)

    trainer = SimpleNamespace(step=0, world=None)
    _run_streaming_rollout_training(
        config,
        trainer,
        receiver=object(),  # type: ignore[arg-type]
        target_step=0,
    )


def test_streaming_rollout_steps_on_chunk_boundary_with_variable_rows() -> None:
    counts = [16] * 800
    counts[32] = 17
    counts[389] = 17
    counts[584] = 14
    counts[639] = 18

    steps = 0
    accumulator = _StreamingChunkAccumulator()
    for row_count in counts:
        accumulator.mark_loaded(rows=row_count, chunks=1, loss_scale=0.0)
        if accumulator.should_step(chunks_per_step=8):
            steps += 1
            accumulator.reset_after_optimizer_step()

    assert steps == 100
    assert accumulator.accumulated_rows == 0
    assert accumulator.accumulated_chunks == 0


def test_streaming_rollout_waits_for_full_chunk_group_when_rows_overshoot() -> None:
    accumulator = _StreamingChunkAccumulator(
        accumulated_rows=1200,
        accumulated_chunks=7,
    )
    assert not accumulator.should_step(chunks_per_step=8)
    accumulator.mark_loaded(rows=100, chunks=1, loss_scale=0.0)
    assert accumulator.should_step(chunks_per_step=8)


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


def test_streaming_rollout_manifest_must_match_current_step(tmp_path: Path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={
            "examples_per_step": 4,
            "rollout_chunk_examples": 2,
            "max_async_level": 2,
            "max_off_policy_steps": 2,
        },
    )
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    batch = FileSystemRolloutSender(tmp_path, config.transport).publish(
        source,
        step=2,
        optimizer_step=1,
        chunk_index=0,
        policy_step=0,
        rows=2,
    )

    _validate_streaming_rollout_batch(
        config,
        batch,
        trainer_step=1,
        row_count=2,
    )
    with pytest.raises(ValueError, match="trainer is at step 0"):
        _validate_streaming_rollout_batch(
            config,
            batch,
            trainer_step=0,
            row_count=2,
        )


def test_streaming_rollout_rejects_stale_policy_and_wrong_manifest(
    tmp_path: Path,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={
            "examples_per_step": 4,
            "rollout_chunk_examples": 2,
            "max_async_level": 2,
            "max_off_policy_steps": 1,
        },
    )
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    stale = sender.publish(
        source,
        step=4,
        optimizer_step=2,
        chunk_index=0,
        policy_step=0,
        rows=2,
    )
    wrong_rows = sender.publish(
        source,
        step=5,
        optimizer_step=2,
        chunk_index=1,
        policy_step=1,
        rows=99,
    )

    with pytest.raises(ValueError, match=r"requires a policy step in \[1, 2\]"):
        _validate_streaming_rollout_batch(
            config,
            stale,
            trainer_step=2,
            row_count=2,
        )
    with pytest.raises(ValueError, match="rows=99"):
        _validate_streaming_rollout_batch(
            config,
            wrong_rows,
            trainer_step=2,
            row_count=2,
        )


def test_synchronous_rollout_manifest_must_match_trainer_state(
    tmp_path: Path,
) -> None:
    config = RLConfig(output_dir=tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    valid = sender.publish(
        source,
        step=3,
        optimizer_step=3,
        policy_step=3,
        rows=2,
    )
    stale = sender.publish(
        source,
        step=4,
        optimizer_step=4,
        policy_step=2,
        rows=2,
    )

    _validate_rollout_batch(
        config,
        valid,
        trainer_step=3,
        row_count=2,
        chunk_index=None,
    )
    with pytest.raises(ValueError, match="policy_step=2"):
        _validate_rollout_batch(
            config,
            stale,
            trainer_step=4,
            row_count=2,
            chunk_index=None,
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

    argv = _serve_argv(config)

    assert "--enable-sleep-mode" in argv


def test_inference_server_returns_sampling_distribution_logprobs() -> None:
    argv = _serve_argv(RLConfig())

    assert _argv_value(argv, "--logprobs-mode") == "processed_logprobs"


def test_inference_server_uses_cuda_graphs_by_default() -> None:
    assert "--enforce-eager" not in _serve_argv(RLConfig())
    eager = RLConfig(inference={"vllm": {"enforce_eager": True}})
    assert "--enforce-eager" in _serve_argv(eager)


def test_inference_server_enables_fully_sharded_loras() -> None:
    config = RLConfig(
        inference={"vllm": {"fully_sharded_loras": True}},
        lora={"rank": 32, "target_modules": ["q_proj"]},
    )

    argv = _serve_argv(config)

    assert "--enable-lora" in argv
    assert _argv_value(argv, "--max-loras") == "1"
    assert _argv_value(argv, "--max-cpu-loras") == "1"
    assert "--fully-sharded-loras" in argv
    assert _argv_value(argv, "--max-lora-rank") == "32"


def test_inference_server_passes_quantized_load_args() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
            }
        }
    )

    argv = _serve_argv(config)

    assert _argv_value(argv, "--quantization") == "bitsandbytes"
    assert _argv_value(argv, "--load-format") == "bitsandbytes"


def test_inference_server_passes_explicit_extra_vllm_args() -> None:
    config = RLConfig(
        inference={
            "vllm": {
                "extra_args": {
                    "max_num_seqs": 64,
                    "compilation-config": {"cudagraph_mode": "NONE"},
                    "enable_prefix_caching": True,
                    "async_scheduling": False,
                }
            }
        }
    )

    argv = _serve_argv(config)

    assert _argv_value(argv, "--max-num-seqs") == "64"
    assert _argv_value(argv, "--compilation-config") == ('{"cudagraph_mode":"NONE"}')
    assert "--enable-prefix-caching" in argv
    assert "--no-async-scheduling" in argv


@pytest.mark.parametrize(
    "key",
    ["model", "port", "tensor_parallel_size", "enable-lora", "logprobs_mode"],
)
def test_inference_server_rejects_managed_extra_vllm_args(key: str) -> None:
    with pytest.raises(ValueError, match="Wavelet-managed"):
        RLConfig(inference={"vllm": {"extra_args": {key: "override"}}})


def test_vllm_passthrough_remains_explicit() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        RLConfig(inference={"vllm": {"max_num_seqs": 64}})


def test_inference_server_uses_nccl_worker_for_nccl_transfer() -> None:
    config = RLConfig(
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        lora=None,
        reward={"mode": "reference_match"},
        policy_transfer={"type": "nccl"},
    )

    argv = _serve_argv(config)

    assert (
        _argv_value(argv, "--worker-extension-cls")
        == "wavelet.inference.vllm_weight_update.NCCLWeightUpdateWorker"
    )


def test_process_eval_only_launcher_skips_trainer_role(tmp_path: Path) -> None:
    config = RLConfig(
        max_steps=0,
        output_dir=tmp_path,
        launcher={"mode": "process", "inference_num_replicas": 1},
        inference={"mode": "vllm_http"},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
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


def test_process_roles_receive_common_and_specific_environment(tmp_path: Path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        launcher={
            "mode": "process",
            "env_vars": {
                "common": {"SHARED": "common"},
                "inference": {"ROLE": "server", "SHARED": "inference"},
                "trainer": {"ROLE": "trainer"},
                "orchestrator": {"ROLE": "orchestrator"},
            },
        },
    )

    roles = _role_specs(
        config,
        trainer_config_path=tmp_path / "trainer.yaml",
        inference_config_path=tmp_path / "inference.yaml",
        inference_ports=[8100],
    )

    assert roles[0].env_vars == {"SHARED": "inference", "ROLE": "server"}
    assert roles[1].env_vars == {"SHARED": "common", "ROLE": "trainer"}
    assert roles[2].env_vars == {"SHARED": "common", "ROLE": "orchestrator"}


def test_process_roles_share_one_online_wandb_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    config = RLConfig(
        output_dir=tmp_path,
        launcher={"mode": "process"},
        monitor={"wandb": {"enabled": True, "mode": "online"}},
    )
    shared_env = _shared_wandb_environment(config)

    roles = _role_specs(
        config,
        trainer_config_path=tmp_path / "trainer.yaml",
        inference_config_path=tmp_path / "inference.yaml",
        inference_ports=[8100],
        wandb_shared_env=shared_env,
    )

    trainer = next(role for role in roles if role.name == "trainer")
    orchestrator = next(role for role in roles if role.name == "inference")
    inference_server = next(role for role in roles if role.name == "inference_server_0")
    assert trainer.env_vars["WANDB_RUN_ID"] == orchestrator.env_vars["WANDB_RUN_ID"]
    assert trainer.env_vars["WANDB_SHARED_LABEL"] == "trainer"
    assert trainer.env_vars["WANDB_SHARED_PRIMARY"] == "trainer"
    assert orchestrator.env_vars["WANDB_SHARED_LABEL"] == "orchestrator"
    assert orchestrator.env_vars["WANDB_SHARED_FINISHER"] == "orchestrator"
    assert "WANDB_RUN_ID" not in inference_server.env_vars
    assert (tmp_path / "wandb_run_id.txt").read_text() == shared_env["WANDB_RUN_ID"]


def test_shared_wandb_is_disabled_for_offline_runs(tmp_path: Path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        monitor={"wandb": {"enabled": True, "mode": "offline"}},
    )

    assert _shared_wandb_environment(config) == {}
    assert not (tmp_path / "wandb_run_id.txt").exists()


def test_shared_wandb_reuses_persisted_run_id_when_resuming(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    (tmp_path / "wandb_run_id.txt").write_text("existing-run", encoding="utf-8")
    config = RLConfig(
        output_dir=tmp_path,
        launcher={"mode": "process"},
        ckpt={"resume_step": 3},
        monitor={"wandb": {"enabled": True, "mode": "online"}},
    )

    assert _shared_wandb_environment(config)["WANDB_RUN_ID"] == "existing-run"


def test_launcher_environment_values_are_redacted_from_serialized_config() -> None:
    config = RLConfig(
        launcher={"env_vars": {"inference": {"SERVICE_TOKEN": "secret-value"}}}
    )

    serialized = config.model_dump_json()

    assert "secret-value" not in serialized
    assert "**********" in serialized
    assert config.launcher.env_vars.for_role("inference") == {
        "SERVICE_TOKEN": "secret-value"
    }


@pytest.mark.parametrize(
    "variable",
    ["CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT"],
)
def test_launcher_role_environment_rejects_managed_variables(variable: str) -> None:
    with pytest.raises(ValueError, match=variable):
        RLConfig(launcher={"env_vars": {"trainer": {variable: "1"}}})


def test_eval_only_base_model_keeps_served_model_name() -> None:
    config = RLConfig(
        max_steps=0,
        model={"name": "Qwen/Qwen2.5-7B-Instruct"},
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        orchestrator={
            "verifier_model": "Qwen/Qwen2.5-7B-Instruct",
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
        },
    )

    rollout_config = _rollout_client_config(config, ports=[8000])

    assert rollout_config.orchestrator.verifier_model == ("Qwen/Qwen2.5-7B-Instruct")


def test_training_lora_uses_policy_adapter_model_name() -> None:
    config = RLConfig(
        max_steps=1,
        inference={"mode": "vllm_http", "vllm": {"server_backend": "openai"}},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            )
        },
    )

    rollout_config = _rollout_client_config(config, ports=[8000])

    assert rollout_config.orchestrator.verifier_model == (
        config.policy_transfer.adapter_name
    )


def test_launcher_uses_native_inference_server_only_when_requested(
    tmp_path: Path,
) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        launcher={"mode": "process", "inference_num_replicas": 1},
        inference={"mode": "vllm_http", "vllm": {"server_backend": "offline"}},
        orchestrator={
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
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

    step_dir = publish_adapter_policy_snapshot(
        output_dir,
        config,
        adapter,
        step=0,
        metadata={"precision": {"trainer": {"torch_dtype": "bfloat16"}}},
    )

    assert (step_dir / "STABLE").exists()
    assert (
        step_dir / "adapter" / "adapter_model.safetensors"
    ).read_bytes() == b"weights"
    assert (step_dir / "policy.json").exists()
    meta = json.loads((step_dir / "policy.json").read_text(encoding="utf-8"))
    assert meta["source_adapter_path"] == str(adapter)
    assert meta["precision"]["trainer"]["torch_dtype"] == "bfloat16"


def test_inference_server_auto_enables_qwen_tool_parser() -> None:
    config = RLConfig(model={"name": "Qwen/Qwen3-4B-Instruct-2507"})

    argv = _serve_argv(config)

    assert _argv_value(argv, "--tool-call-parser") == "hermes"
    assert "--enable-auto-tool-choice" in argv


def test_inference_server_auto_detects_qwen35_tool_parser() -> None:
    config = RLConfig(model={"name": "Qwen/Qwen3.5-397B-A17B"})

    argv = _serve_argv(config)

    assert _argv_value(argv, "--tool-call-parser") == "qwen3_coder"
    assert "--enable-auto-tool-choice" in argv


def test_inference_server_unknown_auto_tool_parser_disabled() -> None:
    config = RLConfig(model={"name": "some/unknown-model"})

    argv = _serve_argv(config)

    assert "--tool-call-parser" not in argv
    assert "--enable-auto-tool-choice" not in argv


@pytest.mark.parametrize(
    ("model_name", "parser"),
    [
        ("zai-org/GLM-4.7-Flash", "glm45"),
        ("MiniMaxAI/MiniMax-M2.5", "minimax_m2_append_think"),
        ("PrimeIntellect/INTELLECT-3.1", "deepseek_r1"),
        ("Qwen/Qwen3-30B-A3B-Thinking-2507", "deepseek_r1"),
        ("Qwen/Qwen3.5-35B-A3B", "qwen3"),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron_v3"),
    ],
)
def test_inference_server_auto_detects_reasoning_parser(
    model_name: str,
    parser: str,
) -> None:
    argv = _serve_argv(RLConfig(model={"name": model_name}))

    assert _argv_value(argv, "--reasoning-parser") == parser


def test_inference_server_auto_reasoning_parser_skips_non_reasoning_models() -> None:
    argv = _serve_argv(RLConfig(model={"name": "Qwen/Qwen3-4B-Instruct-2507"}))

    assert "--reasoning-parser" not in argv


def test_inference_server_reasoning_parser_can_be_overridden_or_disabled() -> None:
    model = {"name": "Qwen/Qwen3-4B-Thinking-2507"}
    custom = RLConfig(
        model=model,
        inference={"vllm": {"reasoning_parser": "custom"}},
    )
    disabled = RLConfig(
        model=model,
        inference={"vllm": {"reasoning_parser": None}},
    )

    assert _argv_value(_serve_argv(custom), "--reasoning-parser") == "custom"
    assert "--reasoning-parser" not in _serve_argv(disabled)


def test_inference_server_allows_disabling_tool_parser() -> None:
    config = RLConfig(
        model={"name": "Qwen/Qwen3-4B-Instruct-2507"},
        inference={"vllm": {"tool_call_parser": None}},
    )

    argv = _serve_argv(config)

    assert "--tool-call-parser" not in argv
    assert "--enable-auto-tool-choice" not in argv


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
        "wavelet.orchestrator.rollout_worker._query_gpu_memory_mib",
        fail_query,
    )

    _wait_for_colocated_training_memory(config)


def test_sleep_colocate_initial_sleep_targets_all_vllm_servers(monkeypatch) -> None:
    config = RLConfig(launcher={"mode": "colocate_sleep"})
    calls = []

    def fake_sleep(_config: RLConfig, *, port: int | None = None) -> None:
        calls.append(port)

    monkeypatch.setattr(
        "wavelet.orchestrator.runtime._sleep_vllm_http_server",
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
        if payload is None:
            return {"status": "ok"}
        return {"policy_step": payload["step"]}

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
        ),
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
        if payload is None:
            return {"status": "ok"}
        return {"policy_step": payload["step"]}

    engine._request = fake_request  # type: ignore[method-assign]

    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    engine.load_policy(policy_dir, step=3)

    assert (policy_dir / NCCL_READY_MARKER).is_file()
    assert calls == [
        ("POST", "/pause", None, "http://127.0.0.1:8000"),
        (
            "POST",
            "/load_policy",
            {"policy_dir": str(policy_dir), "step": 3},
            "http://127.0.0.1:8000",
        ),
        ("POST", "/resume", None, "http://127.0.0.1:8000"),
    ]


def test_launcher_derives_nccl_inference_world_size() -> None:
    config = RLConfig(
        inference={
            "mode": "vllm_http",
            "vllm": {"server_backend": "openai", "tensor_parallel_size": 2},
        },
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


def test_http_openai_load_policy_uses_immutable_snapshot_path(
    tmp_path: Path,
) -> None:
    policy_dir = tmp_path / "policy"
    adapter_dir = policy_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (policy_dir / "policy.json").write_text(
        json.dumps({"artifact": {"bytes": len(b"weights")}}),
        encoding="utf-8",
    )
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
        if payload is None:
            return {"status": "ok"}
        return {"policy_step": payload["step"]}

    engine._request = fake_request  # type: ignore[method-assign]

    engine.load_policy(policy_dir, step=7)

    payload = calls[0][2]
    assert payload is not None
    assert Path(payload["policy_dir"]) == policy_dir
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
        orchestrator={"enabled": False},
        inference={
            "sampling": {
                "top_k": 20,
                "min_p": 0.05,
                "extra_body": {"return_token_ids": False, "allowed_token_ids": [1, 2]},
            },
            "vllm": {"server_backend": "openai"},
        },
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

    kwargs = engine._openai_sampling_kwargs(
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

    assert resolve_rollout_schedule(config).publish_mode is PublishMode.STREAMING
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

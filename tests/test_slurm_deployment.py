from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavelet.configs.config import RLConfig, SFTConfig
from wavelet.deployment import slurm
from wavelet.orchestrator.placement import http_base_urls
from wavelet.utils.serialization import dump_yaml, load_yaml


def test_slurm_worker_cli_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from wavelet import cli

    received = []

    def worker(argv: list[str] | None) -> int:
        received.append(argv)
        return 7

    monkeypatch.setattr(slurm, "main", worker)
    monkeypatch.setattr(
        "sys.argv", ["wavelet", "slurm-worker", "rl", "@", "resolved.yaml"]
    )
    assert cli.main() == 7
    assert received == [["rl", "@", "resolved.yaml"]]


@pytest.mark.parametrize("served_model", ["expected-model", "wrong-model"])
def test_slurm_readiness_checks_workers_and_model(monkeypatch, served_model) -> None:
    requested = []

    def response(url, *, timeout):
        requested.append(url)
        return io.BytesIO(json.dumps({"data": [{"id": served_model}]}).encode())

    monkeypatch.setattr(slurm.urllib.request, "urlopen", response)
    kwargs = {"timeout_seconds": 1, "expected_model_names": {"expected-model"}}
    if served_model == "wrong-model":
        with pytest.raises(ValueError, match="Expected vLLM model"):
            slurm._wait_for_http_servers([], ["http://infer:8000"], **kwargs)
    else:
        slurm._wait_for_http_servers([], ["http://infer:8000"], **kwargs)
    assert requested == [
        "http://infer:8000/health",
        "http://infer:8000/liveness",
        "http://infer:8000/v1/models",
    ]


def test_slurm_readiness_reports_server_exit_before_polling(monkeypatch) -> None:
    process = SimpleNamespace(
        name="inference_server_0",
        process=SimpleNamespace(poll=lambda: 2),
        log_file=SimpleNamespace(name="inference_server_0.log"),
    )
    with pytest.raises(RuntimeError, match="inference_server_0.log"):
        slurm._wait_for_http_servers(
            [process], ["http://infer:8000"], timeout_seconds=1
        )


def _slurm_config(tmp_path: Path) -> dict[str, object]:
    return {
        "job_name": "wavelet-test",
        "project_dir": tmp_path,
        "partition": "gpu",
        "time_limit": "01:00:00",
        "inference_cpus_per_replica": 2,
        "inference_memory_per_replica": "16G",
    }


def test_multinode_requires_slurm() -> None:
    with pytest.raises(ValueError, match="requires a slurm configuration"):
        SFTConfig(
            deployment={
                "type": "multi_node",
                "num_train_nodes": 2,
                "gpus_per_node": 8,
            }
        )


def test_multinode_rl_requires_process_launcher_and_filesystem_transfer(
    tmp_path: Path,
) -> None:
    deployment = {
        "type": "multi_node",
        "num_train_nodes": 2,
        "num_inference_nodes": 1,
        "gpus_per_node": 8,
    }
    with pytest.raises(ValueError, match="launcher.mode='process'"):
        RLConfig(deployment=deployment, slurm=_slurm_config(tmp_path))
    with pytest.raises(ValueError, match="policy_transfer.type='filesystem'"):
        RLConfig(
            deployment=deployment,
            slurm=_slurm_config(tmp_path),
            launcher={"mode": "process"},
            policy_transfer={"type": "nccl"},
        )


def test_multinode_rl_validates_per_node_inference_topology(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replica requires 4 GPU"):
        RLConfig(
            deployment={
                "type": "multi_node",
                "num_train_nodes": 1,
                "num_inference_nodes": 1,
                "gpus_per_node": 2,
            },
            slurm=_slurm_config(tmp_path),
            launcher={"mode": "process"},
            policy_transfer={"type": "filesystem"},
            inference={"vllm": {"tensor_parallel_size": 4}},
        )


def test_http_endpoints_can_span_hosts() -> None:
    config = RLConfig(
        inference={
            "http": {
                "hosts": ["infer-a", "infer-b"],
                "ports": [8000, 8000],
            }
        },
        launcher={"inference_num_replicas": 2},
    )

    assert http_base_urls(config) == [
        "http://infer-a:8000",
        "http://infer-b:8000",
    ]


def test_multinode_rejects_replica_gpu_oversubscription(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replicas per node exceed"):
        RLConfig(
            deployment={
                "type": "multi_node",
                "num_inference_nodes": 1,
                "gpus_per_node": 4,
                "inference_replicas_per_node": 2,
            },
            slurm=_slurm_config(tmp_path),
            launcher={"mode": "process"},
            inference={"vllm": {"tensor_parallel_size": 4}},
        )


def test_multinode_replicas_require_bounded_step_memory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slurm.inference_memory_per_replica"):
        RLConfig(
            deployment={
                "type": "multi_node",
                "num_inference_nodes": 1,
                "gpus_per_node": 8,
                "inference_replicas_per_node": 2,
            },
            slurm={"project_dir": tmp_path},
            launcher={"mode": "process"},
            inference={"vllm": {"tensor_parallel_size": 4}},
        )


def test_duplicate_http_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="host/port pairs must be unique"):
        RLConfig(
            inference={
                "http": {
                    "hosts": ["infer-a", "infer-a"],
                    "ports": [8000, 8000],
                }
            }
        )


def test_render_sbatch_script_uses_typed_resources(tmp_path: Path) -> None:
    config = SFTConfig(
        deployment={
            "type": "multi_node",
            "num_train_nodes": 3,
            "gpus_per_node": 8,
        },
        slurm={
            **_slurm_config(tmp_path),
            "account": "research",
            "setup_commands": ["module load cuda"],
            "extra_directives": ["--requeue"],
        },
    )

    script = slurm.render_sbatch_script(
        config,
        command="sft",
        config_path=tmp_path / "resolved.yaml",
        log_path=tmp_path / "slurm-%j.log",
    )

    assert "#SBATCH --nodes=3" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=research" in script
    assert "#SBATCH --requeue" in script
    assert "module load cuda" in script
    assert "python -m wavelet slurm-worker sft" in script


def test_dry_run_writes_script_without_submitting(tmp_path: Path, monkeypatch) -> None:
    config = SFTConfig(
        dry_run=True,
        deployment={"type": "multi_node", "num_train_nodes": 2},
        slurm=_slurm_config(tmp_path),
    )
    monkeypatch.setattr(
        slurm,
        "submit_sbatch",
        lambda _path: pytest.fail("dry run submitted an sbatch job"),
    )

    result = slurm.launch_slurm(
        config,
        command="sft",
        config_path=tmp_path / "config.yaml",
        script_path=tmp_path / "job.sbatch",
        log_path=tmp_path / "slurm-%j.log",
    )

    assert result == 0
    assert (tmp_path / "job.sbatch").is_file()


def test_sft_worker_marks_prevalidated_output(tmp_path: Path, monkeypatch) -> None:
    config = SFTConfig(
        output_dir=tmp_path / "run",
        deployment={"type": "multi_node", "num_train_nodes": 2},
        slurm=_slurm_config(tmp_path),
    )
    (config.output_dir / "configs").mkdir(parents=True)
    call = {}

    def fake_run(name, command, **kwargs):
        call["name"] = name
        call.update(command=command, **kwargs)
        return 0

    monkeypatch.setattr(slurm, "_run_to_completion", fake_run)

    assert slurm.run_sft_worker(config, hosts=["train-a", "train-b"]) == 0
    assert call["env"]["WAVELET_SLURM_OUTPUT_PREPARED"] == "1"
    assert "torch.distributed.run" in call["command"]


def test_worker_stays_pinned_to_submitted_attempt(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "run"
    submitted_dir = output_dir / "configs" / "attempt_1" / "resolved"
    newer_dir = output_dir / "configs" / "attempt_2" / "resolved"
    submitted_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    (output_dir / "configs" / "latest").symlink_to("attempt_2")
    config = SFTConfig(
        output_dir=output_dir,
        deployment={"type": "multi_node", "num_train_nodes": 2},
        slurm=_slurm_config(tmp_path),
    )
    config_path = submitted_dir / "sft.yaml"
    dump_yaml(config_path, config.model_dump(mode="json", exclude_none=True))
    seen = {}
    monkeypatch.setattr(slurm, "_allocated_hosts", lambda: ["train-a", "train-b"])

    def fake_worker(_config, *, hosts, config_dir):
        seen.update(hosts=hosts, config_dir=config_dir)
        return 0

    monkeypatch.setattr(slurm, "run_sft_worker", fake_worker)

    assert slurm.main(["sft", "@", str(config_path)]) == 0
    assert seen["config_dir"] == submitted_dir
    assert (submitted_dir / "slurm_allocation.json").is_file()
    assert not (newer_dir / "slurm_allocation.json").exists()


@pytest.mark.parametrize(
    ("transfer_type", "broadcast_host"),
    [("filesystem", "127.0.0.1"), ("nccl", "127.0.0.1"), ("nccl", "custom-host")],
)
@pytest.mark.parametrize("replicas_per_node", [1, 2])
@pytest.mark.parametrize("job_fails", [False, True])
def test_rl_worker_materializes_remote_role_endpoints(
    tmp_path: Path,
    monkeypatch,
    transfer_type: str,
    broadcast_host: str,
    replicas_per_node: int,
    job_fails: bool,
) -> None:
    output_dir = tmp_path / "run"
    config_dir = output_dir / "configs" / "attempt_1" / "resolved"
    config_dir.mkdir(parents=True)
    (output_dir / "configs" / "latest").symlink_to("attempt_1")
    config = RLConfig(
        output_dir=output_dir,
        launcher={"mode": "process", "backend": "local"},
        deployment={
            "type": "multi_node",
            "num_train_nodes": 2,
            "num_inference_nodes": 2,
            "gpus_per_node": 2 * replicas_per_node,
            "inference_replicas_per_node": replicas_per_node,
        },
        slurm=_slurm_config(tmp_path),
        lora=None,
        policy_transfer={"type": transfer_type, "nccl_host": broadcast_host},
        inference={"vllm": {"tensor_parallel_size": 2}},
        monitor={"wandb": {"enabled": True, "mode": "online"}},
    )
    started: list[tuple[str, list[str], bool]] = []
    environments: dict[str, dict[str, str]] = {}
    closed = []
    finalized = []
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)

    class FakeProcess:
        def poll(self):
            return None

    class FakeManaged:
        def __init__(self, name: str, service: bool):
            self.name = name
            self.process = FakeProcess()
            self.log_file = SimpleNamespace(name=f"{name}.log")
            self.service = service

        def close(self) -> None:
            closed.append(self.name)

    def fake_start(name, command, *, service=False, env=None, **_kwargs):
        started.append((name, command, service))
        environments[name] = env
        return FakeManaged(name, service)

    monkeypatch.setattr(slurm, "_start_process", fake_start)
    monkeypatch.setattr(slurm, "_wait_for_http_servers", lambda *_args, **_kw: None)

    def wait_for_jobs(*_args, **_kwargs):
        if job_fails:
            raise RuntimeError("trainer checkpoint failed")

    def finish_run(_config, shared_env, *, exit_code):
        assert len(closed) == len(started)
        assert shared_env["WANDB_RUN_ID"]
        finalized.append(exit_code)

    monkeypatch.setattr(slurm, "_wait_for_jobs", wait_for_jobs)
    monkeypatch.setattr("wavelet.monitor.finish_shared_wandb_run", finish_run)
    monkeypatch.setattr(slurm, "_terminate", lambda *_args, **_kw: None)

    if job_fails:
        with pytest.raises(RuntimeError, match="trainer checkpoint failed"):
            slurm.run_rl_worker(
                config, hosts=["infer-a", "infer-b", "train-a", "train-b"]
            )
        assert finalized == [1]
        return

    assert (
        slurm.run_rl_worker(
            config,
            hosts=["infer-a", "infer-b", "train-a", "train-b"],
        )
        == 0
    )

    assert finalized == [0]
    rollout = load_yaml(config_dir / "rl_inference.yaml")
    RLConfig.model_validate(rollout)
    replica_count = 2 * replicas_per_node
    assert rollout["inference"]["http"]["hosts"] == (
        ["infer-a"] * replicas_per_node + ["infer-b"] * replicas_per_node
    )
    assert rollout["inference"]["http"]["ports"] == list(
        range(8000, 8000 + replica_count)
    )
    server = load_yaml(config_dir / "inference_server_0.yaml")
    assert server["inference"]["http"]["host"] == "0.0.0.0"
    if transfer_type == "nccl":
        expected_host = "train-a" if broadcast_host == "127.0.0.1" else broadcast_host
        for role in ["rl_trainer", "rl_inference"] + [
            f"inference_server_{index}" for index in range(replica_count)
        ]:
            role_config = load_yaml(config_dir / f"{role}.yaml")
            assert role_config["policy_transfer"]["nccl_host"] == expected_host
            assert role_config["policy_transfer"]["nccl_inference_world_size"] == (
                replica_count * 2
            )
        for index in range(replica_count):
            replica = load_yaml(config_dir / f"inference_server_{index}.yaml")
            assert replica["policy_transfer"]["nccl_rank_offset"] == 1 + index * 2
    assert [name for name, _, _ in started] == [
        f"inference_server_{index}" for index in range(replica_count)
    ] + ["trainer", "inference"]
    for name, command, _ in started:
        if name.startswith("inference_server_"):
            assert command[command.index("--gpus-per-task") + 1] == "2"
            assert command[command.index("--gpus-per-node") + 1] == "2"
            assert command[command.index("--cpus-per-task") + 1] == "2"
            assert command[command.index("--mem") + 1] == "16G"
            assert "--exclusive" in command
    trainer_command = next(command for name, command, _ in started if name == "trainer")
    assert "train-a,train-b" in trainer_command
    assert "torch.distributed.run" in trainer_command
    assert "--nnodes" in trainer_command
    assert trainer_command[trainer_command.index("--nnodes") + 1] == "2"
    trainer_env = environments["trainer"]
    rollout_env = environments["inference"]
    assert trainer_env["WANDB_RUN_ID"] == rollout_env["WANDB_RUN_ID"]
    assert trainer_env["WANDB_SHARED_LABEL"] == "trainer"
    assert rollout_env["WANDB_SHARED_LABEL"] == "orchestrator"
    assert trainer_env["WANDB_SHARED_PRIMARY"] == "trainer"
    assert rollout_env["WANDB_SHARED_FINISHER"] == "launcher"
    assert (output_dir / "wandb_run_id.txt").read_text().strip() == trainer_env[
        "WANDB_RUN_ID"
    ]

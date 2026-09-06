from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wavelet.configs.config import RLConfig, SFTConfig
from wavelet.deployment import slurm
from wavelet.orchestrator.placement import http_base_urls
from wavelet.utils.serialization import dump_yaml, load_yaml


def _slurm_config(tmp_path: Path) -> dict[str, object]:
    return {
        "job_name": "wavelet-test",
        "project_dir": tmp_path,
        "partition": "gpu",
        "time_limit": "01:00:00",
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

    assert slurm.worker_main(["sft", "@", str(config_path)]) == 0
    assert seen["config_dir"] == submitted_dir
    assert (submitted_dir / "slurm_allocation.json").is_file()
    assert not (newer_dir / "slurm_allocation.json").exists()


def test_rl_worker_materializes_remote_role_endpoints(
    tmp_path: Path,
    monkeypatch,
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
            "gpus_per_node": 2,
        },
        slurm=_slurm_config(tmp_path),
        policy_transfer={"type": "filesystem"},
    )
    started: list[tuple[str, list[str], bool]] = []

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
            pass

    def fake_start(name, command, *, service=False, **_kwargs):
        started.append((name, command, service))
        return FakeManaged(name, service)

    monkeypatch.setattr(slurm, "_start_process", fake_start)
    monkeypatch.setattr(slurm, "_wait_for_http_servers", lambda *_args, **_kw: None)
    monkeypatch.setattr(slurm, "_wait_for_jobs", lambda *_args, **_kw: None)
    monkeypatch.setattr(slurm, "_terminate", lambda *_args, **_kw: None)

    assert (
        slurm.run_rl_worker(
            config,
            hosts=["infer-a", "infer-b", "train-a", "train-b"],
        )
        == 0
    )

    rollout = load_yaml(config_dir / "rl_inference.yaml")
    RLConfig.model_validate(rollout)
    assert rollout["inference"]["http"]["hosts"] == ["infer-a", "infer-b"]
    assert rollout["inference"]["http"]["ports"] == [8000, 8001]
    server = load_yaml(config_dir / "inference_server_0.yaml")
    assert server["inference"]["http"]["host"] == "0.0.0.0"
    assert [name for name, _, _ in started] == [
        "inference_server_0",
        "inference_server_1",
        "trainer",
        "inference",
    ]
    trainer_command = next(command for name, command, _ in started if name == "trainer")
    assert "train-a,train-b" in trainer_command
    assert "torch.distributed.run" in trainer_command
    assert "--nnodes" in trainer_command
    assert trainer_command[trainer_command.index("--nnodes") + 1] == "2"

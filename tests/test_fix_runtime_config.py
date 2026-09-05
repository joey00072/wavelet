from __future__ import annotations

import signal
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.configs.sft import SFTConfig, WandbConfig
from wavelet.orchestrator import launcher as launcher_module
from wavelet.orchestrator.runtime import (
    _raise_keyboard_interrupt,
    _role_config_payload,
)
from wavelet.utils.monitoring import RunMonitor
from wavelet.utils.pathing import validate_output_dir
from wavelet.utils.serialization import load_yaml

RL_EXAMPLE_CONFIGS = [
    path
    for path in sorted(Path("examples").rglob("*.yaml"))
    if path.name.startswith("rl")
]


# ── role configs must re-validate from their full dump ────────────────────────


@pytest.mark.parametrize("path", RL_EXAMPLE_CONFIGS, ids=str)
def test_rl_example_config_round_trips_through_role_payload(path: Path) -> None:
    config = RLConfig.model_validate(load_yaml(path))

    RLConfig.model_validate(_role_config_payload(config))


def test_disabled_checkpoint_block_round_trips_for_process_roles() -> None:
    config = RLConfig(ckpt={}, launcher={"mode": "process"})

    reloaded = RLConfig.model_validate(_role_config_payload(config))

    assert reloaded.ckpt is not None
    assert reloaded.ckpt.mode == "disabled"


def test_disabled_checkpoint_mode_still_rejects_explicit_interval() -> None:
    with pytest.raises(ValueError, match="ckpt.interval"):
        RLConfig(ckpt={"interval": 5})


def test_checkpoint_resume_source_is_unambiguous() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RLConfig(
            ckpt={
                "resume_step": 3,
                "resume_dir": "other/checkpoint-3",
            }
        )

    with pytest.raises(ValueError, match="checkpoint-N"):
        RLConfig(ckpt={"resume_dir": "other/latest"})


def test_checkpoint_skip_flags_require_resume() -> None:
    with pytest.raises(ValueError, match="require checkpoint resume"):
        RLConfig(ckpt={"skip_optimizer": True, "skip_progress": True})


# ── config validation ─────────────────────────────────────────────────────────


def test_inference_ports_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        RLConfig(inference={"http": {"ports": [8000, 8000]}})


def test_legacy_betas_alias_rejects_conflicting_values() -> None:
    config = RLConfig(optim={"betas": [0.8, 0.9], "betas1": 0.8})
    assert (config.optim.betas1, config.optim.betas2) == (0.8, 0.9)

    with pytest.raises(ValueError, match="disagree"):
        RLConfig(optim={"betas": [0.8, 0.9], "betas1": 0.5})


@pytest.mark.parametrize(
    "optim",
    [
        {"type": "muon"},
        {"mu": 0.95},
    ],
)
def test_unimplemented_muon_optimizer_config_is_rejected(
    optim: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="muon|mu"):
        RLConfig(optim=optim)


def test_unimplemented_fsdp_reshard_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="reshard_after_forward"):
        RLConfig(fsdp={"reshard_after_forward": False})


def test_fsdp2_accepts_no_reshard_after_forward() -> None:
    config = RLConfig(
        fsdp={"enabled": True, "impl": "fsdp2", "reshard_after_forward": False}
    )

    assert config.fsdp.impl == "fsdp2"
    assert config.fsdp.reshard_after_forward is False


@pytest.mark.parametrize("config_cls", [RLConfig, SFTConfig])
@pytest.mark.parametrize("field", ["cp", "ep"])
def test_unsupported_parallel_dimensions_are_rejected_before_runtime(
    config_cls: type[RLConfig | SFTConfig], field: str
) -> None:
    with pytest.raises(ValueError, match=rf"fsdp\.{field}=2"):
        config_cls(fsdp={field: 2})


def test_fsdp2_ring_context_parallelism_is_accepted_for_rl() -> None:
    config = RLConfig(
        model={"attn_implementation": "sdpa"},
        fsdp={"enabled": True, "impl": "fsdp2", "cp": 2},
        data={"seq_len": 128, "micro_batch_size": 1},
    )

    assert config.fsdp.cp == 2
    assert config.fsdp.cp_style == "ring"


def test_context_parallelism_is_rejected_for_sft() -> None:
    with pytest.raises(ValueError, match="supported for RLConfig only"):
        SFTConfig(
            model={"attn_implementation": "sdpa"},
            fsdp={"enabled": True, "impl": "fsdp2", "cp": 2},
            data={"pack_function": "cat", "seq_len": 128},
        )


def test_context_parallelism_requires_sdpa() -> None:
    with pytest.raises(ValueError, match="attn_implementation='sdpa'"):
        RLConfig(
            model={"attn_implementation": "auto"},
            fsdp={"enabled": True, "impl": "fsdp2", "cp": 2},
            data={"seq_len": 128},
        )


def test_unimplemented_sft_generation_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="generate"):
        SFTConfig(generate={"prompt": "test", "max_new_tokens": 8})


def test_inert_sft_deployment_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="deployment"):
        SFTConfig(deployment={"type": "single_node", "num_gpus": 2})


@pytest.mark.parametrize(
    "data",
    [
        {"pack_function": "stack"},
        {"stack_bucket_multiple": 256},
        {"stack_bucket_timeout": 10},
    ],
)
def test_unimplemented_sft_stack_packing_config_is_rejected(
    data: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="stack"):
        SFTConfig(data=data)


@pytest.mark.parametrize("config_cls", [RLConfig, SFTConfig])
@pytest.mark.parametrize("field", ["use_streams", "max_fwd_stash_size"])
def test_unimplemented_activation_offload_controls_are_rejected(
    config_cls: type[RLConfig | SFTConfig], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        config_cls(activation_offloading={field: True})


def test_process_mode_export_interval_must_fit_freshness_window() -> None:
    orchestrator = {"max_async_level": 2, "max_off_policy_steps": 1}

    RLConfig(
        launcher={"mode": "process"},
        orchestrator=orchestrator,
        policy_transfer={"export_every_steps": 2},
    )
    with pytest.raises(ValueError, match="export_every_steps=4"):
        RLConfig(
            launcher={"mode": "process"},
            orchestrator=orchestrator,
            policy_transfer={"export_every_steps": 4},
        )


def test_rl_pad_to_multiple_of_must_divide_seq_len() -> None:
    RLConfig(data={"seq_len": 16, "pad_to_multiple_of": 8})

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        RLConfig(data={"seq_len": 8, "pad_to_multiple_of": 16})


# ── destructive cleanup guard ─────────────────────────────────────────────────


def test_clean_output_dir_refuses_cwd_home_and_ancestors(
    monkeypatch, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: work))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    for target in (work, tmp_path, home):
        with pytest.raises(ValueError, match="refuses to delete"):
            validate_output_dir(target, resuming=False, clean=True)
        assert target.exists()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("{}", encoding="utf-8")
    validate_output_dir(run_dir, resuming=False, clean=True)
    assert not run_dir.exists()


# ── launcher teardown ─────────────────────────────────────────────────────────


def test_sigterm_handler_unwinds_like_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        _raise_keyboard_interrupt(signal.SIGTERM, None)


def test_role_subprocess_creates_missing_log_directory(monkeypatch, tmp_path) -> None:
    started: dict[str, object] = {}

    def fake_popen(command_args, **kwargs):
        started["args"] = command_args
        started["stdout"] = kwargs["stdout"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        launcher_module, "_wait_for_role_process", lambda process, **kwargs: 0
    )
    log_path = tmp_path / "logs" / "nested" / "rl_trainer.log"

    exit_code = launcher_module._run_role_subprocess(
        command="rl-trainer",
        config_path="config.yaml",
        cwd=str(tmp_path),
        log_path=str(log_path),
        cuda_visible_devices=None,
    )

    assert exit_code == 0
    assert log_path.exists()
    assert "rl-trainer" in started["args"]


# ── W&B logging ───────────────────────────────────────────────────────────────


class _FakeRun:
    def __init__(self) -> None:
        self.id = "run-123"
        self.logged: list[tuple[tuple, dict]] = []

    def define_metric(self, *_args, **_kwargs) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        self.logged.append((args, kwargs))

    def finish(self) -> None:
        return None


def _monitor(tmp_path: Path) -> RunMonitor:
    return RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        wandb=WandbConfig(enabled=True, project="wavelet-tests", mode="offline"),
    )


def test_wandb_rows_rely_on_step_metric_instead_of_explicit_step(
    monkeypatch, tmp_path: Path
) -> None:
    run = _FakeRun()
    monkeypatch.setitem(
        sys.modules, "wandb", types.SimpleNamespace(init=lambda **kwargs: run)
    )
    monitor = _monitor(tmp_path)
    monitor.start_run(run_config={})

    monitor.log({"eval/env/avg@8": 0.5}, step=7)
    monitor.log({"loss": 1.0}, step=3)

    assert all("step" not in kwargs for _, kwargs in run.logged)
    assert [args[0]["step"] for args, _ in run.logged] == [7, 3]


def test_wandb_resume_reuses_persisted_run_id(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def fake_init(**kwargs):
        captured.append(kwargs)
        return _FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))

    _monitor(tmp_path).start_run(run_config={})
    assert (tmp_path / "wandb_run_id.txt").read_text(encoding="utf-8") == "run-123"
    assert captured[0]["id"] is None
    assert captured[0]["resume"] is None

    _monitor(tmp_path).start_run(run_config={}, resumed_from="checkpoints/step-3")
    assert captured[1]["id"] == "run-123"
    assert captured[1]["resume"] == "allow"


def test_wandb_monitor_joins_shared_online_run(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    settings: dict[str, object] = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return _FakeRun()

    def fake_settings(**kwargs):
        settings.update(kwargs)
        return kwargs

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_SHARED_MODE", "1")
    monkeypatch.setenv("WANDB_RUN_ID", "shared-123")
    monkeypatch.setenv("WANDB_SHARED_LABEL", "trainer")
    monkeypatch.setenv("WANDB_SHARED_PRIMARY", "trainer")
    monkeypatch.setenv("WANDB_SHARED_FINISHER", "orchestrator")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(init=fake_init, Settings=fake_settings),
    )
    monitor = RunMonitor(
        tmp_path,
        log_cuda_memory=False,
        log_disk_usage=False,
        wandb=WandbConfig(enabled=True, project="wavelet-tests", mode="online"),
    )

    monitor.start_run(run_config={})

    assert captured["id"] == "shared-123"
    assert captured["resume"] == "allow"
    assert "mode" not in captured
    assert settings["mode"] == "shared"
    assert settings["x_label"] == "trainer"
    assert settings["x_primary"] is True
    assert settings["x_update_finish_state"] is False
    assert (tmp_path / "wandb_run_id.txt").read_text() == "shared-123"


# ── unknown keys and device placement ─────────────────────────────────────────


def test_unknown_config_keys_are_rejected_everywhere() -> None:
    with pytest.raises(ValueError, match="rollouts_per_examples"):
        RLConfig(orchestrator={"rollouts_per_examples": 8})
    with pytest.raises(ValueError, match="max_step"):
        RLConfig(max_step=5)
    with pytest.raises(ValueError, match="learning_rate"):
        RLConfig(optim={"learning_rate": 1e-5})


def test_process_mode_rejects_overlapping_device_groups() -> None:
    from wavelet.orchestrator.placement import device_group_conflict_error

    overlapping = RLConfig(
        launcher={
            "mode": "process",
            "trainer_cuda_visible_devices": "0,1",
            "inference_cuda_visible_devices": "1,2",
        }
    )
    assert "CUDA device(s) 1" in (device_group_conflict_error(overlapping) or "")

    shared_replicas = RLConfig(
        launcher={
            "mode": "process",
            "inference_num_replicas": 2,
            "trainer_cuda_visible_devices": "0",
            "inference_cuda_visible_devices": "1,2",
        }
    )
    assert "replicas 0 and 1" in (device_group_conflict_error(shared_replicas) or "")

    disjoint = RLConfig(
        launcher={
            "mode": "process",
            "inference_num_replicas": 2,
            "trainer_cuda_visible_devices": "0",
            "inference_cuda_visible_devices": "1;2",
        }
    )
    assert device_group_conflict_error(disjoint) is None

    colocated = RLConfig(
        launcher={
            "mode": "colocate",
            "trainer_cuda_visible_devices": "0",
            "inference_cuda_visible_devices": "0",
        }
    )
    assert device_group_conflict_error(colocated) is None

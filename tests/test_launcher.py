from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.launcher import (
    LocalRoleHandle,
    LocalRoleLauncher,
    RayRoleLauncher,
    RoleSpec,
    _role_env,
)


class _FakeProcess:
    pid = 1234

    def __init__(self, *, timeout_once: bool = False) -> None:
        self.timeout_once = timeout_once
        self.sent_signals: list[int] = []
        self.wait_calls = 0

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.timeout_once and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="role", timeout=1.0)
        return 0

    def send_signal(self, signal_number: int) -> None:
        self.sent_signals.append(signal_number)


def test_local_role_handle_terminates_process_group(tmp_path, monkeypatch) -> None:
    process = _FakeProcess(timeout_once=True)
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("wavelet.orchestrator.launcher.os.getpgid", lambda pid: 4321)
    monkeypatch.setattr(
        "wavelet.orchestrator.launcher.os.killpg",
        lambda pgid, signal_number: killpg_calls.append((pgid, signal_number)),
    )

    with (tmp_path / "role.log").open("w", encoding="utf-8") as log_file:
        handle = LocalRoleHandle(
            RoleSpec("trainer", "rl-trainer", tmp_path / "config.yaml", "trainer"),
            process,  # type: ignore[arg-type]
            log_file,
        )
        handle.terminate(timeout_seconds=1.0)

    assert killpg_calls == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert process.sent_signals == []
    assert process.wait_calls == 2


def test_local_role_handle_falls_back_to_child_signal(tmp_path, monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr("wavelet.orchestrator.launcher.os.getpgid", lambda pid: 4321)

    def raise_os_error(pgid: int, signal_number: int) -> None:
        del pgid, signal_number
        raise OSError("no process group")

    monkeypatch.setattr("wavelet.orchestrator.launcher.os.killpg", raise_os_error)

    with (tmp_path / "role.log").open("w", encoding="utf-8") as log_file:
        handle = LocalRoleHandle(
            RoleSpec("trainer", "rl-trainer", tmp_path / "config.yaml", "trainer"),
            process,  # type: ignore[arg-type]
            log_file,
        )
        handle.terminate(timeout_seconds=1.0)

    assert process.sent_signals == [signal.SIGTERM]
    assert process.wait_calls == 1


def test_local_role_launcher_starts_roles_in_new_session(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr("wavelet.orchestrator.launcher.subprocess.Popen", fake_popen)

    launcher = LocalRoleLauncher(tmp_path)
    handle = launcher.start(
        RoleSpec(
            name="inference",
            command="rl-inference",
            config_path=Path("config.yaml"),
            log_name="rl_inference",
        )
    )
    handle.close()

    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"].name.endswith("rl_inference.log")
    assert captured["kwargs"]["stdout"].mode == "a"


def test_local_role_launcher_applies_role_environment(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setenv("WAVELET_PARENT_VALUE", "parent")
    monkeypatch.setattr("wavelet.orchestrator.launcher.subprocess.Popen", fake_popen)

    handle = LocalRoleLauncher(tmp_path).start(
        RoleSpec(
            name="trainer",
            command="rl-trainer",
            config_path=Path("config.yaml"),
            log_name="trainer",
            cuda_visible_devices="2,3",
            env_vars={"WAVELET_ROLE_VALUE": "trainer"},
        )
    )
    handle.close()

    assert captured["env"]["WAVELET_PARENT_VALUE"] == "parent"
    assert captured["env"]["WAVELET_ROLE_VALUE"] == "trainer"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_role_environment_rejects_launcher_managed_values() -> None:
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        _role_env(None, {"CUDA_VISIBLE_DEVICES": "7"})


def test_local_role_launcher_preserves_existing_log(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "rl_inference.log"
    log_path.write_text("previous run\n", encoding="utf-8")
    monkeypatch.setattr(
        "wavelet.orchestrator.launcher.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    launcher = LocalRoleLauncher(tmp_path)
    handle = launcher.start(
        RoleSpec(
            name="inference",
            command="rl-inference",
            config_path=Path("config.yaml"),
            log_name="rl_inference",
        )
    )
    handle.close()

    assert log_path.read_text(encoding="utf-8") == "previous run\n"


def test_local_role_launcher_writes_to_explicit_attempt_log_dir(
    tmp_path, monkeypatch
) -> None:
    attempt_log_dir = tmp_path / "logs" / "attempt_2"
    monkeypatch.setattr(
        "wavelet.orchestrator.launcher.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    handle = LocalRoleLauncher(tmp_path, log_dir=attempt_log_dir).start(
        RoleSpec(
            name="inference",
            command="rl-inference",
            config_path=Path("config.yaml"),
            log_name="rl_inference",
        )
    )
    handle.close()

    assert handle.log_path == attempt_log_dir / "rl_inference.log"


def test_ray_role_launcher_disconnects_on_close(monkeypatch) -> None:
    calls: list[str] = []

    class FakeRay:
        @staticmethod
        def remote(function):
            return function

        @staticmethod
        def init(**kwargs) -> None:
            del kwargs
            calls.append("init")

        @staticmethod
        def shutdown() -> None:
            calls.append("shutdown")

    monkeypatch.setitem(sys.modules, "ray", FakeRay)

    launcher = RayRoleLauncher(RLConfig(launcher={"backend": "ray"}))
    launcher.close()

    assert calls == ["init", "shutdown"]

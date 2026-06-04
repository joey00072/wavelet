from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

from wavelet.orchestrator.launcher import LocalRoleHandle, LocalRoleLauncher, RoleSpec


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

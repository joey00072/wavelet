from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Any, TextIO

from wavelet.configs.config import validate_role_env_vars
from wavelet.configs.rl_config import RLConfig

_TERMINATE_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class RoleSpec:
    name: str
    command: str
    config_path: Path
    log_name: str
    cuda_visible_devices: str | None = None
    service: bool = False
    torchrun_nproc_per_node: int = 1
    env_vars: dict[str, str] = field(default_factory=dict)


def _role_env(
    cuda_visible_devices: str | None,
    env_vars: dict[str, str] | None = None,
) -> dict[str, str]:
    env_vars = {} if env_vars is None else env_vars
    validate_role_env_vars(env_vars, role="role")
    env = os.environ.copy()
    env.update(env_vars)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _log_path(output_dir: Path, log_name: str) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{log_name}.log"


def _role_command(
    command: str,
    config_path: str | Path,
    *,
    torchrun_nproc_per_node: int = 1,
) -> list[str]:
    if torchrun_nproc_per_node <= 1:
        return [
            sys.executable,
            "-m",
            "wavelet",
            command,
            "@",
            str(config_path),
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(torchrun_nproc_per_node),
        "-m",
        "wavelet",
        command,
        "@",
        str(config_path),
    ]


def _run_role_subprocess(
    *,
    command: str,
    config_path: str,
    cwd: str,
    log_path: str,
    cuda_visible_devices: str | None,
    torchrun_nproc_per_node: int = 1,
    env_vars: dict[str, str] | None = None,
) -> int:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as log_file:
        command_args = _role_command(
            command,
            config_path,
            torchrun_nproc_per_node=torchrun_nproc_per_node,
        )
        process = subprocess.Popen(
            command_args,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=_role_env(cuda_visible_devices, env_vars),
            start_new_session=True,
        )
        return _wait_for_role_process(process)


def _wait_for_role_process(
    process: subprocess.Popen,
    *,
    timeout_seconds: float = _TERMINATE_TIMEOUT_SECONDS,
) -> int:
    """Wait for a role subprocess, tearing down its process group on cancel.

    The child runs in its own session, so a cancelled Ray task (which surfaces
    as ``KeyboardInterrupt`` in the worker) or ``SystemExit`` must forward the
    shutdown explicitly; otherwise the role keeps running as an orphan.
    """
    try:
        return int(process.wait())
    except (KeyboardInterrupt, SystemExit):
        _terminate_process_group(process, timeout_seconds=timeout_seconds)
        raise


def _terminate_process_group(
    process: subprocess.Popen,
    *,
    timeout_seconds: float = _TERMINATE_TIMEOUT_SECONDS,
) -> None:
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait()


def _start_local_role(
    spec: RoleSpec,
    *,
    output_dir: Path,
) -> tuple[subprocess.Popen, TextIO]:
    log_file = _log_path(output_dir, spec.log_name).open("a", encoding="utf-8")
    process = subprocess.Popen(
        _role_command(
            spec.command,
            spec.config_path,
            torchrun_nproc_per_node=spec.torchrun_nproc_per_node,
        ),
        cwd=Path.cwd(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=_role_env(spec.cuda_visible_devices, spec.env_vars),
        start_new_session=True,
    )
    return process, log_file


class LocalRoleHandle:
    def __init__(self, spec: RoleSpec, process: subprocess.Popen, log_file: TextIO):
        self.spec = spec
        self.process = process
        self.log_file = log_file
        self.log_path = Path(log_file.name)

    def poll(self) -> int | None:
        code = self.process.poll()
        return None if code is None else int(code)

    def terminate(self, *, timeout_seconds: float = _TERMINATE_TIMEOUT_SECONDS) -> None:
        _terminate_process_group(self.process, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        self.log_file.close()


class RayRoleHandle:
    def __init__(self, spec: RoleSpec, ref: object, ray_module: Any, log_path: Path):
        self.spec = spec
        self.ref = ref
        self.ray = ray_module
        self.log_path = log_path

    def poll(self) -> int | None:
        ready, _pending = self.ray.wait([self.ref], timeout=0)
        if not ready:
            return None
        return int(self.ray.get(self.ref))

    def terminate(self, *, timeout_seconds: float = _TERMINATE_TIMEOUT_SECONDS) -> None:
        # A non-forced cancel interrupts the worker task so it can tear down the
        # role's process group; force-killing the worker would orphan the child.
        self.ray.cancel(self.ref, force=False)
        ready, _pending = self.ray.wait([self.ref], timeout=timeout_seconds)
        if not ready:
            self.ray.cancel(self.ref, force=True)

    def close(self) -> None:
        return None


class LocalRoleLauncher:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def start(self, spec: RoleSpec) -> LocalRoleHandle:
        process, log_file = _start_local_role(spec, output_dir=self.output_dir)
        return LocalRoleHandle(spec, process, log_file)

    def close(self) -> None:
        return None


class RayRoleLauncher:
    def __init__(self, config: RLConfig) -> None:
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "launcher.backend='ray' requires Ray. Install ray or use "
                "launcher.backend='local'."
            ) from exc
        self.config = config
        self.ray = ray
        self.remote_runner = ray.remote(_run_role_subprocess)
        ray.init(
            address=config.launcher.ray_address,
            runtime_env=config.launcher.ray_runtime_env,
            ignore_reinit_error=True,
        )

    def start(self, spec: RoleSpec) -> RayRoleHandle:
        log_path = _log_path(self.config.output_dir, spec.log_name)
        ref = self.remote_runner.remote(
            command=spec.command,
            config_path=str(spec.config_path),
            cwd=str(Path.cwd()),
            log_path=str(log_path),
            cuda_visible_devices=spec.cuda_visible_devices,
            torchrun_nproc_per_node=spec.torchrun_nproc_per_node,
            env_vars=spec.env_vars,
        )
        return RayRoleHandle(spec, ref, self.ray, log_path)

    def close(self) -> None:
        self.ray.shutdown()


RoleHandle = LocalRoleHandle | RayRoleHandle


def create_role_launcher(config: RLConfig) -> LocalRoleLauncher | RayRoleLauncher:
    if config.launcher.backend == "local":
        return LocalRoleLauncher(config.output_dir)
    if config.launcher.backend == "ray":
        return RayRoleLauncher(config)
    raise ValueError(f"Unsupported launcher backend: {config.launcher.backend}")


def _signal_process_group(process: subprocess.Popen, signal_number: int) -> None:
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal_number)
    except ProcessLookupError:
        return
    except OSError:
        process.send_signal(signal_number)


def terminate_remaining(handles: list[RoleHandle]) -> None:
    for handle in handles:
        if handle.poll() is None:
            handle.terminate()


def close_handles(handles: list[RoleHandle]) -> None:
    for handle in handles:
        handle.close()


def wait_for_roles(
    handles: list[RoleHandle],
    *,
    poll_interval_seconds: float,
) -> None:
    services = {handle.spec.name for handle in handles if handle.spec.service}
    remaining = {handle.spec.name: handle for handle in handles}
    jobs = set(remaining) - services

    while jobs:
        for name, handle in list(remaining.items()):
            code = handle.poll()
            if code is None:
                continue
            del remaining[name]
            if name in jobs and code == 0:
                jobs.remove(name)
                continue
            if name in services and jobs:
                raise RuntimeError(
                    f"RL {name} service exited early with code {code}. Check "
                    f"'{handle.log_path}'."
                )
            terminate_remaining(list(remaining.values()))
            raise RuntimeError(
                f"RL {name} role exited with code {code}. Check '{handle.log_path}'."
            )
        if jobs:
            sleep(poll_interval_seconds)

    for name in services & set(remaining):
        remaining[name].terminate()

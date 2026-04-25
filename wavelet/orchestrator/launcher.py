from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, TextIO

from wavelet.configs.rl_config import RLConfig


@dataclass(frozen=True, slots=True)
class RoleSpec:
    name: str
    command: str
    config_path: Path
    log_name: str
    cuda_visible_devices: str | None = None
    service: bool = False


def _role_env(cuda_visible_devices: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _log_path(output_dir: Path, log_name: str) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{log_name}.log"


def _run_role_subprocess(
    *,
    command: str,
    config_path: str,
    cwd: str,
    log_path: str,
    cuda_visible_devices: str | None,
) -> int:
    with Path(log_path).open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "wavelet",
                command,
                "@",
                config_path,
            ],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=_role_env(cuda_visible_devices),
        )
        return int(process.wait())


class LocalRoleHandle:
    def __init__(self, spec: RoleSpec, process: subprocess.Popen, log_file: TextIO):
        self.spec = spec
        self.process = process
        self.log_file = log_file
        self.log_path = Path(log_file.name)

    def poll(self) -> int | None:
        code = self.process.poll()
        return None if code is None else int(code)

    def terminate(self, *, timeout_seconds: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

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

    def terminate(self, *, timeout_seconds: float = 10.0) -> None:
        del timeout_seconds
        self.ray.cancel(self.ref, force=True)

    def close(self) -> None:
        return None


class LocalRoleLauncher:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def start(self, spec: RoleSpec) -> LocalRoleHandle:
        log_path = _log_path(self.output_dir, spec.log_name)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "wavelet",
                spec.command,
                "@",
                str(spec.config_path),
            ],
            cwd=Path.cwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=_role_env(spec.cuda_visible_devices),
        )
        return LocalRoleHandle(spec, process, log_file)


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
        )
        return RayRoleHandle(spec, ref, self.ray, log_path)


RoleHandle = LocalRoleHandle | RayRoleHandle


def create_role_launcher(config: RLConfig) -> LocalRoleLauncher | RayRoleLauncher:
    if config.launcher.backend == "local":
        return LocalRoleLauncher(config.output_dir)
    if config.launcher.backend == "ray":
        return RayRoleLauncher(config)
    raise ValueError(f"Unsupported launcher backend: {config.launcher.backend}")


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

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from wavelet.configs.config import DeploymentConfig, RLConfig, SFTConfig, TrainerConfig
from wavelet.orchestrator.placement import required_inference_devices
from wavelet.utils.config import load_config
from wavelet.utils.pathing import get_config_dir, launch_config_paths
from wavelet.utils.serialization import dump_yaml

DeploymentCommand = Literal["rl", "sft"]


def _directive(name: str, value: str | int | None) -> str | None:
    if value is None:
        return None
    return f"#SBATCH --{name}={value}"


def total_nodes(config: TrainerConfig, command: DeploymentCommand) -> int:
    nodes = config.deployment.num_train_nodes
    if command == "rl":
        nodes += config.deployment.num_inference_nodes
    return nodes


def render_sbatch_script(
    config: TrainerConfig,
    *,
    command: DeploymentCommand,
    config_path: Path,
    log_path: Path,
) -> str:
    """Render a site-neutral sbatch script for one resolved Wavelet config."""
    slurm = config.slurm
    if slurm is None:
        raise ValueError("Cannot render an sbatch script without a slurm block.")
    project_dir = slurm.project_dir.resolve()
    directives = [
        "#SBATCH --job-name=" + slurm.job_name,
        f"#SBATCH --nodes={total_nodes(config, command)}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-node={config.deployment.gpus_per_node}",
        f"#SBATCH --output={log_path.resolve()}",
        f"#SBATCH --error={log_path.resolve()}",
        _directive("partition", slurm.partition),
        _directive("account", slurm.account),
        _directive("qos", slurm.qos),
        _directive("time", slurm.time_limit),
        _directive("constraint", slurm.constraint),
        _directive("reservation", slurm.reservation),
        _directive("nodelist", slurm.nodelist),
        _directive("exclude", slurm.exclude),
        _directive("cpus-per-task", slurm.cpus_per_task),
        _directive("mem", slurm.memory),
        "#SBATCH --exclusive" if slurm.exclusive else None,
        *(f"#SBATCH {value}" for value in slurm.extra_directives),
    ]
    lines = [
        "#!/usr/bin/env bash",
        *(value for value in directives if value is not None),
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_dir))}",
        *slurm.setup_commands,
        "exec "
        + slurm.python_command
        + " -m wavelet slurm-worker "
        + command
        + " @ "
        + shlex.quote(str(config_path.resolve())),
        "",
    ]
    return "\n".join(lines)


def write_sbatch_script(
    config: TrainerConfig,
    *,
    command: DeploymentCommand,
    config_path: Path,
    script_path: Path,
    log_path: Path,
) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        render_sbatch_script(
            config,
            command=command,
            config_path=config_path,
            log_path=log_path,
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o750)
    return script_path


def submit_sbatch(script_path: Path) -> str:
    if shutil.which("sbatch") is None:
        raise FileNotFoundError(
            "SLURM submission requested but 'sbatch' is not available on PATH."
        )
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id:
        raise RuntimeError("sbatch returned an empty job id.")
    return job_id


def launch_slurm(
    config: TrainerConfig,
    *,
    command: DeploymentCommand,
    config_path: Path,
    script_path: Path,
    log_path: Path,
) -> int:
    script = write_sbatch_script(
        config,
        command=command,
        config_path=config_path,
        script_path=script_path,
        log_path=log_path,
    )
    if config.dry_run:
        print(f"Dry run - wrote SLURM job script to {script}")
        return 0
    job_id = submit_sbatch(script)
    job_id_path = script.parent / "slurm_job_id.txt"
    job_id_path.write_text(f"{job_id}\n", encoding="utf-8")
    print(f"Submitted SLURM job {job_id}")
    print(f"Job script: {script}")
    return 0


def _allocated_hosts() -> list[str]:
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not node_list:
        raise RuntimeError("slurm-worker must run inside an sbatch allocation.")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        check=True,
        capture_output=True,
        text=True,
    )
    hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not hosts:
        raise RuntimeError("SLURM allocation did not resolve to any hosts.")
    return hosts


def _worker_config(config: TrainerConfig) -> TrainerConfig:
    return config.model_copy(
        update={
            "deployment": DeploymentConfig(),
            "slurm": None,
            "clean_output_dir": False,
        }
    )


def _srun_prefix(hosts: list[str], *, gpus_per_task: int) -> list[str]:
    return [
        "srun",
        "--nodes",
        str(len(hosts)),
        "--ntasks",
        str(len(hosts)),
        "--ntasks-per-node",
        "1",
        "--nodelist",
        ",".join(hosts),
        "--gpus-per-task",
        str(gpus_per_task),
        "--kill-on-bad-exit=1",
        "--exclusive",
    ]


def _torchrun_command(
    config: TrainerConfig,
    *,
    hosts: list[str],
    command: Literal["rl-trainer", "sft"],
    config_path: Path,
) -> list[str]:
    job_id = os.environ.get("SLURM_JOB_ID", "wavelet")
    return [
        *_srun_prefix(hosts, gpus_per_task=config.deployment.gpus_per_node),
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(len(hosts)),
        "--nproc-per-node",
        str(config.deployment.gpus_per_node),
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"{hosts[0]}:{config.deployment.trainer_master_port}",
        "--rdzv-id",
        job_id,
        "-m",
        "wavelet",
        command,
        "@",
        str(config_path),
    ]


@dataclass(slots=True)
class _ManagedProcess:
    name: str
    process: subprocess.Popen
    log_file: TextIO
    service: bool = False

    def close(self) -> None:
        self.log_file.close()


def _start_process(
    name: str,
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str] | None = None,
    service: bool = False,
) -> _ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return _ManagedProcess(name, process, log_file, service)


def _terminate(process: subprocess.Popen, *, timeout_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def _raise_keyboard_interrupt(_signal_number, _frame) -> None:
    raise KeyboardInterrupt


def _run_to_completion(
    name: str,
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
) -> int:
    managed = _start_process(
        name,
        command,
        log_path=log_path,
        cwd=cwd,
        env=env,
    )
    try:
        return int(managed.process.wait())
    finally:
        try:
            _terminate(managed.process)
        finally:
            managed.close()


def _wait_for_http_servers(
    processes: list[_ManagedProcess],
    endpoints: list[str],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = set(endpoints)
    last_error: Exception | None = None
    while pending and time.monotonic() < deadline:
        for managed in processes:
            if (code := managed.process.poll()) is not None:
                raise RuntimeError(
                    f"RL {managed.name} exited during startup with code {code}. "
                    f"Check '{managed.log_file.name}'."
                )
        for endpoint in list(pending):
            try:
                with urllib.request.urlopen(
                    f"{endpoint}/health", timeout=5.0
                ) as response:
                    if response.status == 200:
                        pending.remove(endpoint)
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
        if pending:
            time.sleep(1.0)
    if pending:
        raise TimeoutError(
            "Timed out waiting for inference servers: " + ", ".join(sorted(pending))
        ) from last_error


def _wait_for_jobs(processes: list[_ManagedProcess], *, poll_seconds: float) -> None:
    jobs = {process.name for process in processes if not process.service}
    remaining = {process.name: process for process in processes}
    while jobs:
        for name, managed in list(remaining.items()):
            code = managed.process.poll()
            if code is None:
                continue
            del remaining[name]
            if name in jobs and code == 0:
                jobs.remove(name)
                continue
            if managed.service and jobs:
                reason = "service exited early"
            else:
                reason = "role failed"
            raise RuntimeError(
                f"RL {managed.name} {reason} with code {code}. "
                f"Check '{managed.log_file.name}'."
            )
        if jobs:
            time.sleep(poll_seconds)


def _record_allocation(
    config: TrainerConfig,
    *,
    hosts: list[str],
    command: DeploymentCommand,
    config_dir: Path,
) -> None:
    payload = {
        "command": command,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hosts": hosts,
        "train_hosts": hosts[-config.deployment.num_train_nodes :],
        "inference_hosts": (
            hosts[: config.deployment.num_inference_nodes] if command == "rl" else []
        ),
        "gpus_per_node": config.deployment.gpus_per_node,
    }
    path = config_dir / "slurm_allocation.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_sft_worker(
    config: SFTConfig,
    *,
    hosts: list[str],
    config_dir: Path | None = None,
) -> int:
    worker_config = _worker_config(config)
    assert isinstance(worker_config, SFTConfig)
    config_dir = get_config_dir(config.output_dir) if config_dir is None else config_dir
    config_path = config_dir / "sft_worker.yaml"
    dump_yaml(config_path, worker_config.model_dump(mode="json", exclude_none=True))
    command = _torchrun_command(
        config,
        hosts=hosts,
        command="sft",
        config_path=config_path,
    )
    env = os.environ.copy()
    env["WAVELET_SLURM_OUTPUT_PREPARED"] = "1"
    return _run_to_completion(
        "sft_trainer",
        command,
        log_path=config.output_dir / "logs" / config_dir.parent.name / "sft.log",
        cwd=config.slurm.project_dir.resolve(),
        env=env,
    )


def _role_environment(config: RLConfig, role: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(config.launcher.env_vars.for_role(role))
    return env


def run_rl_worker(
    config: RLConfig,
    *,
    hosts: list[str],
    config_dir: Path | None = None,
) -> int:
    from wavelet.orchestrator.runtime import (
        _config_path_for_role,
        _inference_replica_config,
        _rollout_client_config,
        _write_subconfigs,
    )

    inference_count = config.deployment.num_inference_nodes
    inference_hosts = hosts[:inference_count]
    trainer_hosts = hosts[inference_count:]
    if len(trainer_hosts) != config.deployment.num_train_nodes:
        raise RuntimeError("SLURM allocation does not contain the requested trainers.")
    required_gpus = required_inference_devices(config)
    if (
        config.inference.mode == "vllm_http"
        and required_gpus > config.deployment.gpus_per_node
    ):
        raise ValueError(
            "Each inference replica requires "
            f"{required_gpus} GPU(s), but deployment.gpus_per_node="
            f"{config.deployment.gpus_per_node}."
        )

    worker_config = _worker_config(config)
    assert isinstance(worker_config, RLConfig)
    config_dir = get_config_dir(config.output_dir) if config_dir is None else config_dir
    project_dir = config.slurm.project_dir.resolve()
    if not config.orchestrator.enabled:
        trainer_path = config_dir / "rl_trainer.yaml"
        dump_yaml(
            trainer_path,
            worker_config.model_dump(mode="json", exclude_none=False),
        )
        return _run_to_completion(
            "trainer",
            _torchrun_command(
                config,
                hosts=trainer_hosts,
                command="rl-trainer",
                config_path=trainer_path,
            ),
            log_path=(
                config.output_dir / "logs" / config_dir.parent.name / "rl_trainer.log"
            ),
            cwd=project_dir,
            env=_role_environment(config, "trainer"),
        )

    worker_config = worker_config.model_copy(
        update={
            "launcher": worker_config.launcher.model_copy(
                update={"inference_num_replicas": max(1, inference_count)}
            )
        }
    )
    ports = [config.inference.http.port + index for index in range(inference_count)]
    rollout_config = (
        _rollout_client_config(
            worker_config,
            ports=ports,
            hosts=inference_hosts,
        )
        if config.inference.mode == "vllm_http"
        else worker_config
    )
    _write_subconfigs(
        rollout_config,
        worker_config,
        config_dir=config_dir,
    )
    log_dir = config.output_dir / "logs" / config_dir.parent.name
    processes: list[_ManagedProcess] = []
    try:
        server_command = (
            "inference-server"
            if config.inference.vllm.server_backend == "openai"
            else "native-inference-server"
        )
        for index, (host, port) in enumerate(zip(inference_hosts, ports, strict=True)):
            replica_config = _inference_replica_config(worker_config, port=port)
            replica_config = replica_config.model_copy(
                update={
                    "inference": replica_config.inference.model_copy(
                        update={
                            "http": replica_config.inference.http.model_copy(
                                update={"host": "0.0.0.0", "hosts": None, "ports": None}
                            )
                        }
                    )
                }
            )
            replica_path = _config_path_for_role(
                worker_config,
                f"inference_server_{index}",
                replica_config,
                config_dir=config_dir,
            )
            processes.append(
                _start_process(
                    f"inference_server_{index}",
                    [
                        *_srun_prefix(
                            [host],
                            gpus_per_task=config.deployment.gpus_per_node,
                        ),
                        sys.executable,
                        "-m",
                        "wavelet",
                        server_command,
                        "@",
                        str(replica_path),
                    ],
                    log_path=log_dir / f"inference_server_{index}.log",
                    cwd=project_dir,
                    env=_role_environment(config, "inference"),
                    service=True,
                )
            )
        endpoints = [
            f"http://{host}:{port}"
            for host, port in zip(inference_hosts, ports, strict=True)
        ]
        _wait_for_http_servers(
            processes,
            endpoints,
            timeout_seconds=config.inference.http.startup_timeout_seconds,
        )

        if config.max_steps != 0:
            trainer_path = config_dir / "rl_trainer.yaml"
            processes.append(
                _start_process(
                    "trainer",
                    _torchrun_command(
                        config,
                        hosts=trainer_hosts,
                        command="rl-trainer",
                        config_path=trainer_path,
                    ),
                    log_path=log_dir / "rl_trainer.log",
                    cwd=project_dir,
                    env=_role_environment(config, "trainer"),
                )
            )
        orchestrator_env = _role_environment(config, "orchestrator")
        orchestrator_env["CUDA_VISIBLE_DEVICES"] = ""
        processes.append(
            _start_process(
                "inference",
                [
                    sys.executable,
                    "-m",
                    "wavelet",
                    "rl-inference",
                    "@",
                    str(config_dir / "rl_inference.yaml"),
                ],
                log_path=log_dir / "rl_inference.log",
                cwd=project_dir,
                env=orchestrator_env,
            )
        )
        _wait_for_jobs(
            processes,
            poll_seconds=config.launcher.poll_interval_seconds,
        )
    finally:
        try:
            for managed in processes:
                _terminate(managed.process)
        finally:
            for managed in processes:
                managed.close()
    return 0


def worker_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in {"rl", "sft"}:
        print("Usage: wavelet slurm-worker {rl|sft} @ <config.yaml>")
        return 1
    command = argv[0]
    config_type = RLConfig if command == "rl" else SFTConfig
    config = load_config(config_type, argv[1:])
    if config.slurm is None:
        raise ValueError("slurm-worker requires a resolved config with slurm options.")
    hosts = _allocated_hosts()
    expected = total_nodes(config, command)
    if len(hosts) != expected:
        raise RuntimeError(
            f"SLURM allocated {len(hosts)} host(s), but the config requires "
            f"{expected}: {', '.join(hosts)}"
        )
    sources = launch_config_paths(argv[1:])
    if len(sources) != 1:
        raise ValueError("slurm-worker requires exactly one resolved config file.")
    config_dir = sources[0].resolve().parent
    _record_allocation(
        config,
        hosts=hosts,
        command=command,
        config_dir=config_dir,
    )
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        if command == "sft":
            assert isinstance(config, SFTConfig)
            return run_sft_worker(config, hosts=hosts, config_dir=config_dir)
        assert isinstance(config, RLConfig)
        return run_rl_worker(config, hosts=hosts, config_dir=config_dir)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

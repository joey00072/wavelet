from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

STABLE_CHECKPOINT_MARKER = "STABLE"
OUTPUT_RUN_STATE_MARKERS = (
    "configs",
    "eval_metrics.jsonl",
    "evals",
    "events",
    "events.jsonl",
    "heartbeat.json",
    "logs",
    "metrics.csv",
    "metrics.jsonl",
    "policies",
    "rollouts",
    "run_metadata.json",
    "wandb",
)


@dataclass(frozen=True, slots=True)
class LaunchAttemptPaths:
    config_attempt_dir: Path
    config_dir: Path
    log_dir: Path


def _int_suffix(name: str, prefix: str) -> int | None:
    try:
        return int(name.removeprefix(prefix))
    except ValueError:
        return None


def _attempt_numbers(parent: Path) -> set[int]:
    return {
        int(suffix)
        for path in parent.glob("attempt_*")
        if path.is_dir() and (suffix := path.name.removeprefix("attempt_")).isdigit()
    }


def _point_latest(parent: Path, attempt_dir: Path) -> None:
    temporary = parent / f".latest-{os.getpid()}"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(attempt_dir.name, target_is_directory=True)
    os.replace(temporary, parent / "latest")


def create_launch_attempt(output_dir: Path) -> LaunchAttemptPaths:
    """Allocate matching resolved-config and log directories for one launch."""
    configs_root = output_dir / "configs"
    logs_root = output_dir / "logs"
    configs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    attempts = _attempt_numbers(configs_root) | _attempt_numbers(logs_root)
    attempt_name = f"attempt_{max(attempts, default=0) + 1}"
    config_attempt_dir = configs_root / attempt_name
    config_dir = config_attempt_dir / "resolved"
    log_dir = logs_root / attempt_name
    config_dir.mkdir(parents=True)
    log_dir.mkdir()
    _point_latest(configs_root, config_attempt_dir)
    _point_latest(logs_root, log_dir)
    return LaunchAttemptPaths(config_attempt_dir, config_dir, log_dir)


def launch_config_paths(argv: list[str]) -> list[Path]:
    """Return root config files referenced by the supported CLI syntax."""
    paths: list[Path] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "@" and index + 1 < len(argv):
            paths.append(Path(argv[index + 1]))
            index += 2
            continue
        if token.startswith("@") and len(token) > 1:
            paths.append(Path(token[1:]))
        index += 1
    return paths


def write_launch_artifacts(
    attempt: LaunchAttemptPaths,
    *,
    command: str,
    argv: list[str],
) -> None:
    """Record a reproducible command and a copy of its root config."""
    command_line = shlex.join(["uv", "run", "wavelet", command, *argv])
    (attempt.config_attempt_dir / "command.txt").write_text(
        f"{command_line}\n",
        encoding="utf-8",
    )
    sources = launch_config_paths(argv)
    for index, source in enumerate(sources, start=1):
        if not source.is_file():
            continue
        suffix = source.suffix or ".yaml"
        stem = command if len(sources) == 1 else f"{command}_{index}"
        destination = attempt.config_attempt_dir / f"{stem}{suffix}"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)


def get_config_dir(output_dir: Path) -> Path:
    configs_root = output_dir / "configs"
    latest = configs_root / "latest" / "resolved"
    return latest if latest.exists() else configs_root


def get_checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def is_stable_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / STABLE_CHECKPOINT_MARKER).exists()


def list_checkpoint_steps(
    output_dir: Path,
    *,
    stable_only: bool = True,
) -> list[int]:
    return sorted(
        step
        for candidate in output_dir.glob("checkpoint-*")
        if candidate.is_dir()
        and (not stable_only or is_stable_checkpoint(candidate))
        and (step := _int_suffix(candidate.name, "checkpoint-")) is not None
    )


def _require_stable_checkpoint(checkpoint_dir: Path, *, missing: str) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(missing)
    if not is_stable_checkpoint(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_dir.name}' exists but is not stable."
        )
    return checkpoint_dir


def resolve_resume_checkpoint(output_dir: Path, resume_step: int) -> Path:
    if resume_step == -1:
        steps = list_checkpoint_steps(output_dir, stable_only=True)
        if not steps:
            raise FileNotFoundError(
                f"No stable checkpoints found under '{output_dir}'."
            )
        return get_checkpoint_dir(output_dir, steps[-1])
    checkpoint_dir = get_checkpoint_dir(output_dir, resume_step)
    return _require_stable_checkpoint(
        checkpoint_dir,
        missing=(
            f"Checkpoint '{checkpoint_dir.name}' was not found under '{output_dir}'."
        ),
    )


def resolve_resume_checkpoint_source(
    output_dir: Path,
    *,
    resume_step: int | None,
    resume_dir: Path | None,
) -> Path:
    """Resolve a checkpoint from this run's root or an explicit step directory."""
    if resume_dir is None:
        if resume_step is None:
            raise ValueError("Checkpoint resume requires resume_step or resume_dir.")
        return resolve_resume_checkpoint(output_dir, resume_step)
    checkpoint_dir = Path(resume_dir)
    return _require_stable_checkpoint(
        checkpoint_dir,
        missing=f"Checkpoint not found at '{checkpoint_dir}'.",
    )


def existing_run_state_entries(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    entries = [
        marker for marker in OUTPUT_RUN_STATE_MARKERS if (output_dir / marker).exists()
    ]
    entries.extend(path.name for path in sorted(output_dir.glob("checkpoint-*")))
    return entries


def _reject_unsafe_cleanup_root(output_root: Path) -> None:
    """Refuse to recursively delete directories that are clearly not run dirs."""
    cwd = Path.cwd().absolute()
    home = Path.home().absolute()
    reasons: list[str] = []
    if output_root == Path(output_root.anchor):
        reasons.append("the filesystem root")
    if output_root == home:
        reasons.append("the home directory")
    if cwd == output_root or cwd.is_relative_to(output_root):
        reasons.append("the current working directory or one of its parents")
    if reasons:
        raise ValueError(
            f"clean_output_dir=true refuses to delete '{output_root}' because it is "
            + " and ".join(reasons)
            + ". Point output_dir at a dedicated run directory."
        )


def validate_output_dir(
    output_dir: Path,
    *,
    resuming: bool,
    clean: bool,
    protected_paths: Iterable[Path | None] = (),
) -> None:
    if resuming and clean:
        raise ValueError(
            "clean_output_dir=true cannot be used with checkpoint resume. "
            "Choose either a clean run or an explicit resume."
        )
    if resuming:
        return
    if clean:
        output_root = output_dir.absolute()
        _reject_unsafe_cleanup_root(output_root)
        endangered = [
            path
            for path in protected_paths
            if path is not None and path.absolute().is_relative_to(output_root)
        ]
        if endangered:
            raise ValueError(
                "clean_output_dir=true would remove required input path(s): "
                + ", ".join(str(path) for path in endangered)
            )
        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        return
    run_state_entries = existing_run_state_entries(output_dir)
    if run_state_entries:
        preview = ", ".join(run_state_entries[:5])
        if len(run_state_entries) > 5:
            preview += ", ..."
        raise FileExistsError(
            f"Directory '{output_dir}' already contains run state ({preview}). "
            "Use a fresh output_dir, set clean_output_dir=true, or resume from "
            "an existing checkpoint explicitly."
        )

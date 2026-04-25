from __future__ import annotations

import shutil
from pathlib import Path


STABLE_CHECKPOINT_MARKER = "STABLE"


def resolve_output_dir(base_dir: Path, name: str | None = None) -> Path:
    if name is not None:
        return base_dir / name
    return base_dir


def get_config_dir(output_dir: Path) -> Path:
    return output_dir / "configs"


def get_checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def is_stable_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / STABLE_CHECKPOINT_MARKER).exists()


def list_checkpoint_steps(
    output_dir: Path,
    *,
    stable_only: bool = True,
) -> list[int]:
    steps: list[int] = []
    for candidate in output_dir.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        if stable_only and not is_stable_checkpoint(candidate):
            continue
        prefix = "checkpoint-"
        try:
            step = int(candidate.name.removeprefix(prefix))
        except ValueError:
            continue
        steps.append(step)
    return sorted(steps)


def resolve_resume_checkpoint(output_dir: Path, resume_step: int) -> Path:
    if resume_step == -1:
        steps = list_checkpoint_steps(output_dir, stable_only=True)
        if not steps:
            raise FileNotFoundError(
                f"No stable checkpoints found under '{output_dir}'."
            )
        checkpoint_dir = get_checkpoint_dir(output_dir, steps[-1])
    else:
        checkpoint_dir = get_checkpoint_dir(output_dir, resume_step)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' was not found under "
                f"'{output_dir}'."
            )
        if not is_stable_checkpoint(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' exists but is not stable."
            )
    return checkpoint_dir


def has_checkpoints(output_dir: Path) -> bool:
    return any(output_dir.glob("checkpoint-*"))


def validate_output_dir(output_dir: Path, *, resuming: bool, clean: bool) -> None:
    if resuming:
        return
    if clean:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return
    if has_checkpoints(output_dir):
        raise FileExistsError(
            f"Directory '{output_dir}' already contains checkpoints. "
            "Use a fresh output_dir, set clean_output_dir=true, or resume from "
            "an existing checkpoint explicitly."
        )

from __future__ import annotations

import shutil
from collections.abc import Iterable
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


def existing_run_state_entries(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    entries = [
        marker
        for marker in OUTPUT_RUN_STATE_MARKERS
        if (output_dir / marker).exists()
    ]
    entries.extend(path.name for path in sorted(output_dir.glob("checkpoint-*")))
    return entries


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

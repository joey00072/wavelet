from __future__ import annotations

from pathlib import Path

import pytest

from wavelet.utils.pathing import (
    create_launch_attempt,
    existing_run_state_entries,
    get_config_dir,
    validate_output_dir,
    write_launch_artifacts,
)


def test_launch_attempts_preserve_configs_commands_and_logs(tmp_path: Path) -> None:
    source = tmp_path / "source config.yaml"
    source.write_text("max_steps: 3\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    first = create_launch_attempt(output_dir)
    write_launch_artifacts(first, command="rl", argv=["@", str(source)])
    (first.config_dir / "rl_orchestrator.yaml").write_text(
        "max_steps: 3\n", encoding="utf-8"
    )
    (first.log_dir / "rl_inference.log").write_text("first\n", encoding="utf-8")
    second = create_launch_attempt(output_dir)

    assert first.config_attempt_dir.name == "attempt_1"
    assert second.config_attempt_dir.name == "attempt_2"
    assert (first.config_attempt_dir / "rl.yaml").read_text() == "max_steps: 3\n"
    assert (
        "source config.yaml" in (first.config_attempt_dir / "command.txt").read_text()
    )
    assert (first.log_dir / "rl_inference.log").read_text() == "first\n"
    assert (output_dir / "configs" / "latest").resolve() == (
        second.config_attempt_dir.resolve()
    )
    assert (output_dir / "logs" / "latest").resolve() == second.log_dir.resolve()
    assert get_config_dir(output_dir).resolve() == second.config_dir.resolve()


def test_launch_artifacts_copy_each_composed_config(tmp_path: Path) -> None:
    first_source = tmp_path / "base.yaml"
    second_source = tmp_path / "overlay.yaml"
    first_source.write_text("max_steps: 3\n", encoding="utf-8")
    second_source.write_text("launcher:\n  mode: process\n", encoding="utf-8")
    attempt = create_launch_attempt(tmp_path / "run")

    write_launch_artifacts(
        attempt,
        command="rl",
        argv=["@", str(first_source), f"@{second_source}"],
    )

    assert (attempt.config_attempt_dir / "rl_1.yaml").read_text() == ("max_steps: 3\n")
    assert (attempt.config_attempt_dir / "rl_2.yaml").read_text() == (
        "launcher:\n  mode: process\n"
    )


def test_validate_output_dir_allows_empty_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    validate_output_dir(output_dir, resuming=False, clean=False)

    assert output_dir.exists()


def test_validate_output_dir_rejects_previous_policy_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "policies" / "step-000000").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already contains run state"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_validate_output_dir_rejects_previous_rollout_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    rollout_dir = output_dir / "rollouts" / "step-000000"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "STABLE").touch()

    with pytest.raises(FileExistsError, match="rollouts"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_validate_output_dir_allows_explicit_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "policies" / "step-000000").mkdir(parents=True)

    validate_output_dir(output_dir, resuming=True, clean=False)

    assert "policies" in existing_run_state_entries(output_dir)


def test_validate_output_dir_rejects_clean_resume_mix(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="cannot be used with checkpoint resume"):
        validate_output_dir(output_dir, resuming=True, clean=True)


def test_validate_output_dir_clean_removes_previous_run_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "policies" / "step-000000").mkdir(parents=True)

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_validate_output_dir_does_not_delete_protected_input(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    adapter_path = output_dir / "policies" / "step-1" / "adapter"
    adapter_path.mkdir(parents=True)
    marker = adapter_path / "adapter_model.safetensors"
    marker.write_text("weights")

    with pytest.raises(ValueError, match="would remove required input"):
        validate_output_dir(
            output_dir,
            resuming=False,
            clean=True,
            protected_paths=(adapter_path,),
        )

    assert marker.read_text() == "weights"

from __future__ import annotations

from pathlib import Path

import pytest

from wavelet.utils.pathing import existing_run_state_entries, validate_output_dir


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

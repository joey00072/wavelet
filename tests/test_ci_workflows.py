from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(".github/workflows")


@pytest.mark.parametrize("name", ["style.yaml", "cpu-tests.yaml"])
def test_ci_workflow_is_valid_yaml_with_expected_triggers(name: str) -> None:
    workflow = yaml.load(
        (WORKFLOW_DIR / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    assert workflow["name"]
    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["jobs"]


def test_cpu_workflow_splits_unit_and_integration_tests() -> None:
    text = (WORKFLOW_DIR / "cpu-tests.yaml").read_text(encoding="utf-8")

    assert "uv sync --extra dev --locked" in text
    assert 'uv run pytest tests -m "not gpu and not integration"' in text
    assert 'uv run pytest tests -m "integration and not gpu"' in text


def test_gpu_workflow_is_manual_and_targets_self_hosted_gpu_runner() -> None:
    workflow = yaml.load(
        (WORKFLOW_DIR / "gpu-tests.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    text = (WORKFLOW_DIR / "gpu-tests.yaml").read_text(encoding="utf-8")

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["jobs"]["pytest"]["runs-on"] == ["self-hosted", "linux", "gpu"]
    assert "uv sync --extra dev --locked" in text
    assert "uv run pytest tests -m gpu" in text

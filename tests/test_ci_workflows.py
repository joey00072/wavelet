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


def test_cpu_workflow_uses_lockfile_and_excludes_gpu_tests() -> None:
    text = (WORKFLOW_DIR / "cpu-tests.yaml").read_text(encoding="utf-8")

    assert "uv sync --extra dev --locked" in text
    assert 'uv run pytest tests -m "not gpu"' in text

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


def test_gpu_workflow_is_scheduled_and_targets_self_hosted_gpu_runner() -> None:
    workflow = yaml.load(
        (WORKFLOW_DIR / "gpu-tests.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    text = (WORKFLOW_DIR / "gpu-tests.yaml").read_text(encoding="utf-8")

    assert set(workflow["on"]) == {"workflow_dispatch", "schedule"}
    assert workflow["jobs"]["pytest"]["runs-on"] == ["self-hosted", "linux", "gpu"]
    assert "uv sync --extra dev --locked" in text
    assert "uv run pytest tests -m gpu" in text


def test_benchmark_workflow_uses_valid_config_and_retains_diagnostics() -> None:
    from wavelet.configs.config import SFTConfig

    workflow = yaml.load(
        (WORKFLOW_DIR / "benchmarks.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"workflow_dispatch", "schedule"}
    assert workflow["jobs"]["sft"]["runs-on"] == ["self-hosted", "linux", "gpu"]
    config = SFTConfig.model_validate(
        yaml.safe_load(Path("benchmarks/configs/sft.yaml").read_text())
    )
    assert config.max_steps > 5
    assert config.data.source == "fake"
    upload = workflow["jobs"]["sft"]["steps"][-1]
    assert upload["if"] == "always()"
    assert "benchmark.log" in upload["with"]["path"]


@pytest.mark.parametrize(
    ("filename", "job", "variable"),
    [
        ("gpu-tests.yaml", "pytest", "ENABLE_SCHEDULED_GPU_CHECKS"),
        ("benchmarks.yaml", "sft", "ENABLE_SCHEDULED_GPU_BENCHMARKS"),
    ],
)
def test_scheduled_gpu_jobs_require_explicit_budget_opt_in(filename, job, variable):
    workflow = yaml.load((WORKFLOW_DIR / filename).read_text(), Loader=yaml.BaseLoader)
    assert (
        workflow["jobs"][job]["if"]
        == f"github.event_name == 'workflow_dispatch' || vars.{variable} == 'true'"
    )

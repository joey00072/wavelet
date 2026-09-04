from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from wavelet.utils.config import format_config_help, load_config


class _NestedConfig(BaseModel):
    batch_size: int = Field(default=2, description="Examples in one batch.")
    enabled: bool = Field(default=False, description="Enable nested processing.")
    label: str = "base"


class _TestConfig(BaseModel):
    nested: _NestedConfig = _NestedConfig()
    cache_results: bool = True


def test_config_files_deep_merge_left_to_right(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "nested:\n  batch_size: 4\n  label: preserved\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("nested:\n  enabled: false\n", encoding="utf-8")

    config = load_config(
        _TestConfig,
        [
            "@",
            str(base),
            f"@{overlay}",
            "--nested.batch-size",
            "8",
            "--nested.enabled",
            "--no-cache-results",
        ],
    )

    assert config.nested.batch_size == 8
    assert config.nested.enabled is True
    assert config.nested.label == "preserved"
    assert config.cache_results is False


def test_non_boolean_override_still_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="Expected a value"):
        load_config(_TestConfig, ["--nested.batch-size"])


def test_config_help_uses_nested_field_descriptions_and_kebab_case() -> None:
    output = format_config_help(_TestConfig)

    assert "--nested.batch-size" in output
    assert "Examples in one batch." in output
    assert "--nested.enabled" in output
    assert "--no-..." in output


def test_help_flag_prints_config_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        load_config(_TestConfig, ["--help"])

    assert exc_info.value.code == 0
    assert "_TestConfig configuration" in capsys.readouterr().out

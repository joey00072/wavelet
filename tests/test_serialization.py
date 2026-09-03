from __future__ import annotations

import pytest
import yaml

from wavelet.utils.serialization import load_yaml


def test_load_yaml_rejects_duplicate_mapping_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("optim:\n  lr: 1e-5\n  lr: 2e-5\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'lr'"):
        load_yaml(path)


def test_load_yaml_allows_same_key_in_distinct_mappings(tmp_path) -> None:
    path = tmp_path / "valid.yaml"
    path.write_text("train:\n  lr: 1e-5\neval:\n  lr: 2e-5\n", encoding="utf-8")

    assert load_yaml(path) == {
        "train": {"lr": "1e-5"},
        "eval": {"lr": "2e-5"},
    }

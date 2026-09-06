from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from wavelet.tools.verifier_data import import_verifiers, verifier_rows, write_jsonl


def test_verifier_rows_wraps_examples_and_limits() -> None:
    dataset = [{"prompt": "a"}, {"prompt": "b", "id": 7}, {"prompt": "c"}]

    rows = verifier_rows(dataset, limit=2, id_keys=("id",))

    assert [row["metadata"]["example_id"] for row in rows] == [0, 7]
    assert rows[0] == {
        "prompt": "a",
        "completion": "",
        "metadata": {
            "verifier_example": {"prompt": "a", "example_id": 0},
            "example_id": 0,
        },
    }


def test_verifier_rows_keeps_existing_example_id() -> None:
    rows = verifier_rows([{"prompt": "a", "example_id": "x", "id": 3}], id_keys=("id",))

    assert rows[0]["metadata"]["example_id"] == "x"


def test_write_jsonl_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"

    assert write_jsonl(path, [{"a": 1}, {"b": 2}]) == 2
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"a": 1},
        {"b": 2},
    ]


def test_import_verifiers_reports_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "verifiers", None)
    with pytest.raises(SystemExit, match="uv sync --extra verifiers"):
        import_verifiers()

    fake = types.ModuleType("verifiers")
    monkeypatch.setitem(sys.modules, "verifiers", fake)
    assert import_verifiers() is fake

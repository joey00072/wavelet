from dataclasses import replace
from pathlib import Path

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample
from wavelet.inference.http import HTTPPolicyInferenceEngine


def _record(index: int) -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": str(index)}],
        completion=[],
        advantage=None,
        reward=None,
        source=str(index),
    )


def test_round_robin_annotation_restores_input_order() -> None:
    engine = HTTPPolicyInferenceEngine.__new__(HTTPPolicyInferenceEngine)
    engine.base_urls = ["server-0", "server-1", "server-2"]
    records = [_record(index) for index in range(7)]
    chunks: dict[str, list[str]] = {}

    def annotate_chunk(chunk: list[RLExample], base_url: str) -> list[RLExample]:
        chunks[base_url] = [record.source for record in chunk]
        return [
            replace(record, source=f"{record.source}@{base_url}") for record in chunk
        ]

    annotated = engine._annotate_round_robin(records, annotate_chunk)  # noqa: SLF001

    assert chunks == {
        "server-0": ["0", "3", "6"],
        "server-1": ["1", "4"],
        "server-2": ["2", "5"],
    }
    assert [record.source for record in annotated] == [
        "0@server-0",
        "1@server-1",
        "2@server-2",
        "3@server-0",
        "4@server-1",
        "5@server-2",
        "6@server-0",
    ]


def test_policy_load_uses_all_server_request_path(tmp_path: Path, monkeypatch) -> None:
    engine = HTTPPolicyInferenceEngine(RLConfig(output_dir=tmp_path))
    calls: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        engine,
        "_request_all",
        lambda method, path, payload: calls.append((method, path, payload)),
    )

    engine.load_policy(tmp_path / "policy", step=3)

    assert calls == [
        (
            "POST",
            "/load_policy",
            {
                "policy_dir": str(tmp_path / "policy"),
                "step": 3,
                "adapter_name": "policy",
                "load_inplace": True,
            },
        )
    ]
    assert engine.policy_step == 3

from __future__ import annotations

from types import SimpleNamespace

import wavelet.wandb_overview as overview


def test_overview_inputs_extract_rl_environment_names() -> None:
    flavor, train_envs, eval_envs = overview.overview_inputs(
        {
            "orchestrator": {"verifier_env_id": "reverse-text@0.1.0"},
            "eval": {
                "env": [
                    {"id": "aime2025@1.0.0"},
                    {"id": "math500@1.0.0", "name": "held-out-math"},
                ]
            },
        }
    )

    assert flavor == "rl"
    assert train_envs == ["reverse-text"]
    assert eval_envs == ["aime2025", "held-out-math"]


def test_build_sections_uses_wavelet_metrics_and_eval_names() -> None:
    sections = overview.build_sections(
        flavor="rl",
        train_envs=["reverse-text"],
        eval_envs=["aime+2025"],
    )

    assert [section.name for section in sections] == [
        "train/reverse-text",
        "eval",
        "stability",
        "inference",
        "performance",
        "resources",
    ]
    assert sections[0].panels[0].y[0] == "reward/all/mean"
    assert sections[1].panels[0].metric_regex == (
        r"eval/(aime\+2025)/(avg@.*|pass@.*|pass\^.*|failed_rollouts)"
    )
    assert sections[3].panels[0].x == "RelativeTime(Wall)"


def test_ensure_overview_creates_versioned_view(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeWorkspace:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.url = "https://wandb.example/overview-v3"

        def save(self) -> None:
            captured["saved"] = True

    monkeypatch.setattr(
        overview,
        "_list_views",
        lambda _entity, _project: [
            ("overview", "nw-old-v"),
            ("overview-v2", "nw-other-v"),
        ],
    )
    monkeypatch.setattr(overview.ws, "Workspace", FakeWorkspace)

    url = overview.ensure_overview_view(
        "team",
        "project",
        flavor="sft",
    )

    assert url == "https://wandb.example/overview-v3"
    assert captured["name"] == "overview-v3"
    assert captured["saved"] is True
    assert isinstance(captured["sections"], list)


def test_ensure_overview_reuses_matching_view(monkeypatch) -> None:
    sections = overview.build_sections(flavor="sft")

    class FakeWorkspace:
        @classmethod
        def from_url(cls, _url: str):
            return SimpleNamespace(sections=sections)

    monkeypatch.setattr(
        overview,
        "_list_views",
        lambda _entity, _project: [("overview", "nw-existing-v")],
    )
    monkeypatch.setattr(overview.ws, "Workspace", FakeWorkspace)

    assert overview.ensure_overview_view("team", "project", flavor="sft") is None

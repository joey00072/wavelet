"""Curated W&B workspace views for Wavelet runs."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import execute_graphql

OVERVIEW_NAME = "overview"
logger = logging.getLogger(__name__)
_COLUMNS = 4
_ROWS = 6

_RL_TRAIN_METRICS = (
    "reward/all/mean",
    "advantage/all/mean",
    "progress/samples",
    "progress/tokens",
    "fate/all/trainable",
    "fate/all/filtered",
    "fate/all/errored",
)
_SFT_TRAIN_METRICS = (
    "loss",
    "train/loss",
    "val/loss",
    "progress/epoch",
    "progress/total_samples",
    "progress/total_tokens",
)
_STABILITY_METRICS = (
    "optim/grad_norm",
    "entropy/mean",
    "kl/mismatch",
    "lr",
)
_PERFORMANCE_METRICS = (
    "perf/mfu",
    "perf/tokens_per_second",
    "perf/train_tokens_per_second",
    "time/step",
    "time/forward_backward",
    "time/wait_for_batch",
)
_RESOURCE_METRICS = (
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "disk_free_ratio",
    "checkpoint_disk_free_ratio",
)


def _line_panels(
    metrics: Sequence[str] = (),
    regexes: Sequence[str] = (),
) -> list[wr.LinePlot]:
    return [wr.LinePlot(x="step", y=[metric]) for metric in metrics] + [
        wr.LinePlot(x="step", metric_regex=pattern) for pattern in regexes
    ]


def _section(
    name: str,
    *,
    metrics: Sequence[str] = (),
    regexes: Sequence[str] = (),
) -> ws.Section:
    return ws.Section(
        name=name,
        is_open=True,
        panels=_line_panels(metrics, regexes),
        layout_settings=ws.SectionLayoutSettings(columns=_COLUMNS, rows=_ROWS),
    )


def _inference_section() -> ws.Section:
    return ws.Section(
        name="inference",
        is_open=True,
        panels=[wr.LinePlot(x="RelativeTime(Wall)", metric_regex=r"inference/.*")],
        layout_settings=ws.SectionLayoutSettings(columns=_COLUMNS, rows=_ROWS),
    )


def build_sections(
    *,
    flavor: Literal["rl", "sft"],
    train_envs: Sequence[str] = (),
    eval_envs: Sequence[str] = (),
) -> list[ws.Section]:
    """Build panels using Wavelet's canonical metric names."""
    if flavor == "sft":
        train = _section("train", metrics=_SFT_TRAIN_METRICS)
    else:
        train_name = f"train/{train_envs[0]}" if len(train_envs) == 1 else "train"
        train = _section(train_name, metrics=_RL_TRAIN_METRICS)

    escaped_envs = "|".join(re.escape(name) for name in eval_envs) or ".*"
    evaluation = _section(
        "eval",
        regexes=(rf"eval/({escaped_envs})/(avg@.*|pass@.*|pass\^.*|failed_rollouts)",),
    )
    return [
        train,
        evaluation,
        _section("stability", metrics=_STABILITY_METRICS),
        _inference_section(),
        _section("performance", metrics=_PERFORMANCE_METRICS),
        _section("resources", metrics=_RESOURCE_METRICS),
    ]


def overview_inputs(
    run_config: Mapping[str, Any] | None,
) -> tuple[Literal["rl", "sft"], list[str], list[str]]:
    """Infer overview flavor and environment labels from a serialized config."""
    if not run_config or "orchestrator" not in run_config:
        return "sft", [], []

    orchestrator = run_config.get("orchestrator")
    train_envs: list[str] = []
    if isinstance(orchestrator, Mapping):
        env_id = orchestrator.get("verifier_env_id")
        if isinstance(env_id, str) and env_id:
            train_envs.append(env_id.split("@", 1)[0])

    eval_envs: list[str] = []
    evaluation = run_config.get("eval")
    if isinstance(evaluation, Mapping):
        environments = evaluation.get("env")
        if isinstance(environments, list):
            for environment in environments:
                if not isinstance(environment, Mapping):
                    continue
                name = environment.get("name") or environment.get("id")
                if isinstance(name, str) and name:
                    eval_envs.append(name.split("@", 1)[0])
    return "rl", train_envs, eval_envs


def _list_views(entity: str, project: str) -> list[tuple[str, str]]:
    query = """
        query Views($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            allViews(viewType: "project-view") {
              edges { node { name displayName } }
            }
          }
        }
    """
    response = execute_graphql(
        wandb.Api(),
        query,
        {"entity": entity, "project": project},
    )
    edges = ((response.get("project") or {}).get("allViews") or {}).get("edges") or []
    return [
        (node["displayName"], node["name"])
        for edge in edges
        if isinstance(edge, Mapping)
        and isinstance((node := edge.get("node")), Mapping)
        and isinstance(node.get("displayName"), str)
        and isinstance(node.get("name"), str)
    ]


def _view_signature(sections: Sequence[ws.Section]) -> tuple[object, ...]:
    return tuple(
        (
            section.name,
            tuple(
                (
                    getattr(panel.x, "name", panel.x),
                    tuple(getattr(metric, "name", metric) for metric in panel.y or ()),
                    panel.metric_regex,
                )
                for panel in section.panels
                if isinstance(panel, wr.LinePlot)
            ),
        )
        for section in sections
    )


def _next_name(base: str, existing: Sequence[str]) -> str:
    if base not in existing:
        return base
    prefix = f"{base}-v"
    versions = [
        int(name.removeprefix(prefix))
        for name in existing
        if name.startswith(prefix) and name.removeprefix(prefix).isdigit()
    ]
    return f"{base}-v{max(versions, default=1) + 1}"


def ensure_overview_view(
    entity: str,
    project: str,
    *,
    flavor: Literal["rl", "sft"],
    train_envs: Sequence[str] = (),
    eval_envs: Sequence[str] = (),
) -> str | None:
    """Create a versioned overview unless an equivalent saved view exists."""
    sections = build_sections(
        flavor=flavor,
        train_envs=train_envs,
        eval_envs=eval_envs,
    )
    target = _view_signature(sections)
    views = [
        (display_name, internal_name)
        for display_name, internal_name in _list_views(entity, project)
        if display_name == OVERVIEW_NAME
        or display_name.startswith(f"{OVERVIEW_NAME}-v")
    ]
    for _, internal_name in views:
        slug = internal_name.removeprefix("nw-").removesuffix("-v")
        try:
            existing = ws.Workspace.from_url(
                f"https://wandb.ai/{entity}/{project}?nw={slug}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not inspect W&B overview %s: %s", internal_name, exc)
            continue
        if _view_signature(existing.sections) == target:
            return None

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=_next_name(OVERVIEW_NAME, [name for name, _ in views]),
        sections=sections,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace.save()
    return workspace.url

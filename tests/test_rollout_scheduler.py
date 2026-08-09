import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.scheduler import (
    IntegratedRolloutScheduler,
    PublishMode,
    _ChunkPublisherStrategy,
    _SchedulerStateMachine,
    resolve_rollout_schedule,
)
from wavelet.orchestrator.sources import RolloutSourceKind


@pytest.mark.parametrize(
    ("mode", "function", "async_level", "source", "publish_mode"),
    [
        ("integrated", None, 0, RolloutSourceKind.NATIVE, PublishMode.BATCH),
        ("process", None, 0, RolloutSourceKind.NATIVE, PublishMode.BATCH),
        ("process", None, 2, RolloutSourceKind.NATIVE, PublishMode.STREAMING),
        ("process", None, 2, RolloutSourceKind.NATIVE, PublishMode.BATCH),
        (
            "process",
            "wavelet.orchestrator.verifiers:generate_rollouts",
            2,
            RolloutSourceKind.VERIFIER,
            PublishMode.STREAMING,
        ),
        (
            "process",
            "package.module:rollout",
            2,
            RolloutSourceKind.CUSTOM,
            PublishMode.BATCH,
        ),
    ],
)
def test_legacy_scheduler_configs_resolve_explicitly(
    mode,
    function,
    async_level,
    source,
    publish_mode,
):
    examples_per_step = None if (
        mode == "process"
        and function is None
        and async_level == 2
        and publish_mode is PublishMode.BATCH
    ) else 8
    config = RLConfig(
        launcher={"mode": mode},
        orchestrator={
            "custom_rollout_function": function,
            "max_async_level": async_level,
            "examples_per_step": examples_per_step,
        },
    )

    schedule = resolve_rollout_schedule(config)

    assert schedule.source is source
    assert schedule.publish_mode is publish_mode
    assert schedule.is_sync is (async_level == 0)
    expected_chunk_examples = 1 if examples_per_step is None else (
        8 if async_level == 0 else 4
    )
    assert schedule.chunk_examples == expected_chunk_examples


def test_integrated_scheduler_runs_one_ordered_cycle_per_step() -> None:
    step = 0
    events: list[tuple[str, int]] = []

    def consume() -> None:
        nonlocal step
        events.append(("consume", step))
        step += 1

    IntegratedRolloutScheduler(
        target_step=2,
        current_step=lambda: step,
        prepare_policy=lambda value: events.append(("prepare", value)),
        publish=lambda value: events.append(("publish", value)),
        consume_and_train=consume,
        after_step=lambda: events.append(("after", step)),
    ).run()

    assert events == [
        ("prepare", 0),
        ("publish", 0),
        ("consume", 0),
        ("after", 1),
        ("prepare", 1),
        ("publish", 1),
        ("consume", 1),
        ("after", 2),
    ]


def test_scheduler_strategies_share_queue_to_optimizer_step_mapping() -> None:
    batched = object.__new__(_SchedulerStateMachine)
    chunked = object.__new__(_ChunkPublisherStrategy)
    chunked.chunks_per_step = 3

    assert batched._rollout_step(4) == 4
    assert [chunked._rollout_step(step) for step in range(7)] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
    ]

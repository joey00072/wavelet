from unittest.mock import Mock

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.scheduler import (
    IntegratedRolloutScheduler,
    PublishMode,
    _ChunkPublisherStrategy,
    _resume_optimizer_step,
    _reusable_rollout_batch,
    _SchedulerStateMachine,
    resolve_rollout_schedule,
)
from wavelet.orchestrator.sources import RolloutSourceKind
from wavelet.transport.queue import FileSystemPolicyReceiver, FileSystemRolloutSender
from wavelet.utils.pathing import STABLE_CHECKPOINT_MARKER


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
    examples_per_step = (
        None
        if (
            mode == "process"
            and function is None
            and async_level == 2
            and publish_mode is PublishMode.BATCH
        )
        else 8
    )
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
    expected_chunk_examples = (
        1 if examples_per_step is None else (8 if async_level == 0 else 4)
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


def test_process_scheduler_resumes_from_latest_stable_optimizer_step(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-7"
    checkpoint_dir.mkdir()
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    config = RLConfig(
        output_dir=tmp_path,
        ckpt={"mode": "async", "interval": 1, "resume_step": -1},
    )

    assert _resume_optimizer_step(config) == 7


def test_process_scheduler_resumes_from_checkpoint_output_dir(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "checkpoint-7"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    config = RLConfig(
        output_dir=tmp_path / "run",
        ckpt={
            "mode": "async",
            "interval": 1,
            "resume_step": -1,
            "output_dir": checkpoint_root,
        },
    )

    assert _resume_optimizer_step(config) == 7


def test_process_scheduler_resumes_from_external_checkpoint_dir(tmp_path) -> None:
    checkpoint_dir = tmp_path / "other-run" / "checkpoint-9"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    config = RLConfig(
        output_dir=tmp_path / "new-run",
        ckpt={"resume_dir": checkpoint_dir},
    )

    assert _resume_optimizer_step(config) == 9


def test_process_scheduler_resets_step_when_progress_restore_is_skipped(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "other-run" / "checkpoint-9"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    config = RLConfig(
        output_dir=tmp_path / "new-run",
        ckpt={"resume_dir": checkpoint_dir, "skip_progress": True},
    )

    assert _resume_optimizer_step(config) == 0


def test_process_scheduler_rejects_checkpoint_after_target_step(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-7"
    checkpoint_dir.mkdir()
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    config = RLConfig(
        output_dir=tmp_path,
        max_steps=6,
        ckpt={"mode": "async", "interval": 1, "resume_step": 7},
    )

    with pytest.raises(ValueError, match="exceeds configured max_steps"):
        _resume_optimizer_step(config)


def test_chunk_scheduler_initializes_in_resumed_queue_step_space(tmp_path) -> None:
    config = RLConfig(output_dir=tmp_path)
    context = _ChunkPublisherStrategy(
        config=config,
        orchestrator=Mock(),
        inference_engine=Mock(),
        policy_receiver=FileSystemPolicyReceiver(
            tmp_path,
            config.policy_transfer,
        ),
        state=None,
        rollout_sender=FileSystemRolloutSender(tmp_path, config.transport),
        chunks_per_step=4,
        chunk_examples=8,
        next_step_to_submit=28,
        next_step_to_publish=28,
    )

    assert context.next_step_to_submit == 28
    assert context.next_step_to_publish == 28


def test_resume_reuses_only_valid_stable_rollout_batch(tmp_path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={"max_async_level": 2, "max_off_policy_steps": 1},
    )
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    expected = sender.publish(
        source,
        step=2,
        optimizer_step=2,
        policy_step=1,
        rows=1,
    )

    reused = _reusable_rollout_batch(
        config,
        sender,
        queue_step=2,
        optimizer_step=2,
        chunk_index=None,
    )

    assert reused == expected


def test_resume_rejects_stale_stable_rollout_batch(tmp_path) -> None:
    config = RLConfig(
        output_dir=tmp_path,
        orchestrator={"max_async_level": 2, "max_off_policy_steps": 1},
    )
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    sender = FileSystemRolloutSender(tmp_path, config.transport)
    sender.publish(
        source,
        step=3,
        optimizer_step=3,
        policy_step=0,
        rows=1,
    )

    with pytest.raises(ValueError, match="policy_step=0"):
        _reusable_rollout_batch(
            config,
            sender,
            queue_step=3,
            optimizer_step=3,
            chunk_index=None,
        )

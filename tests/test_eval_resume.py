from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from wavelet.orchestrator.envs import _run_eval_examples
from wavelet.orchestrator.eval_utils import EvaluationJournal, evaluation_signature


def _run(env, examples, journal, *, group_size=1, concurrency=2):
    return _run_eval_examples(
        SimpleNamespace(RolloutInput=lambda **kwargs: kwargs),
        env,
        examples,
        clients=[object()],
        model="model",
        sampling_args={"seed": 3},
        rollouts_per_example=group_size,
        max_retries=0,
        max_inflight_rollouts=concurrency,
        journal=journal,
    )


def test_interrupted_eval_keeps_committed_episodes_and_cancels_workers(tmp_path):
    async def exercise():
        committed = asyncio.Event()
        calls = []
        cancelled = []

        async def rollout(example, **_kwargs):
            calls.append(example["id"])
            if example["id"] == "a":
                committed.set()
                return {"reward": 1.0, "error": None, "completion": []}
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(example["id"])

        examples = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        path = tmp_path / "journal"
        with EvaluationJournal(path, signature="policy") as journal:
            pending = asyncio.create_task(
                _run(SimpleNamespace(run_rollout=rollout), examples, journal)
            )
            await committed.wait()
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert cancelled
        resumed_calls = []

        async def resumed(example, **_kwargs):
            resumed_calls.append(example["id"])
            return {"reward": 0.5}

        with EvaluationJournal(path, signature="policy") as journal:
            outputs = await _run(
                SimpleNamespace(run_rollout=resumed), examples, journal
            )
        assert resumed_calls == ["b", "c"]
        assert [row["reward"] for row in outputs] == [1.0, 0.5, 0.5]

    asyncio.run(exercise())


def test_eval_can_extend_groups_and_reorder_identified_tasks(tmp_path):
    calls = []

    async def rollout(example, **kwargs):
        calls.append((example["id"], kwargs["sampling_args"]["seed"]))
        return {"reward": 1.0}

    env = SimpleNamespace(run_rollout=rollout)
    path = tmp_path / "journal"
    with EvaluationJournal(path, signature="plan") as journal:
        asyncio.run(_run(env, [{"id": "a"}, {"id": "b"}], journal))
    calls.clear()
    with EvaluationJournal(path, signature="plan") as journal:
        outputs = asyncio.run(
            _run(env, [{"id": "b"}, {"id": "a"}], journal, group_size=2)
        )
    assert calls == [("b", 4), ("a", 4)]
    assert len(outputs) == 4


def test_changed_task_content_does_not_reuse_old_result(tmp_path):
    calls = []

    async def rollout(example, **_kwargs):
        calls.append(example["prompt"])
        return {"reward": 1.0}

    for prompt in ("first", "changed"):
        with EvaluationJournal(tmp_path / "journal", signature="plan") as journal:
            asyncio.run(
                _run(
                    SimpleNamespace(run_rollout=rollout),
                    [{"id": "a", "prompt": prompt}],
                    journal,
                )
            )
    assert calls == ["first", "changed"]


def test_journal_structural_messages_and_failed_outputs_roundtrip(tmp_path):
    class Message(BaseModel):
        role: str
        content: str

    async def rollout(example, **_kwargs):
        if example["id"] == "failed":
            return {"reward": 0.0, "error": RuntimeError("failed")}
        return {
            "reward": 1.0,
            "completion": [Message(role="assistant", content="ok")],
            "error": None,
        }

    examples = [{"id": "success"}, {"id": "failed"}]
    with EvaluationJournal(tmp_path / "journal", signature="plan") as journal:
        initial = asyncio.run(
            _run(SimpleNamespace(run_rollout=rollout), examples, journal)
        )
    calls = []

    async def again(example, **kwargs):
        calls.append(example["id"])
        return await rollout(example, **kwargs)

    with EvaluationJournal(tmp_path / "journal", signature="plan") as journal:
        resumed = asyncio.run(
            _run(SimpleNamespace(run_rollout=again), examples, journal)
        )
    assert resumed == initial
    assert calls == ["failed"]
    assert resumed[0]["completion"] == [{"role": "assistant", "content": "ok"}]
    assert "reward" not in resumed[1]


def test_journal_ignores_uncommitted_files_but_rejects_committed_corruption(tmp_path):
    path = tmp_path / "journal"
    with EvaluationJournal(path, signature="plan") as journal:
        journal.record(0, {"reward": 1.0})
    (path / "episode-other.json.tmp").write_text('{"partial":')
    with EvaluationJournal(path, signature="plan") as journal:
        assert journal.completed(0)["reward"] == 1.0
    next(path.glob("episode-*.json")).write_text('{"partial":')
    with pytest.raises(json.JSONDecodeError), EvaluationJournal(path, signature="plan"):
        pass


def test_journal_requires_explicit_resume_and_excludes_concurrent_writer(tmp_path):
    path = tmp_path / "journal"
    with (
        EvaluationJournal(path, signature="plan"),
        pytest.raises(BlockingIOError),
        EvaluationJournal(path, signature="plan"),
    ):
        pass
    with (
        pytest.raises(FileExistsError, match="eval.resume"),
        EvaluationJournal(path, signature="plan", resume=False),
    ):
        pass


def test_eval_signature_distinguishes_policy_and_sampling():
    original = evaluation_signature(model="model", policy_step=1, sampling={"seed": 3})
    assert original != evaluation_signature(
        model="model", policy_step=2, sampling={"seed": 3}
    )
    assert original != evaluation_signature(
        model="model", policy_step=1, sampling={"seed": 4}
    )


@pytest.mark.parametrize("reward", ["bad", "nan", float("inf"), float("nan"), True])
def test_invalid_reward_is_failed_and_retried(tmp_path, reward):
    async def rollout(*_args, **_kwargs):
        return {"reward": reward}

    with EvaluationJournal(tmp_path / "journal", signature="plan") as journal:
        output = asyncio.run(
            _run(SimpleNamespace(run_rollout=rollout), [{"id": "a"}], journal)
        )[0]
        assert "reward" not in output
        assert "finite number" in output["error"]
        assert all(journal.completed(key) is None for key in journal.rows)


def test_provider_limit_is_classified_before_error_truncation(tmp_path):
    async def rollout(*_args, **_kwargs):
        return {"error": "x" * 600 + " rate limit 429"}

    with (
        EvaluationJournal(tmp_path / "journal", signature="plan") as journal,
        pytest.raises(RuntimeError, match="rate limit"),
    ):
        asyncio.run(_run(SimpleNamespace(run_rollout=rollout), [{"id": "a"}], journal))

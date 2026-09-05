from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import RLExample, load_rl_records, serialize_rl_record
from wavelet.inference.policy import RLInference
from wavelet.orchestrator.admission import RolloutAdmissionController
from wavelet.orchestrator.algorithms import (
    algorithm_epsilon,
    algorithm_scope,
    build_algorithm,
    score_algorithm_records,
    uses_group_advantages,
)
from wavelet.orchestrator.reward import RLRewardScorer, assistant_text
from wavelet.orchestrator.schedule import required_policy_step
from wavelet.transport.queue import (
    FileSystemRolloutSender,
    QueueEvent,
    RolloutBatch,
    append_event_best_effort,
    utc_now,
    validate_rollout_manifest,
)

CustomRolloutFunction = Callable[
    ["RLOrchestrator", list[RLExample], object | None],
    list[RLExample],
]

logger = logging.getLogger(__name__)


class RLOrchestrator:
    def __init__(self, config: RLConfig) -> None:
        self.config = config
        self._verifier_admission: RolloutAdmissionController | None = None
        self._rollout_metrics: dict[str, float] = {}

    def verifier_admission(
        self,
        *,
        max_inflight: int,
        minimum_burst: int,
    ) -> RolloutAdmissionController:
        """Return the run-scoped verifier admission and rate controller."""
        if self._verifier_admission is None:
            self._verifier_admission = RolloutAdmissionController(
                max_inflight=max_inflight,
                minimum_burst=minimum_burst,
                tasks_per_minute=self.config.orchestrator.tasks_per_minute,
            )
        return self._verifier_admission

    def add_rollout_metrics(self, metrics: dict[str, float]) -> None:
        """Accumulate metrics produced before materialized rows exist."""
        for name, value in metrics.items():
            self._rollout_metrics[name] = self._rollout_metrics.get(name, 0.0) + value

    def consume_rollout_metrics(self) -> dict[str, float]:
        """Return and clear metrics accumulated for the next published batch."""
        metrics = self._rollout_metrics
        self._rollout_metrics = {}
        return metrics

    def materialize(
        self,
        *,
        step: int | None = None,
        inference_engine=None,
    ) -> Path:
        attempts = self.config.orchestrator.zero_advantage_max_retries + 1
        trainable_records: list[RLExample] = []
        for retry in range(attempts):
            records = self._load_step_records(step=step, retry=retry)
            scored_records = self._generate_and_score(
                records,
                inference_engine=inference_engine,
            )
            trainable_records = self._append_new_groups(
                trainable_records,
                self._filter_zero_advantage_records(scored_records),
                target_groups=self.config.orchestrator.examples_per_step,
            )
            if self._has_target_groups(trainable_records):
                break
        else:
            accepted_groups = self._group_count(trainable_records)
            raise RuntimeError(
                "Could not materialize the requested rollout group count after "
                f"{attempts} generation attempt(s) (accepted "
                f"{accepted_groups}). Increase "
                "orchestrator.zero_advantage_max_retries, relax filtering, or "
                "check the reward/model output format."
            )
        return self._write_records(trainable_records, step=step)

    def materialize_native_chunk(
        self,
        *,
        optimizer_step: int,
        chunk_index: int,
        queue_step: int,
        chunk_examples: int,
        inference_engine=None,
    ) -> Path:
        if self.config.orchestrator.custom_rollout_function is not None:
            raise ValueError("Native rollout chunks require native rollouts.")
        attempts = self.config.orchestrator.zero_advantage_max_retries + 1
        trainable_records: list[RLExample] = []
        expected_groups: int | None = None
        for retry in range(attempts):
            records = self._load_native_chunk_records(
                optimizer_step=optimizer_step,
                chunk_index=chunk_index,
                chunk_examples=chunk_examples,
                retry=retry,
            )
            if expected_groups is None:
                expected_groups = len(records)
            scored_records = self._generate_and_score(
                records,
                inference_engine=inference_engine,
            )
            trainable_records = self._append_new_groups(
                trainable_records,
                self._filter_zero_advantage_records(scored_records),
                target_groups=expected_groups,
            )
            if self._has_target_groups(
                trainable_records,
                target_groups=expected_groups,
            ):
                break
        else:
            accepted_groups = self._group_count(trainable_records)
            raise RuntimeError(
                "Could not materialize the requested native chunk group count after "
                f"{attempts} generation attempt(s) (accepted "
                f"{accepted_groups}). Increase "
                "orchestrator.zero_advantage_max_retries, relax filtering, or "
                "check the reward/model output format."
            )
        return self._write_records(trainable_records, step=queue_step)

    def _append_new_groups(
        self,
        accumulated: list[RLExample],
        candidates: list[RLExample],
        *,
        target_groups: int | None,
    ) -> list[RLExample]:
        """Collect unique rollout groups across bounded generation attempts."""
        result = list(accumulated)
        seen = {self._group_key(record) for record in result}
        grouped: dict[str, list[RLExample]] = {}
        for record in candidates:
            grouped.setdefault(self._group_key(record), []).append(record)
        for key, group in grouped.items():
            if key in seen:
                continue
            if target_groups is not None and len(seen) >= target_groups:
                break
            result.extend(group)
            seen.add(key)
        return result

    def _group_count(self, records: list[RLExample]) -> int:
        return len({self._group_key(record) for record in records})

    def _has_target_groups(
        self,
        records: list[RLExample],
        *,
        target_groups: int | None = None,
    ) -> bool:
        target = (
            self.config.orchestrator.examples_per_step
            if target_groups is None
            else target_groups
        )
        if target is None:
            return bool(records)
        return self._group_count(records) >= target

    def _load_step_records(
        self,
        *,
        step: int | None,
        retry: int,
    ) -> list[RLExample]:
        records = load_rl_records(self.config.data)
        return self._select_step_records(
            records,
            seed=self._step_seed(step=step, retry=retry),
        )

    def _load_native_chunk_records(
        self,
        *,
        optimizer_step: int,
        chunk_index: int,
        chunk_examples: int,
        retry: int,
    ) -> list[RLExample]:
        examples_per_step = self.config.orchestrator.examples_per_step
        if examples_per_step is None:
            raise ValueError("orchestrator.examples_per_step is required.")
        if chunk_examples < 1:
            raise ValueError("chunk_examples must be at least 1.")
        records = load_rl_records(self.config.data)
        selected = self._select_step_records(
            records,
            seed=self._step_seed(step=optimizer_step, retry=retry),
            limit=examples_per_step,
        )
        start = chunk_index * chunk_examples
        chunk_records = selected[start : start + chunk_examples]
        if not chunk_records:
            raise RuntimeError(
                f"Native rollout chunk {chunk_index} for optimizer step "
                f"{optimizer_step} is empty."
            )
        return chunk_records

    def _step_seed(self, *, step: int | None, retry: int) -> int:
        data_config = self.config.data
        step_seed = data_config.seed
        if step is not None:
            step_seed += step
        if retry:
            step_seed += retry * 1_000_003
        return step_seed

    def _select_step_records(
        self,
        records: list[RLExample],
        *,
        seed: int,
        limit: int | None = None,
    ) -> list[RLExample]:
        if limit is None:
            limit = self.config.orchestrator.examples_per_step
        if limit is None or len(records) <= limit:
            return records
        if self.config.orchestrator.custom_rollout_function is not None:
            limit = min(
                len(records),
                math.ceil(limit * self.config.orchestrator.oversampling_factor),
            )
        rng = random.Random(seed)
        start = rng.randrange(len(records))
        return [records[(start + offset) % len(records)] for offset in range(limit)]

    def _generate_and_score(
        self,
        records: list[RLExample],
        *,
        inference_engine=None,
    ) -> list[RLExample]:
        if self.config.orchestrator.custom_rollout_function is not None:
            custom_rollout = self._load_custom_rollout_function(
                self.config.orchestrator.custom_rollout_function
            )
            return self.trim_to_step_examples(
                self._assign_advantages(custom_rollout(self, records, inference_engine))
            )

        inference = RLInference(self.config)
        rollout_records = self._expand_native_rollout_records(records)
        if inference_engine is not None:
            annotated = inference_engine.annotate(rollout_records)
        else:
            annotated = inference.annotate(rollout_records)
        scorer = RLRewardScorer(self.config.reward)
        annotated = self._filter_degenerate_native_rollout_records(annotated)
        annotated = self._drop_incomplete_native_rollout_groups(annotated)
        return self._assign_advantages(
            [self._score_record(record, scorer) for record in annotated]
        )

    def _expand_native_rollout_records(
        self,
        records: list[RLExample],
    ) -> list[RLExample]:
        rollouts_per_example = self.config.orchestrator.rollouts_per_example
        if rollouts_per_example <= 1:
            return records

        expanded: list[RLExample] = []
        for record in records:
            for rollout_index in range(rollouts_per_example):
                metadata = dict(record.metadata or {})
                metadata.setdefault("rollout_index", rollout_index)
                expanded.append(replace(record, metadata=metadata))
        return expanded

    def publish(
        self,
        *,
        step: int,
        inference_engine=None,
        policy_step: int | None = None,
    ) -> RolloutBatch:
        """Publish the rollout batch for ``step``, reusing a valid stable batch."""
        sender = FileSystemRolloutSender(self.config.output_dir, self.config.transport)
        existing = _reusable_rollout_batch(
            self.config,
            sender,
            queue_step=step,
            optimizer_step=step,
            chunk_index=None,
        )
        if existing is not None:
            return existing
        materialized_path = self.materialize(
            step=step,
            inference_engine=inference_engine,
        )
        return sender.publish(
            materialized_path,
            step=step,
            optimizer_step=step,
            policy_step=policy_step,
            rows=_count_nonempty_lines(materialized_path),
        )

    def run(
        self, *, start_step: int = 0, max_steps: int | None = None
    ) -> list[RolloutBatch]:
        total_steps = (
            max_steps
            if max_steps is not None
            else self.config.max_steps
            if self.config.max_steps is not None
            else 1
        )
        published: list[RolloutBatch] = []
        for offset in range(total_steps):
            published.append(self.publish(step=start_step + offset))
        return published

    def _resolve_output_path(self, *, step: int | None = None) -> Path:
        configured = self.config.orchestrator.materialize_path
        if configured is not None:
            return Path(configured)
        if step is not None:
            return (
                self.config.output_dir
                / "rollouts"
                / f"materialized-step-{step:06d}.jsonl"
            )
        return (
            self.config.output_dir / "rollouts" / self.config.transport.rollout_filename
        )

    def _write_records(
        self,
        records: list[RLExample],
        *,
        step: int | None,
    ) -> Path:
        if not records:
            raise RuntimeError("Rollout materialization produced no trainable records.")
        output_path = self._resolve_output_path(step=step)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not self.config.orchestrator.overwrite:
            raise FileExistsError(
                f"Rollout file '{output_path}' already exists and overwrite is "
                "disabled."
            )
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.temperatures is None:
                    raise ValueError("Rollout record is missing temperatures.")
                handle.write(json.dumps(self._serialize_record(record)) + "\n")
        return output_path

    def _serialize_record(self, record: RLExample) -> dict[str, object]:
        return serialize_rl_record(
            record,
            self.config.data,
            task=self.config.reward.mode,
            example_id=self._example_id(record),
        )

    def _score_record(
        self,
        record: RLExample,
        scorer: RLRewardScorer,
    ) -> RLExample:
        return replace(record, reward=scorer.score(record))

    def trim_to_step_examples(self, records: list[RLExample]) -> list[RLExample]:
        limit = self.config.orchestrator.examples_per_step
        if limit is None:
            return records
        selected_keys: set[str] = set()
        trimmed: list[RLExample] = []
        for record in records:
            key = self._group_key(record)
            if key not in selected_keys:
                if len(selected_keys) >= limit:
                    break
                selected_keys.add(key)
            trimmed.append(record)
        return trimmed

    def _assign_advantages(self, records: list[RLExample]) -> list[RLExample]:
        algorithm = build_algorithm(self.config.algo)
        return score_algorithm_records(
            algorithm,
            records,
            scope=algorithm_scope(self.config.algo),
            group_key=self._group_key,
        )

    def _filter_zero_advantage_records(
        self, records: list[RLExample]
    ) -> list[RLExample]:
        if not self.config.orchestrator.filter_zero_advantage:
            return records
        if not uses_group_advantages(self.config.algo):
            return records
        epsilon = algorithm_epsilon(self.config.algo)
        return [
            record
            for record in records
            if record.advantage is not None and abs(float(record.advantage)) > epsilon
        ]

    def _filter_degenerate_native_rollout_records(
        self,
        records: list[RLExample],
    ) -> list[RLExample]:
        valid: list[RLExample] = []
        dropped_empty = 0
        dropped_untrainable = 0
        for record in records:
            if not assistant_text(record.completion).strip():
                dropped_empty += 1
                continue
            if (
                record.loss_mask is not None
                and sum(bool(item) for item in record.loss_mask) == 0
            ):
                dropped_untrainable += 1
                continue
            valid.append(record)

        dropped = dropped_empty + dropped_untrainable
        if dropped:
            logger.warning(
                "Dropped %s degenerate native rollout(s) before scoring "
                "(empty_completion=%s, untrainable=%s).",
                dropped,
                dropped_empty,
                dropped_untrainable,
            )
        return valid

    def _drop_incomplete_native_rollout_groups(
        self,
        records: list[RLExample],
    ) -> list[RLExample]:
        if (
            not uses_group_advantages(self.config.algo)
            or self.config.orchestrator.rollouts_per_example <= 1
        ):
            return records

        grouped: dict[str, list[RLExample]] = {}
        for record in records:
            grouped.setdefault(self._group_key(record), []).append(record)

        expected = self.config.orchestrator.rollouts_per_example
        complete_group_keys = {
            key for key, group in grouped.items() if len(group) == expected
        }
        dropped = sum(
            len(group)
            for key, group in grouped.items()
            if key not in complete_group_keys
        )
        if dropped:
            logger.warning(
                "Dropped %s rollout(s) from native group(s) with unexpected "
                "cardinality before group-reward scoring.",
                dropped,
            )
        return [
            record
            for record in records
            if self._group_key(record) in complete_group_keys
        ]

    def _group_key(self, record: RLExample) -> str:
        if record.metadata is not None and "group_key" in record.metadata:
            return str(record.metadata["group_key"])
        payload = {
            "prompt": record.prompt,
            "target_completion": record.target_completion,
        }
        return json.dumps(payload, sort_keys=True)

    def _example_id(self, record: RLExample) -> str:
        return hashlib.sha1(self._group_key(record).encode("utf-8")).hexdigest()[:12]

    def _load_custom_rollout_function(
        self,
        function_path: str,
    ) -> CustomRolloutFunction:
        if ":" not in function_path:
            raise ValueError(
                "orchestrator.custom_rollout_function must be formatted as "
                "'module.path:function_name'."
            )
        module_name, function_name = function_path.split(":", 1)
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"Custom rollout function is not callable: {function_path}")
        return function


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _reusable_rollout_batch(
    config: RLConfig,
    sender: FileSystemRolloutSender,
    *,
    queue_step: int,
    optimizer_step: int,
    chunk_index: int | None,
) -> RolloutBatch | None:
    """Return the stable batch for ``queue_step`` if the trainer would accept it."""
    batch = sender.stable_batch(queue_step)
    if batch is None:
        return None
    row_count = _count_nonempty_lines(batch.path)
    validate_rollout_manifest(
        batch,
        queue_step=queue_step,
        optimizer_step=optimizer_step,
        chunk_index=chunk_index,
        rows=row_count,
        minimum_policy_step=required_policy_step(config, optimizer_step),
        maximum_policy_step=optimizer_step,
    )
    append_event_best_effort(
        config.output_dir / "events",
        QueueEvent(
            time=utc_now(),
            kind="rollout_reused",
            queue_step=queue_step,
            optimizer_step=optimizer_step,
            details={"path": str(batch.path)},
        ),
    )
    return batch

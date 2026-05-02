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
from wavelet.data.rl_dataset import RLExample, load_rl_records
from wavelet.inference.policy import RLInference
from wavelet.orchestrator.reward import RLRewardScorer
from wavelet.orchestrator.queue import FileSystemRolloutSender, RolloutBatch

CustomRolloutFunction = Callable[
    ["RLOrchestrator", list[RLExample], object | None],
    list[RLExample],
]

logger = logging.getLogger(__name__)


def _assistant_text(messages: list[dict[str, str]] | None) -> str:
    if messages is None:
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    )


class RLOrchestrator:
    def __init__(self, config: RLConfig) -> None:
        self.config = config

    def materialize(
        self,
        *,
        step: int | None = None,
        inference_engine=None,
    ) -> Path:
        attempts = self.config.orchestrator.zero_advantage_max_retries + 1
        last_scored_records: list[RLExample] = []
        for retry in range(attempts):
            records = self._load_step_records(step=step, retry=retry)
            scored_records = self._generate_and_score(
                records,
                inference_engine=inference_engine,
            )
            trainable_records = self._filter_zero_advantage_records(scored_records)
            last_scored_records = trainable_records
            if trainable_records:
                break
        else:
            raise RuntimeError(
                "All rollout groups were filtered after "
                f"{attempts} generation attempt(s). Increase "
                "orchestrator.zero_advantage_max_retries, relax filtering, or "
                "check the reward/model output format."
            )
        return self._write_records(last_scored_records, step=step)

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
        last_scored_records: list[RLExample] = []
        for retry in range(attempts):
            records = self._load_native_chunk_records(
                optimizer_step=optimizer_step,
                chunk_index=chunk_index,
                chunk_examples=chunk_examples,
                retry=retry,
            )
            scored_records = self._generate_and_score(
                records,
                inference_engine=inference_engine,
            )
            trainable_records = self._filter_zero_advantage_records(scored_records)
            last_scored_records = trainable_records
            if trainable_records:
                break
        else:
            raise RuntimeError(
                "All rollout groups in the native chunk were filtered after "
                f"{attempts} generation attempt(s). Increase "
                "orchestrator.zero_advantage_max_retries, relax filtering, or "
                "check the reward/model output format."
            )
        return self._write_records(last_scored_records, step=queue_step)

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
    ) -> RolloutBatch:
        materialized_path = self.materialize(
            step=step,
            inference_engine=inference_engine,
        )
        sender = FileSystemRolloutSender(self.config.output_dir, self.config.transport)
        return sender.publish(materialized_path, step=step)

    def run(
        self, *, start_step: int = 0, max_steps: int | None = None
    ) -> list[RolloutBatch]:
        total_steps = max_steps or self.config.max_steps or 1
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
                f"Rollout file '{output_path}' already exists and overwrite is disabled."
            )
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.temperatures is None:
                    raise ValueError("Rollout record is missing temperatures.")
                handle.write(json.dumps(self._serialize_record(record)) + "\n")
        return output_path

    def _serialize_record(self, record: RLExample) -> dict[str, object]:
        payload: dict[str, object] = {
            self.config.data.prompt_column: record.prompt,
            self.config.data.completion_column: record.completion,
            "target_completion": record.target_completion,
            "source": record.source,
            "env_name": record.source,
            "task": self.config.reward.mode,
            "example_id": self._example_id(record),
            self.config.data.advantage_column: record.advantage,
            self.config.data.reward_column: record.reward,
            self.config.data.temperature_column: record.temperatures,
        }
        if record.input_ids is not None:
            payload["input_ids"] = record.input_ids
        if record.target_ids is not None:
            payload["target_ids"] = record.target_ids
        if record.loss_mask is not None:
            payload["loss_mask"] = record.loss_mask
        if record.inference_logprobs is not None:
            payload[self.config.data.inference_logprobs_column] = (
                record.inference_logprobs
            )
        if record.teacher_logprobs is not None:
            payload[self.config.data.teacher_logprobs_column] = record.teacher_logprobs
        if record.tools is not None:
            payload[self.config.data.tools_column] = record.tools
        if record.chat_template_kwargs is not None:
            payload[self.config.data.chat_template_kwargs_column] = (
                record.chat_template_kwargs
            )
        if record.metadata is not None:
            payload[self.config.data.metadata_column] = record.metadata
        return payload

    def _score_record(
        self,
        record: RLExample,
        scorer: RLRewardScorer,
    ) -> RLExample:
        reward = scorer.score(record)
        return RLExample(
            prompt=record.prompt,
            completion=record.completion,
            advantage=record.advantage,
            reward=reward,
            input_ids=record.input_ids,
            target_ids=record.target_ids,
            loss_mask=record.loss_mask,
            target_completion=record.target_completion,
            inference_logprobs=record.inference_logprobs,
            teacher_logprobs=record.teacher_logprobs,
            temperatures=record.temperatures,
            tools=record.tools,
            chat_template_kwargs=record.chat_template_kwargs,
            metadata=record.metadata,
            source=record.source,
        )

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
        mode = self.config.orchestrator.advantage_mode
        if mode == "passthrough":
            return records
        if mode == "reward":
            return [
                self._replace_advantage(record, record.reward)
                if record.advantage is None
                else record
                for record in records
            ]
        if mode != "group_reward":
            raise ValueError(f"Unsupported advantage mode: {mode}")
        if records and all(record.advantage is not None for record in records):
            return records

        grouped: dict[str, list[RLExample]] = {}
        for record in records:
            grouped.setdefault(self._group_key(record), []).append(record)

        updated: list[RLExample] = []
        for record in records:
            group = grouped[self._group_key(record)]
            rewards = [item.reward for item in group]
            if any(reward is None for reward in rewards):
                raise ValueError(
                    "group_reward advantages require rewards for all rollouts."
                )
            reward_values = [float(reward) for reward in rewards if reward is not None]
            mean = sum(reward_values) / len(reward_values)
            advantage = (
                float(record.reward) - mean if record.reward is not None else 0.0
            )
            if self.config.orchestrator.normalize_group_advantages:
                variance = sum((reward - mean) ** 2 for reward in reward_values) / len(
                    reward_values
                )
                std = math.sqrt(variance)
                if std > self.config.orchestrator.advantage_epsilon:
                    advantage /= std
            updated.append(self._replace_advantage(record, advantage))
        return updated

    def _should_retry_zero_advantage(self, records: list[RLExample]) -> bool:
        return not self._filter_zero_advantage_records(records)

    def _filter_zero_advantage_records(
        self, records: list[RLExample]
    ) -> list[RLExample]:
        if not self.config.orchestrator.filter_zero_advantage:
            return records
        if self.config.orchestrator.advantage_mode != "group_reward":
            return records
        epsilon = self.config.orchestrator.advantage_epsilon
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
            if not _assistant_text(record.completion).strip():
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
            self.config.orchestrator.advantage_mode != "group_reward"
            or self.config.orchestrator.rollouts_per_example <= 1
        ):
            return records

        grouped: dict[str, list[RLExample]] = {}
        for record in records:
            grouped.setdefault(self._group_key(record), []).append(record)

        expected = self.config.orchestrator.rollouts_per_example
        complete_group_keys = {
            key for key, group in grouped.items() if len(group) >= expected
        }
        dropped = sum(
            len(group)
            for key, group in grouped.items()
            if key not in complete_group_keys
        )
        if dropped:
            logger.warning(
                "Dropped %s rollout(s) from incomplete native group(s) before "
                "group-reward scoring.",
                dropped,
            )
        return [
            record
            for record in records
            if self._group_key(record) in complete_group_keys
        ]

    def _replace_advantage(
        self,
        record: RLExample,
        advantage: float | None,
    ) -> RLExample:
        return replace(record, advantage=advantage)

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

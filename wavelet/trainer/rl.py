from __future__ import annotations

import contextlib
import json
import logging
import random
import sys
from math import ceil
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import (
    PackedRLDataset,
    RLDataset,
    _normalize_rl_record,
    _pretokenized_sample,
    count_nonempty_jsonl_rows,
    setup_rl_dataloader,
    setup_rl_dataset,
)
from wavelet.data.sft import Example, build_sample
from wavelet.orchestrator.schedule import (
    chunks_per_step as _chunks_per_step,
)
from wavelet.orchestrator.schedule import required_policy_step
from wavelet.orchestrator.schedule import (
    target_steps as _target_steps,
)
from wavelet.trainer.ckpt import TrainerState
from wavelet.trainer.distributed import barrier
from wavelet.trainer.losses import (
    component_normalization_unit_counts,
    compute_entropy,
    compute_loss,
    normalization_unit_count,
    selective_log_softmax,
    setup_rl_loss_fn,
)
from wavelet.trainer.model import is_fsdp_model, sync_hf_tp_lora_replicated_grads
from wavelet.trainer.moe import moe_load_balance_metrics
from wavelet.trainer.perf import training_flop_metrics
from wavelet.trainer.trainer import BaseTrainer
from wavelet.trainer.types import LossOutput, TrainOutput
from wavelet.transport.policy import (
    PolicyExportMixin,
)
from wavelet.transport.queue import (
    FileSystemRolloutReceiver,
    RolloutBatch,
    RolloutChunkAccumulator,
    prune_consumed_rollout_batches,
    record_rollout_claim,
    record_rollout_consumed,
    validate_rollout_manifest,
)
from wavelet.utils.config import load_config
from wavelet.utils.monitoring import emit_perf, setup_config_logger

logger = logging.getLogger(__name__)
SUM_SYNCED_METRIC_KEYS = {
    "rollout/count",
    "micro_batch/count",
    "tokens/train",
    "tokens/model",
}
ZERO_LOSS_METRIC_KEYS = (
    "mismatch_kl",
    "masked_mismatch_kl",
    "unmasked_mismatch_kl",
    "is_masked",
    "is_masked_low",
    "is_masked_high",
    "policy_loss",
    "kl_loss",
    "advantage_mean",
)
TRAIN_METRIC_ALIASES = {
    "loss": "train/loss",
    "policy_loss": "train/policy_loss",
    "kl_loss": "train/kl_loss",
    "mismatch_kl": "kl/mismatch",
    "masked_mismatch_kl": "kl/masked_mismatch",
    "unmasked_mismatch_kl": "kl/unmasked_mismatch",
    "is_masked": "dppo/is_masked",
    "is_masked_low": "dppo/is_masked_low",
    "is_masked_high": "dppo/is_masked_high",
    "advantage_mean": "advantage/token/mean",
}


def _scalar_or_json(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _packed_causal_attention_mask(
    attention_mask: Tensor,
    position_ids: Tensor,
) -> Tensor | None:
    """Build a block-causal mask for packed samples when position ids reset."""
    if attention_mask.ndim != 2 or position_ids.ndim != 2:
        return attention_mask

    valid_tokens = attention_mask.bool()
    starts = (position_ids == 0) & valid_tokens
    if not (starts.sum(dim=1) > 1).any():
        return None if valid_tokens.all() else attention_mask

    batch_size, seq_len = position_ids.shape
    segment_ids = starts.long().cumsum(dim=1) - 1
    segment_ids = torch.where(
        valid_tokens, segment_ids, torch.full_like(segment_ids, -1)
    )

    query_segments = segment_ids.unsqueeze(2)
    key_segments = segment_ids.unsqueeze(1)
    same_segment = query_segments == key_segments

    positions = torch.arange(seq_len, device=position_ids.device)
    causal = positions.view(1, seq_len, 1) >= positions.view(1, 1, seq_len)
    key_valid = valid_tokens.unsqueeze(1)
    query_valid = valid_tokens.unsqueeze(2)
    allow = same_segment & causal & key_valid

    # Fully masked padded queries can produce NaNs in attention. They do not
    # contribute to the loss, so let padded queries attend to themselves only.
    diagonal = torch.eye(
        seq_len,
        dtype=torch.bool,
        device=position_ids.device,
    ).unsqueeze(0)
    allow = torch.where(query_valid, allow, diagonal)

    mask = torch.full(
        (batch_size, seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=position_ids.device,
    )
    mask = mask.masked_fill(allow, 0.0)
    return mask.unsqueeze(1)


def _has_packed_position_resets(attention_mask: Tensor, position_ids: Tensor) -> bool:
    if attention_mask.ndim != 2 or position_ids.ndim != 2:
        return False
    valid_tokens = attention_mask.bool()
    starts = (position_ids == 0) & valid_tokens
    return bool((starts.sum(dim=1) > 1).any().item())


def _model_attn_implementation(model: object) -> str | None:
    current = model
    seen: set[int] = set()
    for _ in range(8):
        if id(current) in seen:
            break
        seen.add(id(current))
        config = getattr(current, "config", None)
        implementation = getattr(config, "_attn_implementation", None)
        if implementation is None:
            implementation = getattr(config, "attn_implementation", None)
        if implementation is not None:
            return str(implementation)
        next_model = (
            getattr(current, "module", None)
            or getattr(current, "_fsdp_wrapped_module", None)
            or getattr(current, "base_model", None)
            or getattr(current, "model", None)
        )
        if next_model is None or next_model is current:
            break
        current = next_model
    return None


def _packed_training_attention_mask(
    model: object,
    attention_mask: Tensor,
    position_ids: Tensor,
) -> Tensor | None:
    """Select the mask path for packed RL training.

    Packed examples use reset position ids as sequence boundaries. HF's
    FlashAttention 2 implementation consumes those position ids directly and
    dispatches to flash_attn_varlen_func, but only when no explicit attention
    mask is passed. Non-flash attention still needs an explicit block-causal
    mask to prevent packed samples from attending across boundaries.
    """
    if not _has_packed_position_resets(attention_mask, position_ids):
        valid_tokens = attention_mask.bool()
        return None if valid_tokens.all() else attention_mask

    implementation = _model_attn_implementation(model)
    if implementation in {
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "fa4",
    }:
        if position_ids.shape[0] != 1:
            raise ValueError(
                "Packed FlashAttention varlen training requires one packed row per "
                "micro-batch. Set data.micro_batch_size=1 or use "
                "model.attn_implementation='sdpa'."
            )
        if not attention_mask.bool().all():
            raise ValueError(
                "Packed FlashAttention varlen training requires pad-free packed rows. "
                "Set data.pad_to_multiple_of=1 and data.micro_batch_size=1 or use "
                "model.attn_implementation='sdpa'."
            )
        return None

    return _packed_causal_attention_mask(attention_mask, position_ids)


def _torch_dtype_from_name(name: str) -> torch.dtype | None:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return None


class RLTrainer(PolicyExportMixin, BaseTrainer):
    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.config = config
        self._accumulated_micro_batches = 0
        self._reward_accum: list[float] = []
        self._rollout_metric_accum: list[dict[str, float]] = []
        self._train_loss_accum: list[float] = []
        self._train_metric_accum: list[dict[str, float]] = []
        self._optimizer_batch_loss_scale: float | None = None
        self._optimizer_batch_loss_scales: dict[str, float] | None = None
        self._gradient_accumulation_loss_scale: float | None = None
        self._dynamic_loss_scale_local = 0.0
        self._loaded_micro_batch_count = 0
        self._step_compute_seconds = 0.0
        self._step_model_tokens = 0
        self._run_closed = False
        self._rl_loss_fn = setup_rl_loss_fn(config.loss)
        self._init_policy_transport()

    def _setup_data(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be set up before data")
        if self.world is None:
            raise RuntimeError("World must be set up before data")
        if self.config.data.pack_sequences:
            self.accumulation_steps = 1
        else:
            self._setup_accumulation_steps()

        data_rank, data_world_size = self._data_partition()
        self.dataset = setup_rl_dataset(
            self.tokenizer,
            self.config.data,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )
        self.dataloader = setup_rl_dataloader(
            self.dataset,
            self.config.data,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if isinstance(self.dataset, PackedRLDataset):
            self.accumulation_steps = self._packed_dataloader_batch_count()
        self._set_optimizer_batch_loss_scales(
            self._estimate_optimizer_batch_loss_scales()
        )

    def _validate_ready(self) -> None:
        super()._validate_ready()
        self._validate_reference_policy_support()

    def _after_resume(self) -> None:
        self._accumulated_micro_batches = 0
        self._dynamic_loss_scale_local = 0.0
        # StatefulDataLoader applies restored dataset state only when the next
        # iterator is created, so the dataset cursor still describes step 0 here.
        # Use measured token counts for the first resumed optimizer step; the
        # static estimate is recomputed from the moved cursor after that step.
        self._optimizer_batch_loss_scale = None
        self._optimizer_batch_loss_scales = None

    def _validate_resume_state(self, state: TrainerState) -> None:
        self._validate_progress_state(state)
        if state.step < 0 or state.micro_step < state.step:
            raise ValueError(
                "RL checkpoint step counters are invalid: expected non-negative "
                "counters and at least one micro-step per optimizer step."
            )

    def _checkpoint_dataloader(self) -> StatefulDataLoader | None:
        if self.config.orchestrator.enabled:
            return None
        return self.dataloader

    def _log_train_output(self, output: TrainOutput, progress: tqdm) -> None:
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self.step % self.config.log.log_every != 0:
            return
        metrics = output.metrics
        metrics["lr"] = self._get_lr()
        metrics["optim/lr"] = metrics["lr"]
        metrics["progress/step"] = float(self.step)
        metrics["progress/micro_step"] = float(self._micro_step)
        metrics.update(self._progress_metrics())
        self.monitor.log(metrics, self.step)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            kl=f"{metrics['mismatch_kl']:.4f}",
            lr=f"{metrics['lr']:.2e}",
        )

    def load_rollout_path(self, rollout_path: Path) -> None:
        rollout_path = Path(rollout_path)
        rollout_count = count_nonempty_jsonl_rows(
            rollout_path,
            description="Rollout batch",
        )
        optimizer_batch_size = rollout_count
        pack_sequences = self.config.data.pack_sequences
        if self.world is not None and not pack_sequences:
            _, data_world_size = self._data_partition()
            global_micro_batch = self.config.data.micro_batch_size * data_world_size
            remainder = rollout_count % global_micro_batch
            if remainder:
                optimizer_batch_size = rollout_count - remainder
                if optimizer_batch_size <= 0:
                    raise ValueError(
                        "Rollout batch contains fewer rows than one distributed "
                        "micro-batch "
                        f"({rollout_count} < {global_micro_batch})."
                    )
                logger.warning(
                    "Trimming rollout optimizer batch from %s to %s rows so it is "
                    "divisible by distributed micro-batch size %s.",
                    rollout_count,
                    optimizer_batch_size,
                    global_micro_batch,
                )
        self._maybe_log_rollout_samples(rollout_path)
        self.config = self.config.model_copy(
            update={
                "data": self.config.data.model_copy(
                    update={
                        "source": "local",
                        "path": rollout_path,
                        "batch_size": optimizer_batch_size,
                        "pack_sequences": pack_sequences,
                    }
                )
            }
        )
        self._setup_data()
        self._loaded_micro_batch_count = self._current_micro_batch_count(
            optimizer_batch_size
        )
        self._validate_reference_policy_support()

    def record_rollout_claim(
        self,
        batch: RolloutBatch,
        *,
        trainer_step_before: int,
    ) -> None:
        if not self.is_main_process():
            return
        record_rollout_claim(
            batch,
            trainer_step_before=trainer_step_before,
            events_dir=self.output_dir / "events",
        )

    def validate_rollout_batch(
        self,
        batch: RolloutBatch,
        *,
        row_count: int,
        chunk_index: int | None = None,
    ) -> None:
        """Validate queue provenance before loading a rollout batch."""
        _validate_rollout_batch(
            self.config,
            batch,
            trainer_step=self.step,
            row_count=row_count,
            chunk_index=chunk_index,
        )

    def record_rollout_consumed(
        self,
        batch: RolloutBatch,
        *,
        trainer_step_before: int,
        optimizer_step_completed: bool,
    ) -> None:
        if not self.is_main_process():
            return
        record_rollout_consumed(
            batch,
            trainer_step_before=trainer_step_before,
            trainer_step_after=self.step,
            optimizer_step_completed=optimizer_step_completed,
            events_dir=self.output_dir / "events",
        )
        if self.config.transport.cleanup_consumed:
            prune_consumed_rollout_batches(
                self.output_dir,
                self.config.transport,
                keep_last=self.config.transport.keep_last_consumed,
            )

    def rollout_events_dir(self) -> Path | None:
        if not self.is_main_process():
            return None
        return self.output_dir / "events"

    def is_main_process(self) -> bool:
        return self.world is None or self.world.is_main

    def _current_micro_batch_count(self, optimizer_batch_size: int) -> int:
        if isinstance(self.dataset, PackedRLDataset):
            return self._packed_dataloader_batch_count()
        if self.world is None:
            raise RuntimeError("World must be set up before computing micro-batches")
        _, data_world_size = self._data_partition()
        global_micro_batch = self.config.data.micro_batch_size * data_world_size
        return max(optimizer_batch_size // global_micro_batch, 1)

    def _packed_dataloader_batch_count(self) -> int:
        if not isinstance(self.dataset, PackedRLDataset):
            raise RuntimeError(  # noqa: TRY004 - invalid internal trainer state
                "Packed batch count requires a packed RL dataset."
            )
        return max(
            ceil(self.dataset.micro_batch_count() / self.config.data.micro_batch_size),
            1,
        )

    def train_loaded_rollouts_once(self) -> dict[str, float] | None:
        if self.dataloader is None:
            raise RuntimeError("Trainer dataloader is not set up.")
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self._run_closed:
            raise RuntimeError("Trainer run has already been finalized.")

        self.model.train()
        metrics: dict[str, float] | None = None
        for index, batch in enumerate(self.dataloader):
            if index >= self._loaded_micro_batch_count:
                break
            batch = self._prepare_batch(batch)
            output = self._train_step(batch)
            if output.stepped:
                metrics = output.metrics
                progress = tqdm(total=1, disable=not self.world.is_main)
                self._log_train_output(output, progress)
                self._maybe_checkpoint()
                progress.update(1)
                progress.close()
                return metrics
        return metrics

    def _maybe_log_rollout_samples(self, rollout_path: Path) -> None:
        if self.monitor is None:
            return
        sample_config = self.config.monitor.samples
        if not sample_config.enabled:
            return
        if self.step % sample_config.interval != 0:
            return

        rows = self._sample_rollout_rows(rollout_path)
        if rows:
            self.monitor.log_samples(rows, self.step)

    def _sample_rollout_rows(self, rollout_path: Path) -> list[dict[str, object]]:
        sample_config = self.config.monitor.samples
        payloads: list[dict[str, object]] = []
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payloads.append(json.loads(stripped))
        if not payloads:
            return []

        max_samples = len(payloads)
        if sample_config.sample_ratio is not None:
            if sample_config.sample_ratio <= 0.0:
                return []
            max_samples = max(1, int(len(payloads) * sample_config.sample_ratio))
        if sample_config.max_samples is not None:
            max_samples = min(max_samples, sample_config.max_samples)
        if len(payloads) > max_samples:
            rng = random.Random(self.config.seed + self.step)
            payloads = rng.sample(payloads, max_samples)

        rows = []
        for index, payload in enumerate(payloads):
            row = self._sample_log_row(payload, fallback_example_id=index)
            if row is not None:
                rows.append(row)
        return rows

    def _sample_log_row(
        self,
        payload: dict[str, object],
        *,
        fallback_example_id: int,
    ) -> dict[str, object] | None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be set up before sample logging.")

        record = _normalize_rl_record(payload, self.config.data)
        tokenized = _pretokenized_sample(record, self.config.data.seq_len)
        if tokenized is None:
            tokenized = build_sample(
                Example(
                    prompt=record.prompt,
                    completion=record.completion,
                    tools=record.tools,
                    chat_template_kwargs=record.chat_template_kwargs,
                    source=record.source,
                ),
                self.tokenizer,
                seq_len=self.config.data.seq_len,
                loss_mask_config=self.config.data.loss_mask,
            )
        if tokenized is None:
            return None

        input_ids = [
            *tokenized["input_ids"],
            tokenized["target_ids"][-1],
        ]
        return {
            "env_name": str(
                payload.get(
                    "env_name",
                    payload.get("source", payload.get("__source", "")),
                )
            ),
            "task": str(payload.get("task", "")),
            "example_id": str(payload.get("example_id", fallback_example_id)),
            "prompt": self._messages_text(record.prompt),
            "completion": self._messages_text(record.completion),
            "messages": self.tokenizer.decode(input_ids),
            "input_ids": json.dumps(input_ids),
            "reward": _scalar_or_json(payload.get(self.config.data.reward_column)),
        }

    def _messages_text(self, messages: list[dict[str, str]]) -> str:
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        except Exception:  # noqa: BLE001 - sample logging has a JSON fallback
            return json.dumps(messages, ensure_ascii=False)

    def finalize(self, *, status: str = "completed") -> None:
        self._close_policy_transport()
        self._close_step_profiler()
        self._close_memory_profiler()
        if self.monitor is None or self._run_closed:
            self._close_garbage_collector()
            return
        try:
            if status == "completed" and self.ckpt_manager is not None:
                self._save_final_checkpoint()
                self.ckpt_manager.wait_for_pending_save()
            self.monitor.finish(status=status, step=self.step)
            self._run_closed = True
            if status == "completed" and not self._uses_sleep_colocation():
                self._save_model()
        finally:
            self._close_garbage_collector()

    def _validate_reference_policy_support(self) -> None:
        if self.config.orchestrator.enabled:
            return
        if not isinstance(self.dataset, RLDataset):
            return
        if self.model is None:
            raise RuntimeError("Model must be set up before validating RL data")
        from wavelet.trainer.model import unwrap_model

        missing_references = any(
            record.inference_logprobs is None
            and (
                record.advantage is not None
                or record.reward is not None
                or record.ref_kl_weight is not None
            )
            for record in self.dataset.records
        )
        if not missing_references:
            return
        model = unwrap_model(self.model)
        if not callable(getattr(model, "disable_adapter", None)):
            raise ValueError(  # noqa: TRY004 - incompatible dataset/model config
                "RL data is missing inference_logprobs for at least one sample, but the "
                "current model cannot derive a reference policy by disabling adapters. "
                "Use LoRA or provide inference_logprobs in the dataset."
            )

    def _setup_accumulation_steps(self) -> None:
        if self.world is None:
            raise RuntimeError("World must be set up before accumulation steps")
        _, data_world_size = self._data_partition()
        global_micro_batch = self.config.data.micro_batch_size * data_world_size
        if self.config.data.batch_size % global_micro_batch != 0:
            raise ValueError(
                "RL data.batch_size is the global optimizer batch size and must be "
                "divisible by data.micro_batch_size * data_parallel_world_size "
                f"({self.config.data.micro_batch_size} * {data_world_size})."
            )
        self.accumulation_steps = self.config.data.batch_size // global_micro_batch

    def _estimate_optimizer_batch_loss_scale(self) -> float | None:
        scales = self._estimate_optimizer_batch_loss_scales()
        return None if scales is None else scales["rl"]

    def _estimate_optimizer_batch_loss_scales(self) -> dict[str, float] | None:
        if self.config.data.num_workers != 0:
            return None
        if not isinstance(self.dataset, (RLDataset, PackedRLDataset)):
            return None
        local_optimizer_batch_size = (
            self.accumulation_steps * self.config.data.micro_batch_size
        )
        local_loss_scales = self.dataset.loss_scales_for_next_local_batch(
            local_optimizer_batch_size,
            rl_normalization=self.config.loss.normalization,
        )
        return self._average_data_parallel_loss_scales(local_loss_scales)

    def _set_optimizer_batch_loss_scales(
        self,
        scales: dict[str, float] | None,
    ) -> None:
        self._optimizer_batch_loss_scales = scales
        self._optimizer_batch_loss_scale = None if scales is None else scales["rl"]

    def _average_data_parallel_loss_scale(self, local_loss_scale: float) -> float:
        """Return the denominator compatible with averaged DP gradients."""
        return self._average_data_parallel_loss_scales(
            {"rl": local_loss_scale, "ce": 0.0, "ref_kl": 0.0}
        )["rl"]

    def _average_data_parallel_loss_scales(
        self,
        local_loss_scales: dict[str, float | int],
    ) -> dict[str, float]:
        """Return per-component denominators for averaged DP gradients."""
        components = ("rl", "ce", "ref_kl")
        _, data_world_size = self._data_partition()
        cp_world_size = self.parallel_dims.cp if self.parallel_dims is not None else 1
        normalization_world_size = data_world_size * cp_world_size
        if normalization_world_size == 1:
            return {
                component: max(float(local_loss_scales[component]), 1.0)
                for component in components
            }
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed RL loss normalization requires an initialized "
                "process group."
            )
        if self.world is None or self.parallel_dims is None:
            raise RuntimeError(
                "Distributed world and parallel dimensions must be initialized "
                "before RL loss normalization."
            )

        loss_scales = torch.tensor(
            [float(local_loss_scales[component]) for component in components],
            dtype=torch.float64,
            device=self.world.device,
        )
        reduction_mesh = "dp_cp" if cp_world_size > 1 else "dp"
        dp_group = self.parallel_dims.get_mesh(reduction_mesh).get_group()
        torch.distributed.all_reduce(
            loss_scales,
            op=torch.distributed.ReduceOp.SUM,
            group=dp_group,
        )
        # DDP and FSDP average gradients across the same data-parallel ranks.
        # Dividing each local loss by the average token count therefore yields
        # a global sum divided by the global token count after gradient sync.
        return {
            component: max(float(scale) / normalization_world_size, 1.0)
            for component, scale in zip(
                components,
                loss_scales.tolist(),
                strict=True,
            )
        }

    def _train_step(self, batch: dict[str, Tensor]) -> TrainOutput:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer not set up")
        micro_step_started_at = perf_counter()
        if self._step_model_tokens == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._step_model_tokens += int(batch["input_ids"].numel())

        attention_mask = _packed_training_attention_mask(
            self.model,
            batch["attention_mask"],
            batch["position_ids"],
        )
        if attention_mask is not None and attention_mask.is_floating_point():
            mask_dtype = _torch_dtype_from_name(self.config.model.torch_dtype)
            if mask_dtype is not None:
                attention_mask = attention_mask.to(dtype=mask_dtype)

        rl_weights, ce_weights, ref_kl_weights = self._component_weights(batch)
        has_auxiliary_component = bool(
            (ce_weights != 0).any() or (ref_kl_weights != 0).any()
        )
        if has_auxiliary_component and self._optimizer_batch_loss_scales is None:
            local_scales = component_normalization_unit_counts(
                batch["loss_mask"],
                rl_weights=rl_weights,
                ce_weights=ce_weights,
                ref_kl_weights=ref_kl_weights,
                rl_normalization=self.config.loss.normalization,
                position_ids=batch["position_ids"],
            )
            remaining_local_samples = (
                self.accumulation_steps - 1
            ) * self.config.data.micro_batch_size
            if remaining_local_samples > 0:
                if self.config.data.num_workers != 0 or not isinstance(
                    self.dataset, (RLDataset, PackedRLDataset)
                ):
                    raise ValueError(
                        "CE/ref-KL loss components with data.num_workers=1 require "
                        "data.batch_size to equal the global micro-batch size."
                    )
                remaining_scales = self.dataset.loss_scales_for_next_local_batch(
                    remaining_local_samples,
                    rl_normalization=self.config.loss.normalization,
                )
                for component in local_scales:
                    local_scales[component] += int(remaining_scales[component])
            estimated_scales = self._average_data_parallel_loss_scales(local_scales)
            self._set_optimizer_batch_loss_scales(estimated_scales)

        extra_buffers = (
            [(attention_mask, 2)]
            if attention_mask is not None and attention_mask.ndim == 4
            else None
        )
        with self._context_parallel_batch(
            batch,
            extra_buffers=extra_buffers,
        ), self.act_offload_ctx:
            loss_output = self._forward_rl_loss(batch, attention_mask)
            self._require_finite_loss(loss_output.loss, label="RL loss")
            if self._optimizer_batch_loss_scale is None:
                self._dynamic_loss_scale_local += normalization_unit_count(
                    batch["loss_mask"],
                    normalization=self.config.loss.normalization,
                    position_ids=batch["position_ids"],
                )
            self._backward_rl_loss(loss_output.loss)

        self._record_micro_batch_metrics(batch, loss_output)

        self._micro_step += 1
        self._accumulated_micro_batches += 1
        if self._accumulated_micro_batches < self.accumulation_steps:
            self._step_compute_seconds += perf_counter() - micro_step_started_at
            return TrainOutput(
                loss=loss_output,
                stepped=False,
                step=self.step,
                micro_step=self._micro_step,
            )

        grad_norm = self._apply_optimizer_step()
        metrics = self._finalize_optimizer_metrics(grad_norm)
        self._step_compute_seconds += perf_counter() - micro_step_started_at
        metrics.update(self._finish_step_performance_metrics())
        return TrainOutput(
            loss=loss_output,
            stepped=True,
            step=self.step,
            micro_step=self._micro_step,
            metrics=metrics,
        )

    def _finish_step_performance_metrics(self) -> dict[str, float]:
        elapsed = max(self._step_compute_seconds, 1e-9)
        global_tokens = self._step_model_tokens * self._data_parallel_world_size()
        peak_memory_gib = (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        compute_dtype = self._model_compute_dtype
        if (
            self.world is not None
            and self.world.device.type == "cuda"
            and self.config.model.torch_dtype == "float32"
        ):
            compute_dtype = torch.bfloat16
        metrics = {
            "perf/tokens_per_second": global_tokens / elapsed,
            "perf/peak_memory_gib": peak_memory_gib,
        }
        metrics.update(
            training_flop_metrics(
                flops_per_token=self._model_flops_per_token,
                model_tokens=global_tokens,
                elapsed_seconds=elapsed,
                world_size=self.world.world_size if self.world is not None else 1,
                dtype=compute_dtype,
            )
        )
        self._step_compute_seconds = 0.0
        self._step_model_tokens = 0
        return metrics

    def _forward_rl_loss(
        self,
        batch: dict[str, Tensor],
        attention_mask: Tensor | None,
    ) -> LossOutput:
        trainer_logprobs, entropy, moe_metrics = self._model_logprobs_and_entropy(
            batch,
            attention_mask,
        )
        if not batch["loss_mask"].bool().any():
            loss = trainer_logprobs.sum() * 0.0
            metrics = self._zero_loss_metrics(loss)
            metrics.update(moe_metrics)
            return LossOutput(loss=loss, metrics=metrics)
        rl_weights, ce_weights, ref_kl_weights = self._component_weights(batch)
        output = compute_loss(
            trainer_logprobs,
            self._inference_logprobs(batch, attention_mask),
            self._teacher_logprobs(batch),
            batch["advantages"],
            batch["loss_mask"],
            self.config.loss,
            loss_scale=(
                self._optimizer_batch_loss_scale
                if self._optimizer_batch_loss_scale is not None
                else 1.0
            ),
            position_ids=batch["position_ids"],
            rl_weights=rl_weights,
            ce_weights=ce_weights,
            ref_kl_weights=ref_kl_weights,
            component_loss_scales=self._optimizer_batch_loss_scales,
            rl_loss_fn=self._rl_loss_fn,
        )
        output.metrics.update(self._entropy_metrics(entropy, batch["loss_mask"]))
        output.metrics.update(moe_metrics)
        return output

    @staticmethod
    def _component_weights(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        loss_mask = batch["loss_mask"]
        return (
            batch.get("rl_weights", loss_mask.to(dtype=torch.float32)),
            batch.get("ce_weights", torch.zeros_like(loss_mask, dtype=torch.float32)),
            batch.get(
                "ref_kl_weights",
                torch.zeros_like(loss_mask, dtype=torch.float32),
            ),
        )

    def _backward_rl_loss(self, loss: Tensor) -> None:
        with self._maybe_no_sync():
            loss.backward()

    def _record_micro_batch_metrics(
        self,
        batch: dict[str, Tensor],
        loss_output: LossOutput,
    ) -> None:
        reward_mean = self._reward_mean(
            batch["rewards"],
            sample_counts=batch.get("sample_counts"),
        )
        if reward_mean is not None:
            self._reward_accum.append(reward_mean)
        self._rollout_metric_accum.append(self._batch_rollout_metrics(batch))
        self._train_loss_accum.append(float(loss_output.loss.detach().item()))
        self._train_metric_accum.append(
            {
                key: float(value.detach().item())
                for key, value in loss_output.metrics.items()
            }
        )

    def _apply_optimizer_step(self) -> float | None:
        if self._optimizer_batch_loss_scale is None:
            self._gradient_accumulation_loss_scale = (
                self._average_data_parallel_loss_scale(
                    self._dynamic_loss_scale_local,
                )
            )
        self._apply_gradient_accumulation_loss_scale()
        self._sync_tensor_parallel_lora_grads()
        grad_norm = self._clip_grad_norm() if self.config.max_grad_norm > 0 else None
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self._accumulated_micro_batches = 0
        self._dynamic_loss_scale_local = 0.0
        return grad_norm

    def _finalize_optimizer_metrics(self, grad_norm: float | None) -> dict[str, float]:
        if self._gradient_accumulation_loss_scale is not None:
            logged_loss = sum(self._train_loss_accum) / max(
                self._gradient_accumulation_loss_scale,
                1.0,
            )
        elif self._optimizer_batch_loss_scale is None:
            logged_loss = sum(self._train_loss_accum) / len(self._train_loss_accum)
        else:
            logged_loss = sum(self._train_loss_accum)
        metrics = {"loss": logged_loss}
        metrics.update(self._aggregate_train_metrics(self._train_metric_accum))
        self._train_loss_accum.clear()
        self._train_metric_accum.clear()
        if self._reward_accum:
            metrics["reward_mean"] = sum(self._reward_accum) / len(self._reward_accum)
            self._reward_accum.clear()
        if self._rollout_metric_accum:
            metrics.update(self._aggregate_rollout_metrics(self._rollout_metric_accum))
            self._rollout_metric_accum.clear()
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm
        metrics.update(self._standard_metric_aliases(metrics))
        metrics = self._sync_metrics(metrics)
        metrics = self._finalize_synced_metrics(metrics)
        self._set_optimizer_batch_loss_scales(
            self._estimate_optimizer_batch_loss_scales()
        )
        self._gradient_accumulation_loss_scale = None
        return metrics

    def _sync_tensor_parallel_lora_grads(self) -> None:
        if self.model is None:
            raise RuntimeError("Trainer not set up")
        sync_hf_tp_lora_replicated_grads(self.model, self.parallel_dims)

    def _apply_gradient_accumulation_loss_scale(self) -> None:
        if self._gradient_accumulation_loss_scale is None:
            return
        if self.model is None:
            raise RuntimeError("Trainer not set up")
        scale = max(float(self._gradient_accumulation_loss_scale), 1.0)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)

    def _zero_loss_metrics(self, loss: Tensor) -> dict[str, Tensor]:
        zero = loss.detach() * 0.0
        return {key: zero for key in ZERO_LOSS_METRIC_KEYS}

    def _inference_logprobs(
        self, batch: dict[str, Tensor], attention_mask: Tensor | None
    ) -> Tensor:
        inference_logprobs = batch["inference_logprobs"].clone()
        has_inference = batch["has_inference_logprobs"].bool()
        rl_weights, _, ref_kl_weights = self._component_weights(batch)
        needs_inference = (rl_weights != 0).any(dim=1) | (ref_kl_weights != 0).any(
            dim=1
        )
        missing_required = ~has_inference & needs_inference
        if not missing_required.any():
            inference_logprobs[~has_inference] = 0.0
            return inference_logprobs

        from wavelet.trainer.model import unwrap_model

        model = unwrap_model(self.model)
        disable_adapter = getattr(model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(  # noqa: TRY004 - invalid initialized model state
                "inference_logprobs are required when the model cannot disable adapters."
            )

        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), disable_adapter():
                computed_inference = self._model_logprobs(batch, attention_mask)
        finally:
            if training:
                self.model.train()
        inference_logprobs[missing_required] = computed_inference[missing_required]
        return inference_logprobs

    def _model_logprobs(
        self,
        batch: dict[str, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        logprobs, _, _ = self._model_logprobs_and_entropy(
            batch,
            attention_mask,
            include_entropy=False,
        )
        return logprobs

    def _model_logprobs_and_entropy(
        self,
        batch: dict[str, Tensor],
        attention_mask: Tensor | None,
        *,
        include_entropy: bool = True,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        if self.model is None:
            raise RuntimeError("Model not set up")
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": batch["position_ids"],
        }
        if self.config.model.fused_lm_head_token_chunk_size != "disabled":
            model_kwargs["labels"] = batch["target_ids"]
            model_kwargs["temperature"] = batch["temperatures"]

        with self._model_forward_context():
            outputs = self.model(**model_kwargs)
        moe_metrics = moe_load_balance_metrics(
            self.model,
            outputs,
            token_mask=attention_mask,
        )
        if isinstance(outputs, dict) and outputs.get("logprobs") is not None:
            entropy = outputs.get("entropy") if include_entropy else None
            return (
                outputs["logprobs"].float().contiguous(),
                entropy.float().contiguous() if entropy is not None else None,
                moe_metrics,
            )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        scaled_logits = logits.float() / batch["temperatures"].float().unsqueeze(-1)
        entropy = compute_entropy(scaled_logits) if include_entropy else None
        return (
            selective_log_softmax(scaled_logits, batch["target_ids"]),
            entropy,
            moe_metrics,
        )

    def _entropy_metrics(
        self,
        entropy: Tensor | None,
        loss_mask: Tensor,
    ) -> dict[str, Tensor]:
        if entropy is None:
            return {}
        values = entropy.detach()[loss_mask.bool()]
        if values.numel() == 0:
            return {}
        return {
            "_entropy_sum": values.sum(),
            "_entropy_count": values.new_tensor(values.numel()),
            "entropy/mean": values.mean(),
            "entropy/min": values.min(),
            "entropy/max": values.max(),
        }

    def _model_forward_context(self):
        if (
            self.world is not None
            and self.world.device.type == "cuda"
            and self.config.model.torch_dtype == "float32"
        ):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _teacher_logprobs(self, batch: dict[str, Tensor]) -> Tensor | None:
        has_teacher = batch["has_teacher_logprobs"].bool()
        _, _, ref_kl_weights = self._component_weights(batch)
        needs_teacher = (ref_kl_weights != 0).any(dim=1)
        if (~has_teacher & needs_teacher).any():
            raise ValueError(
                "teacher_logprobs are required for samples with nonzero ref_kl_weight."
            )
        if not has_teacher.any():
            return None
        if self.config.loss.teacher_tau > 0.0 and not has_teacher.all():
            raise ValueError(
                "teacher_logprobs must be provided for all samples when "
                "loss.teacher_tau is nonzero."
            )
        return batch["teacher_logprobs"]

    def _reward_mean(
        self,
        rewards: Tensor,
        *,
        sample_counts: Tensor | None = None,
    ) -> float | None:
        valid = ~torch.isnan(rewards)
        if not valid.any():
            return None
        if sample_counts is not None:
            weights = sample_counts[valid].float().clamp_min(0)
            if weights.sum() > 0:
                return float(
                    (rewards[valid].float() * weights).sum().item()
                    / weights.sum().item()
                )
        return float(rewards[valid].mean().item())

    def _batch_rollout_metrics(self, batch: dict[str, Tensor]) -> dict[str, float]:
        loss_mask = batch["loss_mask"].bool()
        seq_lens = loss_mask.sum(dim=1).float()
        sample_counts = batch.get("sample_counts")
        rollout_count = (
            float(sample_counts.sum().item())
            if sample_counts is not None
            else float(batch["input_ids"].shape[0])
        )
        metrics: dict[str, float] = {
            "rollout/count": rollout_count,
            "micro_batch/count": float(batch["input_ids"].shape[0]),
            "tokens/train": float(loss_mask.sum().item()),
            "tokens/model": float(batch["input_ids"].numel()),
            "seq_len/all/mean": float(seq_lens.mean().item()),
            "seq_len/all/max": float(seq_lens.max().item()),
            "seq_len/all/min": float(seq_lens.min().item()),
        }

        rewards = batch["rewards"]
        valid_rewards = rewards[~torch.isnan(rewards)]
        reward_mean = self._reward_mean(rewards, sample_counts=sample_counts)
        if reward_mean is not None:
            metrics.update(
                {
                    "reward/all/mean": reward_mean,
                    "reward/all/max": float(valid_rewards.max().item()),
                    "reward/all/min": float(valid_rewards.min().item()),
                }
            )

        advantages = batch["advantages"][loss_mask]
        if advantages.numel() > 0:
            metrics.update(
                {
                    "_advantage_sum": float(advantages.sum().item()),
                    "_advantage_sumsq": float(advantages.square().sum().item()),
                    "_advantage_count": float(advantages.numel()),
                    "advantage/all/mean": float(advantages.mean().item()),
                    "advantage/all/max": float(advantages.max().item()),
                    "advantage/all/min": float(advantages.min().item()),
                    "advantage/all/std": float(advantages.std(unbiased=False).item()),
                }
            )
        return metrics

    def _aggregate_rollout_metrics(
        self,
        micro_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        if not micro_metrics:
            return {}
        aggregated: dict[str, float] = {}
        all_keys = set().union(*(metrics.keys() for metrics in micro_metrics))
        for key in all_keys:
            if key.startswith("_"):
                continue
            values = [metrics[key] for metrics in micro_metrics if key in metrics]
            if not values:
                continue
            aggregated[key] = self._aggregate_rollout_metric_value(
                key,
                values,
                micro_metrics,
                aggregated,
            )
        self._add_aggregate_advantage_stats(aggregated, micro_metrics)
        return aggregated

    @staticmethod
    def _aggregate_rollout_metric_value(
        key: str,
        values: list[float],
        micro_metrics: list[dict[str, float]],
        aggregated: dict[str, float],
    ) -> float:
        if key.endswith("/max"):
            return max(values)
        if key.endswith("/min"):
            return min(values)
        if key == "reward/all/mean":
            weighted_sum = sum(
                metrics[key] * metrics.get("rollout/count", 1.0)
                for metrics in micro_metrics
                if key in metrics
            )
            total_weight = sum(
                metrics.get("rollout/count", 1.0)
                for metrics in micro_metrics
                if key in metrics
            )
            aggregated["_reward_weighted_sum"] = weighted_sum
            aggregated["_reward_weight"] = total_weight
            return (
                weighted_sum / total_weight
                if total_weight > 0
                else sum(values) / len(values)
            )
        if key in {
            "tokens/train",
            "tokens/model",
            "rollout/count",
            "micro_batch/count",
        }:
            return sum(values)
        return sum(values) / len(values)

    @staticmethod
    def _add_aggregate_advantage_stats(
        aggregated: dict[str, float],
        micro_metrics: list[dict[str, float]],
    ) -> None:
        count = sum(metrics.get("_advantage_count", 0.0) for metrics in micro_metrics)
        if count <= 0:
            return
        total = sum(metrics.get("_advantage_sum", 0.0) for metrics in micro_metrics)
        total_squares = sum(
            metrics.get("_advantage_sumsq", 0.0) for metrics in micro_metrics
        )
        mean = total / count
        variance = max(total_squares / count - mean**2, 0.0)
        aggregated["advantage/all/mean"] = mean
        aggregated["advantage/all/std"] = variance**0.5

    def _aggregate_train_metrics(
        self,
        micro_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        if not micro_metrics:
            return {}
        aggregated: dict[str, float] = {}
        all_keys = set().union(*(metrics.keys() for metrics in micro_metrics))
        for key in all_keys:
            values = [metrics[key] for metrics in micro_metrics if key in metrics]
            if values:
                if key in {"_entropy_sum", "_entropy_count"}:
                    aggregated[key] = sum(values)
                elif key == "entropy/min":
                    aggregated[key] = min(values)
                elif key == "entropy/max":
                    aggregated[key] = max(values)
                else:
                    aggregated[key] = sum(values) / len(values)
        return aggregated

    def _standard_metric_aliases(self, metrics: dict[str, float]) -> dict[str, float]:
        aliases = {
            alias: metrics[key]
            for key, alias in TRAIN_METRIC_ALIASES.items()
            if key in metrics
        }
        if "reward_mean" in metrics:
            aliases.setdefault("reward/all/mean", metrics["reward_mean"])
        return aliases

    def _maybe_no_sync(self) -> contextlib.AbstractContextManager[None]:
        if self.world is None or self.world.world_size <= 1:
            return contextlib.nullcontext()
        will_step = self._accumulated_micro_batches + 1 >= self.accumulation_steps
        if will_step:
            return contextlib.nullcontext()
        no_sync = getattr(self.model, "no_sync", None)
        if not callable(no_sync):
            return contextlib.nullcontext()
        return no_sync()

    def _sync_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if (
            self.world is None
            or self.world.world_size <= 1
            or not torch.distributed.is_initialized()
        ):
            return metrics

        rank = torch.distributed.get_rank()
        gathered = self._gather_metric_payloads(
            {key: float(value) for key, value in metrics.items()}
            if self._is_data_parallel_metric_leader()
            else {},
            rank=rank,
        )
        synced = self._reduce_gathered_metrics(gathered) if rank == 0 else {}
        return self._broadcast_synced_metrics(synced, rank=rank)

    def _gather_metric_payloads(
        self,
        metric_payload: dict[str, float],
        *,
        rank: int,
    ) -> list[dict[str, float] | None] | None:
        if self.world is None:
            raise RuntimeError("World must be set up before metric synchronization")

        if rank == 0:
            gathered = [None for _ in range(self.world.world_size)]
        else:
            gathered = None
        torch.distributed.gather_object(
            metric_payload,
            gathered,
            dst=0,
        )
        return gathered

    def _reduce_gathered_metrics(
        self,
        gathered: list[dict[str, float] | None] | None,
    ) -> dict[str, float]:
        if gathered is None:
            raise RuntimeError("Metric synchronization returned no gathered metrics.")
        synced: dict[str, float] = {}
        contributors = [item for item in gathered if item]
        keys = sorted({key for item in contributors for key in item})
        for key in keys:
            values = [float(item[key]) for item in contributors if key in item]
            if values:
                synced[key] = self._reduce_metric_values(key, values)
        return synced

    def _reduce_metric_values(self, key: str, values: list[float]) -> float:
        if key.endswith("/max"):
            return max(values)
        if key.endswith("/min"):
            return min(values)
        if key.startswith("_") or key in SUM_SYNCED_METRIC_KEYS:
            return sum(values)
        return sum(values) / len(values)

    def _broadcast_synced_metrics(
        self,
        synced: dict[str, float],
        *,
        rank: int,
    ) -> dict[str, float]:
        synced_payload: list[dict[str, float] | None] = [synced if rank == 0 else None]
        torch.distributed.broadcast_object_list(synced_payload, src=0)
        if synced_payload[0] is None:
            raise RuntimeError("Metric synchronization broadcast no metrics.")
        return synced_payload[0]

    def _finalize_synced_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        reward_sum = metrics.pop("_reward_weighted_sum", None)
        reward_weight = metrics.pop("_reward_weight", None)
        if reward_sum is not None and reward_weight is not None and reward_weight > 0:
            metrics["reward/all/mean"] = reward_sum / reward_weight
            metrics["reward_mean"] = metrics["reward/all/mean"]
        entropy_sum = metrics.pop("_entropy_sum", None)
        entropy_count = metrics.pop("_entropy_count", None)
        if entropy_sum is not None and entropy_count is not None and entropy_count > 0:
            metrics["entropy/mean"] = entropy_sum / entropy_count
        return metrics

    def _save_model(self) -> None:
        if not self.world:
            return
        if self.model is None or self.tokenizer is None:
            return

        from wavelet.trainer.model import save_lora_adapter_snapshot_from_fsdp

        if (
            self.config.lora is not None
            and self.config.policy_transfer.lightweight_lora
            and is_fsdp_model(self.model)
        ):
            saved_path = save_lora_adapter_snapshot_from_fsdp(
                self.model,
                self.output_dir,
                is_main_process=self.world.is_main,
                parallel_dims=self.parallel_dims,
            )
            if self.world.is_main:
                self.tokenizer.save_pretrained(saved_path)
                logger.info(f"Model saved to {self.output_dir}")
            return
        super()._save_model()


_StreamingChunkAccumulator = RolloutChunkAccumulator


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    config = load_config(RLConfig, argv)
    setup_config_logger("rl_trainer", config)
    trainer = RLTrainer(config)
    try:
        trainer.setup()
        if config.orchestrator.enabled:
            trainer.export_policy(
                step=trainer.step,
                force=trainer.resume_checkpoint_dir is not None,
            )
            trainer.offload_after_refit()
            try:
                target_step = _target_steps(config)
                if _use_streaming_rollout_chunks(config):
                    receiver = FileSystemRolloutReceiver(
                        config.output_dir,
                        config.transport,
                        start_step=trainer.step * _chunks_per_step(config),
                        events_dir=trainer.rollout_events_dir(),
                    )
                    _run_streaming_rollout_training(
                        config,
                        trainer,
                        receiver,
                        target_step=target_step,
                    )
                    trainer.finalize(status="completed")
                    return 0
                receiver = FileSystemRolloutReceiver(
                    config.output_dir,
                    config.transport,
                    start_step=trainer.step,
                    events_dir=trainer.rollout_events_dir(),
                )
                while trainer.step < target_step:
                    loop_started_at = perf_counter()
                    wait_started_at = perf_counter()
                    batch = receiver.wait()
                    wait_seconds = perf_counter() - wait_started_at
                    trainer_step_before = trainer.step
                    row_count = count_nonempty_jsonl_rows(
                        batch.path,
                        description="Rollout batch",
                    )
                    trainer.validate_rollout_batch(
                        batch,
                        row_count=row_count,
                    )
                    trainer.record_rollout_claim(
                        batch,
                        trainer_step_before=trainer_step_before,
                    )
                    load_started_at = perf_counter()
                    trainer.load_rollout_path(batch.path)
                    load_seconds = perf_counter() - load_started_at
                    train_started_at = perf_counter()
                    trainer.prepare_for_training()
                    trainer.train_until(trainer.step + 1)
                    train_seconds = perf_counter() - train_started_at
                    trainer.record_rollout_consumed(
                        batch,
                        trainer_step_before=trainer_step_before,
                        optimizer_step_completed=True,
                    )
                    export_started_at = perf_counter()
                    trainer.export_policy(step=trainer.step)
                    trainer.offload_after_refit()
                    export_seconds = perf_counter() - export_started_at
                    total_seconds = perf_counter() - loop_started_at
                    emit_perf(
                        "trainer_step",
                        step=trainer.step,
                        wait_batch=wait_seconds,
                        load_rollout=load_seconds,
                        train=train_seconds,
                        export_policy=export_seconds,
                        total=total_seconds,
                    )
            except Exception:
                trainer.finalize(status="failed")
                raise
            trainer.finalize(status="completed")
        else:
            trainer.train()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def _use_streaming_rollout_chunks(config: RLConfig) -> bool:
    from wavelet.orchestrator.scheduler import PublishMode, resolve_rollout_schedule

    return resolve_rollout_schedule(config).publish_mode is PublishMode.STREAMING and (
        config.orchestrator.examples_per_step is not None
        or config.orchestrator.token_batch_size is not None
    )


def _run_streaming_rollout_training(
    config: RLConfig,
    trainer: RLTrainer,
    receiver: FileSystemRolloutReceiver,
    *,
    target_step: int,
) -> None:
    examples_per_step = config.orchestrator.examples_per_step
    token_batch_size = config.orchestrator.token_batch_size
    if examples_per_step is None and token_batch_size is None:
        raise ValueError(
            "Either orchestrator.examples_per_step or "
            "orchestrator.token_batch_size is required."
        )
    chunks_per_step = _chunks_per_step(config)
    min_loadable_rows = _min_loadable_rollout_rows(config, trainer)
    accumulator = _StreamingChunkAccumulator()

    while trainer.step < target_step:
        loop_started_at = perf_counter()
        wait_started_at = perf_counter()
        batch = receiver.wait()
        wait_seconds = perf_counter() - wait_started_at
        trainer_step_before = trainer.step
        row_count = count_nonempty_jsonl_rows(
            batch.path,
            description="Rollout chunk",
        )
        _validate_streaming_rollout_batch(
            config,
            batch,
            trainer_step=trainer.step,
            row_count=row_count,
        )
        trainer.record_rollout_claim(
            batch,
            trainer_step_before=trainer_step_before,
        )
        accumulator.buffer(batch, row_count)
        if not accumulator.should_load(min_rows=min_loadable_rows):
            emit_perf(
                "trainer_chunk_buffered",
                queue_step=batch.step,
                trainer_step=trainer.step,
                rows=row_count,
                pending_rows=accumulator.pending_rows,
                min_loadable_rows=min_loadable_rows,
                wait_batch=wait_seconds,
            )
            continue

        load_started_at = perf_counter()
        rollout_path = _combined_rollout_path(
            config,
            trainer=trainer,
            paths=accumulator.pending_paths,
            chunk_index=accumulator.chunk_index,
            min_rows=min_loadable_rows,
        )
        row_count = count_nonempty_jsonl_rows(
            rollout_path,
            description="Rollout chunk",
        )
        _, loaded_batches, loaded_chunks = accumulator.drain_pending_batches()
        trainer.load_rollout_path(rollout_path)
        accumulator.mark_loaded(
            rows=row_count,
            chunks=loaded_chunks,
            loss_scale=trainer._optimizer_batch_loss_scale,
        )
        _configure_streaming_accumulation(
            trainer,
            accumulator,
            chunks_per_step=chunks_per_step,
        )
        load_seconds = perf_counter() - load_started_at
        train_started_at = perf_counter()
        trainer.prepare_for_training()
        metrics = trainer.train_loaded_rollouts_once()
        _validate_distributed_step_sync(trainer, metrics is not None)
        train_seconds = perf_counter() - train_started_at
        for loaded_batch in loaded_batches:
            trainer.record_rollout_consumed(
                loaded_batch,
                trainer_step_before=trainer_step_before,
                optimizer_step_completed=metrics is not None,
            )
        _remove_combined_rollout_path(config, trainer=trainer, path=rollout_path)

        export_seconds = 0.0
        if metrics is not None:
            _log_step_perf_metrics(
                trainer,
                metrics,
                train_seconds=train_seconds,
                loop_seconds=perf_counter() - loop_started_at,
            )
            accumulator.reset_after_optimizer_step()
            export_started_at = perf_counter()
            trainer.export_policy(step=trainer.step)
            trainer.offload_after_refit()
            export_seconds = perf_counter() - export_started_at
        total_seconds = perf_counter() - loop_started_at
        emit_perf(
            "trainer_chunk",
            queue_step=batch.step,
            trainer_step=trainer.step,
            rows=row_count,
            accumulated_rows=accumulator.accumulated_rows,
            accumulated_chunks=accumulator.accumulated_chunks,
            wait_batch=wait_seconds,
            load_rollout=load_seconds,
            train=train_seconds,
            export_policy=export_seconds,
            optimizer_step=int(metrics is not None),
            total=total_seconds,
        )


def _validate_streaming_rollout_batch(
    config: RLConfig,
    batch: RolloutBatch,
    *,
    trainer_step: int,
    row_count: int,
) -> None:
    """Reject a chunk that cannot belong to the current optimizer step."""
    chunks_per_step = _chunks_per_step(config)
    expected_optimizer_step = batch.step // chunks_per_step
    expected_chunk_index = batch.step % chunks_per_step
    if expected_optimizer_step != trainer_step:
        raise ValueError(
            f"Rollout queue step {batch.step} belongs to optimizer step "
            f"{expected_optimizer_step}, but trainer is at step {trainer_step}."
        )

    _validate_rollout_batch(
        config,
        batch,
        trainer_step=trainer_step,
        row_count=row_count,
        chunk_index=expected_chunk_index,
    )


def _validate_rollout_batch(
    config: RLConfig,
    batch: RolloutBatch,
    *,
    trainer_step: int,
    row_count: int,
    chunk_index: int | None,
) -> None:
    """Reject rollout data whose manifest disagrees with trainer state."""
    expected_queue_step = batch.step if chunk_index is not None else trainer_step
    expected_optimizer_step = trainer_step

    minimum_policy_step = required_policy_step(config, trainer_step)
    validate_rollout_manifest(
        batch,
        queue_step=expected_queue_step,
        optimizer_step=expected_optimizer_step,
        chunk_index=chunk_index,
        rows=row_count,
        minimum_policy_step=minimum_policy_step,
        maximum_policy_step=trainer_step,
    )


def _configure_streaming_accumulation(
    trainer: RLTrainer,
    accumulator: _StreamingChunkAccumulator,
    *,
    chunks_per_step: int,
) -> None:
    should_step = accumulator.should_step(chunks_per_step=chunks_per_step)
    remaining_chunks = max(chunks_per_step - accumulator.accumulated_chunks, 0)
    loaded_micro_batches = trainer._loaded_micro_batch_count
    if should_step:
        trainer.accumulation_steps = (
            trainer._accumulated_micro_batches + loaded_micro_batches
        )
    else:
        trainer.accumulation_steps = (
            trainer._accumulated_micro_batches
            + loaded_micro_batches * max(remaining_chunks + 1, 2)
        )

    records = getattr(trainer.dataset, "records", ())
    has_auxiliary_components = any(
        record.ce_weight is not None or record.ref_kl_weight is not None
        for record in records
    )
    if has_auxiliary_components:
        if chunks_per_step != 1:
            raise ValueError(
                "CE/ref-KL components require non-streaming rollout publication "
                "when an optimizer step spans multiple chunks."
            )
        trainer._gradient_accumulation_loss_scale = None
        return

    if accumulator.accumulated_loss_scale > 0.0:
        trainer._set_optimizer_batch_loss_scales({"rl": 1.0, "ce": 1.0, "ref_kl": 1.0})
        trainer._gradient_accumulation_loss_scale = accumulator.accumulated_loss_scale
    else:
        trainer._set_optimizer_batch_loss_scales(None)
        trainer._gradient_accumulation_loss_scale = None


def _log_step_perf_metrics(
    trainer: RLTrainer,
    metrics: dict[str, float],
    *,
    train_seconds: float,
    loop_seconds: float,
) -> None:
    if trainer.monitor is None:
        return
    train_tokens = metrics.get("tokens/train")
    model_tokens = metrics.get("tokens/model", train_tokens)
    if train_tokens is None or train_tokens <= 0:
        return
    if model_tokens is None or model_tokens <= 0:
        model_tokens = train_tokens
    world_size = trainer.world.world_size if trainer.world is not None else 1
    train_tokens_per_second = train_tokens / max(train_seconds, 1e-9)
    model_tokens_per_second = model_tokens / max(train_seconds, 1e-9)
    perf_metrics = {
        "perf/train_seconds": train_seconds,
        "perf/step_seconds": loop_seconds,
        "perf/train_tokens_per_second": train_tokens_per_second,
        "perf/model_tokens_per_second": model_tokens_per_second,
        "perf/step_tokens_per_second": model_tokens / max(loop_seconds, 1e-9),
        "perf/throughput": model_tokens_per_second,
        "perf/throughput_per_gpu": model_tokens_per_second / max(world_size, 1),
        "time/forward_backward": train_seconds,
        "time/step": loop_seconds,
    }
    trainer.monitor.log(perf_metrics, step=trainer.step)


def _validate_distributed_step_sync(trainer: RLTrainer, stepped: bool) -> None:
    if not torch.distributed.is_initialized():
        return
    if trainer.world is None:
        raise RuntimeError("World must be set up before distributed step sync.")
    flag = torch.tensor(int(stepped), device=trainer.world.device)
    min_flag = flag.clone()
    max_flag = flag.clone()
    torch.distributed.all_reduce(min_flag, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(max_flag, op=torch.distributed.ReduceOp.MAX)
    if int(min_flag.item()) != int(max_flag.item()):
        raise RuntimeError(
            "Distributed trainer ranks disagreed on optimizer-step completion for "
            "the current rollout chunk. Increase orchestrator.rollout_chunk_examples "
            "or disable sequence packing for smaller chunks."
        )


def _min_loadable_rollout_rows(config: RLConfig, trainer: RLTrainer) -> int:
    if trainer.world is None or config.data.pack_sequences:
        return 1
    return config.data.micro_batch_size * trainer.world.world_size


def _combined_rollout_path(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    paths: list[Path],
    chunk_index: int,
    min_rows: int,
) -> Path:
    row_count = sum(
        count_nonempty_jsonl_rows(path, description="Rollout chunk") for path in paths
    )
    target_rows = _padded_row_count(row_count, multiple=min_rows)
    if len(paths) == 1 and row_count == target_rows:
        return paths[0]

    output_dir = config.output_dir / "rollouts" / "combined"
    path = output_dir / f"trainer-step-{trainer.step:06d}-chunk-{chunk_index:06d}.jsonl"
    world = trainer.world
    if world is None or world.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".jsonl.tmp")
        first_row: dict[str, object] | None = None
        written = 0
        with tmp_path.open("w", encoding="utf-8") as output:
            for source_path in paths:
                with source_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        if first_row is None:
                            first_row = json.loads(line)
                        output.write(line)
                        written += 1
            if first_row is None:
                raise ValueError("Cannot pad an empty rollout chunk.")
            for _ in range(target_rows - written):
                output.write(json.dumps(_dummy_rollout_row(config, first_row)) + "\n")
        tmp_path.replace(path)
    if world is not None and world.world_size > 1:
        barrier(world)
    return path


def _remove_combined_rollout_path(
    config: RLConfig,
    *,
    trainer: RLTrainer,
    path: Path,
) -> None:
    """Remove the redundant merged rollout after every rank has trained on it."""
    combined_dir = config.output_dir / "rollouts" / "combined"
    if path.parent != combined_dir or not trainer.is_main_process():
        return
    path.unlink(missing_ok=True)


def _padded_row_count(row_count: int, *, multiple: int) -> int:
    if multiple <= 1:
        return row_count
    return ((row_count + multiple - 1) // multiple) * multiple


def _dummy_rollout_row(
    config: RLConfig, source: dict[str, object]
) -> dict[str, object]:
    row = dict(source)
    loss_mask = row.get("loss_mask")
    if not isinstance(loss_mask, list):
        raise TypeError("Cannot create a dummy rollout row without a loss_mask.")
    row["loss_mask"] = [False] * len(loss_mask)
    row[config.data.advantage_column] = 0.0
    row[config.data.reward_column] = None
    row[config.data.temperature_column] = []
    if config.data.inference_logprobs_column in source:
        row[config.data.inference_logprobs_column] = []
    if config.data.teacher_logprobs_column in source:
        row[config.data.teacher_logprobs_column] = []
    metadata = dict(row.get(config.data.metadata_column) or {})
    metadata["_wavelet_dummy_rollout"] = True
    row[config.data.metadata_column] = metadata
    return row


if __name__ == "__main__":
    sys.exit(main())

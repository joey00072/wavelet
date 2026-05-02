from __future__ import annotations

import contextlib
import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from wavelet.configs.rl_config import RLConfig
from wavelet.data.loading import Example
from wavelet.data.rl_dataset import (
    PackedRLDataset,
    RLDataset,
    _normalize_rl_record,
    _pretokenized_sample,
    setup_rl_dataloader,
    setup_rl_dataset,
)
from wavelet.data.tokenization import build_sample
from wavelet.orchestrator.queue import (
    POLICY_META_FILENAME,
    STABLE_BATCH_MARKER,
    get_policy_step_dir,
    resolve_policy_dir,
)
from wavelet.trainer.base import BaseTrainer
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.rl_loss import compute_loss, selective_log_softmax
from wavelet.utils.monitoring import RunMonitor
from wavelet.utils.pathing import resolve_resume_checkpoint


logger = logging.getLogger(__name__)


def _count_rollout_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    if rows == 0:
        raise ValueError(f"Rollout batch '{path}' contains no rows.")
    return rows


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
    segment_ids = torch.where(valid_tokens, segment_ids, torch.full_like(segment_ids, -1))

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

    PrimeRL-style packing uses reset position ids as sequence boundaries. HF's
    FlashAttention 2 implementation consumes those position ids directly and
    dispatches to flash_attn_varlen_func, but only when no explicit attention
    mask is passed. Non-flash attention still needs an explicit block-causal
    mask to prevent packed samples from attending across boundaries.
    """
    if not _has_packed_position_resets(attention_mask, position_ids):
        valid_tokens = attention_mask.bool()
        return None if valid_tokens.all() else attention_mask

    implementation = _model_attn_implementation(model)
    if implementation in {"flash_attention_2", "flash_attention_3"}:
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


class RLTrainer(BaseTrainer):
    def __init__(self, config: RLConfig) -> None:
        super().__init__(config)
        self.config = config
        self.dataloader = None
        self.monitor: RunMonitor | None = None
        self.step = 0
        self._micro_step = 0
        self._accumulated_micro_batches = 0
        self._reward_accum: list[float] = []
        self._rollout_metric_accum: list[dict[str, float]] = []
        self._train_loss_accum: list[float] = []
        self._train_metric_accum: list[dict[str, float]] = []
        self._optimizer_batch_loss_scale: int | None = None
        self._loaded_micro_batch_count = 0
        self.accumulation_steps = 1
        self.ckpt_manager: CheckpointManager | None = None
        self.resume_checkpoint_dir: Path | None = None
        self._run_closed = False

    def _setup_data(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be set up before data")
        if self.world is None:
            raise RuntimeError("World must be set up before data")
        if self.config.data.pack_sequences:
            self.accumulation_steps = 1
        else:
            self._setup_accumulation_steps()

        self.dataset = setup_rl_dataset(
            self.tokenizer,
            self.config.data,
            data_rank=self.world.rank,
            data_world_size=self.world.world_size,
        )
        self.dataloader = setup_rl_dataloader(
            self.dataset,
            self.config.data,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if isinstance(self.dataset, PackedRLDataset):
            self.accumulation_steps = self.dataset.micro_batch_count()
        self._optimizer_batch_loss_scale = self._estimate_optimizer_batch_loss_scale()

    def setup(self) -> None:
        super().setup()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = RunMonitor(
            output_dir=self.output_dir,
            enabled=self.config.monitor.enabled,
            write_events=self.config.monitor.write_events,
            write_metrics_jsonl=self.config.monitor.write_metrics_jsonl,
            write_metrics_csv=self.config.monitor.write_metrics_csv,
            write_run_metadata=self.config.monitor.write_run_metadata,
            write_heartbeat=self.config.monitor.write_heartbeat,
            log_cuda_memory=self.config.monitor.log_cuda_memory,
            log_disk_usage=self.config.monitor.log_disk_usage,
            wandb=self.config.monitor.wandb,
        )
        if (
            self.model is None
            or self.optimizer is None
            or self.dataloader is None
            or self.world is None
        ):
            raise RuntimeError("Trainer not set up. Call setup() first.")

        self._validate_reference_policy_support()

        self.ckpt_manager = CheckpointManager(
            self.model,
            self.optimizer,
            self.scheduler,
            self.config.ckpt,
            self.output_dir,
            self.world,
        )
        if self.config.ckpt is not None and self.config.ckpt.resume_step is not None:
            self.resume_checkpoint_dir = resolve_resume_checkpoint(
                self.output_dir,
                self.config.ckpt.resume_step,
            )
            trainer_state = self.ckpt_manager.load(
                self.resume_checkpoint_dir,
                dataloader=self.dataloader,
            )
            if trainer_state.micro_step != trainer_state.step * self.accumulation_steps:
                raise ValueError(
                    "Checkpoint micro_step does not match the expected optimizer-step "
                    "boundary for this trainer configuration."
                )
            self.step = trainer_state.step
            self._micro_step = trainer_state.micro_step
            self._accumulated_micro_batches = 0

        self.monitor.start_run(
            run_config=self.config.model_dump(mode="json", exclude_none=True),
            world=self.world,
            resumed_from=(
                str(self.resume_checkpoint_dir)
                if self.resume_checkpoint_dir is not None
                else None
            ),
        )

    def train(self) -> None:
        target_step = self.config.max_steps or 1000
        self.train_until(target_step, finish_run=True)

    def train_until(self, target_step: int, *, finish_run: bool = False) -> None:
        if self.model is None or self.optimizer is None or self.dataloader is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self._run_closed:
            raise RuntimeError("Trainer run has already been finalized.")

        self.model.train()
        remaining_steps = max(target_step - self.step, 0)
        progress = tqdm(total=remaining_steps, disable=not self.world.is_main)

        try:
            while self.step < target_step:
                for batch in self.dataloader:
                    batch = self._prepare_batch(batch)
                    metrics = self._train_step(batch)
                    if metrics is None:
                        continue

                    self._log_train_metrics(metrics, progress)
                    self._maybe_checkpoint()
                    progress.update(1)
                    if self.step >= target_step:
                        break

            if self.ckpt_manager is not None:
                self.ckpt_manager.wait_for_pending_save()
        except Exception:
            self._finish_if_requested(finish_run, status="failed")
            raise
        finally:
            progress.close()
        self._finish_if_requested(finish_run, status="completed")

    def _log_train_metrics(self, metrics: dict[str, float], progress: tqdm) -> None:
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self.step % self.config.log.log_every != 0:
            return
        metrics["lr"] = self._get_lr()
        metrics["optim/lr"] = metrics["lr"]
        metrics["progress/step"] = float(self.step)
        metrics["progress/micro_step"] = float(self._micro_step)
        self.monitor.log(metrics, self.step)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            kl=f"{metrics['mismatch_kl']:.4f}",
            lr=f"{metrics['lr']:.2e}",
        )

    def _maybe_checkpoint(self) -> None:
        if self.ckpt_manager is None:
            return
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")

        self.ckpt_manager.poll_pending_save()
        did_save = self.ckpt_manager.save(
            TrainerState(step=self.step, micro_step=self._micro_step),
            dataloader=self.dataloader,
        )
        if not did_save:
            return
        self.monitor.log_event(
            "checkpoint_triggered",
            step=self.step,
            payload={"mode": self.config.ckpt.mode if self.config.ckpt else "disabled"},
        )

    def _finish_if_requested(self, finish_run: bool, *, status: str) -> None:
        if finish_run:
            if self.monitor is None:
                raise RuntimeError("Monitor not set up. Call setup() first.")
            self.monitor.finish(status=status, step=self.step)
            self._run_closed = True
            if status == "completed":
                self._save_model()

    def load_rollout_path(self, rollout_path: Path) -> None:
        rollout_path = Path(rollout_path)
        rollout_count = _count_rollout_rows(rollout_path)
        optimizer_batch_size = rollout_count
        pack_sequences = self.config.data.pack_sequences
        if self.world is not None and not pack_sequences:
            global_micro_batch = (
                self.config.data.micro_batch_size * self.world.world_size
            )
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

    def _current_micro_batch_count(self, optimizer_batch_size: int) -> int:
        if isinstance(self.dataset, PackedRLDataset):
            return self.dataset.micro_batch_count()
        if self.world is None:
            raise RuntimeError("World must be set up before computing micro-batches")
        global_micro_batch = self.config.data.micro_batch_size * self.world.world_size
        return max(optimizer_batch_size // global_micro_batch, 1)

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
            metrics = self._train_step(batch)
            if metrics is not None:
                progress = tqdm(total=1, disable=not self.world.is_main)
                self._log_train_metrics(metrics, progress)
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
        except Exception:
            return json.dumps(messages, ensure_ascii=False)

    def should_export_policy(self, step: int) -> bool:
        if step == 0:
            return self.config.policy_transfer.export_initial
        return step % self.config.policy_transfer.export_every_steps == 0

    def export_policy(self, *, step: int | None = None) -> Path | None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")

        export_step = self.step if step is None else step
        if not self.should_export_policy(export_step):
            return None
        if self.config.policy_transfer.type == "nccl":
            raise NotImplementedError(
                "NCCL policy transfer primitives are available in "
                "wavelet.trainer.nccl_broadcast, but the vLLM worker-side hot "
                "weight receiver is not wired into RL training yet. Use "
                "policy_transfer.type='filesystem' for runnable LoRA RL."
            )

        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        step_dir = get_policy_step_dir(policy_dir, export_step)
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")

        from wavelet.trainer.model import (
            export_model_for_save,
            save_lora_adapter_snapshot,
            save_lora_adapter_snapshot_from_fsdp,
            save_model,
        )

        export_model = None
        state_dict = None
        if (
            self.config.policy_transfer.lightweight_lora
            and isinstance(self.model, FSDP)
        ):
            if tmp_dir.exists() and self.world.is_main:
                shutil.rmtree(tmp_dir)
            if step_dir.exists() and self.world.is_main:
                shutil.rmtree(step_dir)
            if self.world.is_main:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            saved_path = save_lora_adapter_snapshot_from_fsdp(
                self.model,
                tmp_dir,
                is_main_process=self.world.is_main,
            )
        else:
            export_model, state_dict = export_model_for_save(self.model)
            if self.world.is_main:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                if step_dir.exists():
                    shutil.rmtree(step_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
            if self.config.policy_transfer.lightweight_lora and isinstance(
                export_model, PeftModel
            ):
                saved_path = save_lora_adapter_snapshot(
                    export_model,
                    tmp_dir,
                    state_dict=state_dict,
                    is_main_process=self.world.is_main,
                )
            else:
                saved_path = save_model(
                    export_model,
                    self.tokenizer,
                    tmp_dir,
                    state_dict=state_dict,
                    is_main_process=self.world.is_main,
                )
        export_model = None
        state_dict = None
        if self.world.is_main:
            meta = {
                "format_version": 1,
                "step": export_step,
                "kind": saved_path.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (tmp_dir / POLICY_META_FILENAME).write_text(json.dumps(meta))
        self.offload_after_refit()
        if self.world.world_size > 1:
            torch.distributed.barrier()
        if self.world.is_main:
            (tmp_dir / STABLE_BATCH_MARKER).touch()
            tmp_dir.replace(step_dir)
        if self.world.world_size > 1:
            torch.distributed.barrier()
        return step_dir

    def finalize(self, *, status: str = "completed") -> None:
        if self.monitor is None or self._run_closed:
            return
        self.monitor.finish(status=status, step=self.step)
        self._run_closed = True
        if status == "completed" and not self._uses_sleep_colocation():
            self._save_model()

    def _validate_reference_policy_support(self) -> None:
        if not isinstance(self.dataset, RLDataset):
            return
        if self.model is None:
            raise RuntimeError("Model must be set up before validating RL data")
        from wavelet.trainer.model import unwrap_model

        missing_references = any(
            record.inference_logprobs is None for record in self.dataset.records
        )
        if not missing_references:
            return
        model = unwrap_model(self.model)
        if not callable(getattr(model, "disable_adapter", None)):
            raise ValueError(
                "RL data is missing inference_logprobs for at least one sample, but the "
                "current model cannot derive a reference policy by disabling adapters. "
                "Use LoRA or provide inference_logprobs in the dataset."
            )

    def _setup_accumulation_steps(self) -> None:
        if self.world is None:
            raise RuntimeError("World must be set up before accumulation steps")
        global_micro_batch = self.config.data.micro_batch_size * self.world.world_size
        if self.config.data.batch_size % global_micro_batch != 0:
            raise ValueError(
                "RL data.batch_size is the global optimizer batch size and must be "
                "divisible by data.micro_batch_size * world_size "
                f"({self.config.data.micro_batch_size} * {self.world.world_size})."
            )
        self.accumulation_steps = self.config.data.batch_size // global_micro_batch

    def _estimate_optimizer_batch_loss_scale(self) -> float | None:
        if self.config.data.num_workers != 0:
            return None
        if not isinstance(self.dataset, (RLDataset, PackedRLDataset)):
            return None
        records = self.dataset.records
        if any(
            record.advantage is None and record.reward is None
            for record in records
        ):
            return None
        return self.dataset.loss_scale_for_next_local_batch(self.accumulation_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.world is None:
            raise RuntimeError("World not set up")
        return {
            key: value.to(self.world.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float] | None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer not set up")

        attention_mask = _packed_training_attention_mask(
            self.model,
            batch["attention_mask"],
            batch["position_ids"],
        )
        if attention_mask is not None and attention_mask.is_floating_point():
            mask_dtype = _torch_dtype_from_name(self.config.model.torch_dtype)
            if mask_dtype is not None:
                attention_mask = attention_mask.to(dtype=mask_dtype)

        with self.act_offload_ctx:
            trainer_logprobs = self._model_logprobs(batch, attention_mask)
            if not batch["loss_mask"].bool().any():
                # Dummy packed batches keep FSDP ranks aligned after filtering.
                # Make the zero loss explicitly depend on this rank's forward pass
                # so backward still traverses the sharded graph with zero grads.
                loss = trainer_logprobs.sum() * 0.0
                raw_metrics = self._zero_loss_metrics(loss)
            else:
                inference_logprobs = self._inference_logprobs(batch, attention_mask)
                teacher_logprobs = self._teacher_logprobs(batch)
                loss, raw_metrics = compute_loss(
                    trainer_logprobs,
                    inference_logprobs,
                    teacher_logprobs,
                    batch["advantages"],
                    batch["loss_mask"],
                    self.config.loss,
                    loss_scale=self._optimizer_batch_loss_scale,
                )

            if torch.isnan(loss):
                logger.warning(f"NaN RL loss at step {self.step}, skipping backward")
                self._micro_step += 1
                self._accumulated_micro_batches += 1
                if self._accumulated_micro_batches >= self.accumulation_steps:
                    self._reward_accum.clear()
                    self._rollout_metric_accum.clear()
                    self._train_loss_accum.clear()
                    self._train_metric_accum.clear()
                    self._accumulated_micro_batches = 0
                    self._optimizer_batch_loss_scale = (
                        self._estimate_optimizer_batch_loss_scale()
                    )
                return None

            sync_context = self._maybe_no_sync()
            with sync_context:
                if self._optimizer_batch_loss_scale is None:
                    backward_loss = loss / self.accumulation_steps
                else:
                    backward_loss = loss
                backward_loss.backward()

        reward_mean = self._reward_mean(
            batch["rewards"],
            sample_counts=batch.get("sample_counts"),
        )
        if reward_mean is not None:
            self._reward_accum.append(reward_mean)
        self._rollout_metric_accum.append(self._batch_rollout_metrics(batch))
        self._train_loss_accum.append(float(loss.detach().item()))
        self._train_metric_accum.append(
            {key: float(value.detach().item()) for key, value in raw_metrics.items()}
        )

        self._micro_step += 1
        self._accumulated_micro_batches += 1
        if self._accumulated_micro_batches < self.accumulation_steps:
            return None

        grad_norm = None
        if self.config.max_grad_norm > 0:
            grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self._accumulated_micro_batches = 0

        if self._optimizer_batch_loss_scale is None:
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
        metrics.update(self._prime_style_metric_aliases(metrics))
        metrics = self._sync_metrics(metrics)
        self._optimizer_batch_loss_scale = self._estimate_optimizer_batch_loss_scale()
        return metrics

    def _zero_loss_metrics(self, loss: Tensor) -> dict[str, Tensor]:
        zero = loss.detach() * 0.0
        return {
            "mismatch_kl": zero,
            "masked_mismatch_kl": zero,
            "unmasked_mismatch_kl": zero,
            "is_masked": zero,
            "is_masked_low": zero,
            "is_masked_high": zero,
            "policy_loss": zero,
            "kl_loss": zero,
            "advantage_mean": zero,
        }

    def _inference_logprobs(
        self, batch: dict[str, Tensor], attention_mask: Tensor | None
    ) -> Tensor:
        inference_logprobs = batch["inference_logprobs"].clone()
        has_inference = batch["has_inference_logprobs"].bool()
        if has_inference.all():
            return inference_logprobs

        from wavelet.trainer.model import unwrap_model

        model = unwrap_model(self.model)
        disable_adapter = getattr(model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "inference_logprobs are required when the model cannot disable adapters."
            )

        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                with disable_adapter():
                    computed_inference = self._model_logprobs(batch, attention_mask)
        finally:
            if training:
                self.model.train()
        inference_logprobs[~has_inference] = computed_inference[~has_inference]
        return inference_logprobs

    def _model_logprobs(
        self,
        batch: dict[str, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        if self.model is None:
            raise RuntimeError("Model not set up")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            position_ids=batch["position_ids"],
            labels=batch["target_ids"],
            temperature=batch["temperatures"],
        )
        if isinstance(outputs, dict) and outputs.get("logprobs") is not None:
            return outputs["logprobs"]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        scaled_logits = logits / batch["temperatures"].unsqueeze(-1)
        return selective_log_softmax(scaled_logits, batch["target_ids"])

    def _teacher_logprobs(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not batch["has_teacher_logprobs"].bool().any():
            return None
        if not batch["has_teacher_logprobs"].bool().all():
            raise ValueError(
                "teacher_logprobs must be provided for either all samples in a batch or none."
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
            "seq_len/all/mean": float(seq_lens.mean().item()),
            "seq_len/all/max": float(seq_lens.max().item()),
            "seq_len/all/min": float(seq_lens.min().item()),
        }

        rewards = batch["rewards"]
        valid_rewards = rewards[~torch.isnan(rewards)]
        if valid_rewards.numel() > 0:
            if sample_counts is not None:
                valid = ~torch.isnan(rewards)
                weights = sample_counts[valid].float().clamp_min(0)
                if weights.sum() > 0:
                    reward_mean = float(
                        (rewards[valid].float() * weights).sum().item()
                        / weights.sum().item()
                    )
                else:
                    reward_mean = float(valid_rewards.mean().item())
            else:
                reward_mean = float(valid_rewards.mean().item())
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
            if key.endswith("/max"):
                aggregated[key] = max(values)
            elif key.endswith("/min"):
                aggregated[key] = min(values)
            elif key in {"tokens/train", "rollout/count", "micro_batch/count"}:
                aggregated[key] = sum(values)
            else:
                aggregated[key] = sum(values) / len(values)
        advantage_count = sum(
            metrics.get("_advantage_count", 0.0) for metrics in micro_metrics
        )
        if advantage_count > 0:
            advantage_sum = sum(
                metrics.get("_advantage_sum", 0.0) for metrics in micro_metrics
            )
            advantage_sumsq = sum(
                metrics.get("_advantage_sumsq", 0.0) for metrics in micro_metrics
            )
            advantage_mean = advantage_sum / advantage_count
            advantage_var = max(
                advantage_sumsq / advantage_count - advantage_mean**2,
                0.0,
            )
            aggregated["advantage/all/mean"] = advantage_mean
            aggregated["advantage/all/std"] = advantage_var**0.5
        return aggregated

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
                aggregated[key] = sum(values) / len(values)
        return aggregated

    def _prime_style_metric_aliases(
        self, metrics: dict[str, float]
    ) -> dict[str, float]:
        aliases: dict[str, float] = {}
        if "loss" in metrics:
            aliases["train/loss"] = metrics["loss"]
        if "policy_loss" in metrics:
            aliases["train/policy_loss"] = metrics["policy_loss"]
        if "kl_loss" in metrics:
            aliases["train/kl_loss"] = metrics["kl_loss"]
        if "mismatch_kl" in metrics:
            aliases["kl/mismatch"] = metrics["mismatch_kl"]
        if "masked_mismatch_kl" in metrics:
            aliases["kl/masked_mismatch"] = metrics["masked_mismatch_kl"]
        if "unmasked_mismatch_kl" in metrics:
            aliases["kl/unmasked_mismatch"] = metrics["unmasked_mismatch_kl"]
        if "is_masked" in metrics:
            aliases["dppo/is_masked"] = metrics["is_masked"]
        if "is_masked_low" in metrics:
            aliases["dppo/is_masked_low"] = metrics["is_masked_low"]
        if "is_masked_high" in metrics:
            aliases["dppo/is_masked_high"] = metrics["is_masked_high"]
        if "advantage_mean" in metrics:
            aliases["advantage/token/mean"] = metrics["advantage_mean"]
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

    def _clip_grad_norm(self) -> float:
        if self.model is None:
            raise RuntimeError("Trainer not set up")
        clip_grad_norm = getattr(self.model, "clip_grad_norm_", None)
        if callable(clip_grad_norm):
            clipped = clip_grad_norm(self.config.max_grad_norm)
        else:
            clipped = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        if isinstance(clipped, Tensor):
            return float(clipped.detach().item())
        return float(clipped)

    def _sync_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if (
            self.world is None
            or self.world.world_size <= 1
            or not torch.distributed.is_initialized()
        ):
            return metrics

        synced: dict[str, float] = {}
        device = self.world.device
        for key in sorted(metrics):
            value = torch.tensor(float(metrics[key]), device=device)
            if key.endswith("/max"):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
                synced[key] = float(value.item())
            elif key.endswith("/min"):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
                synced[key] = float(value.item())
            elif key in {"rollout/count", "micro_batch/count", "tokens/train"}:
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                synced[key] = float(value.item())
            else:
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                synced[key] = float(value.item() / self.world.world_size)
        return synced

    def _get_lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def _save_model(self) -> None:
        if not self.world:
            return
        if self.model is None or self.tokenizer is None:
            return

        from wavelet.trainer.model import (
            export_model_for_save,
            save_lora_adapter_snapshot_from_fsdp,
            save_model,
        )

        if self.config.policy_transfer.lightweight_lora and isinstance(
            self.model, FSDP
        ):
            saved_path = save_lora_adapter_snapshot_from_fsdp(
                self.model,
                self.output_dir,
                is_main_process=self.world.is_main,
            )
            if self.world.is_main:
                self.tokenizer.save_pretrained(saved_path)
                logger.info(f"Model saved to {self.output_dir}")
            return

        saveable_model, state_dict = export_model_for_save(self.model)
        save_model(
            saveable_model,
            self.tokenizer,
            self.output_dir,
            state_dict=state_dict,
            is_main_process=self.world.is_main,
        )
        if self.world.is_main:
            logger.info(f"Model saved to {self.output_dir}")

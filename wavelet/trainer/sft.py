from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from wavelet.configs.sft import SFTConfig
from wavelet.data.dataloader import setup_dataloader
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.base import BaseTrainer
from wavelet.utils.pathing import resolve_resume_checkpoint
from wavelet.utils.monitoring import RunMonitor


logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    def __init__(self, config: SFTConfig) -> None:
        super().__init__(config)
        self.dataloader = None
        self.monitor: RunMonitor | None = None
        self.step = 0  # optimizer step counter; max_steps is in optimizer-step units
        self._micro_step = 0  # internal micro-batch counter for accumulation tracking
        self.accumulation_steps = 1
        self.ckpt_manager: CheckpointManager | None = None
        self.resume_checkpoint_dir: Path | None = None

    def _setup_data(self) -> None:
        super()._setup_data()
        if not self.tokenizer:
            raise RuntimeError("Tokenizer must be set up before data")
        self._setup_accumulation_steps()

        self.dataloader = setup_dataloader(
            self.dataset,
            self.config.data,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _setup_accumulation_steps(self) -> None:
        if self.world is None:
            raise RuntimeError("World must be set up before accumulation steps")
        global_micro_batch = self.config.data.micro_batch_size * self.world.world_size
        if self.config.data.batch_size % global_micro_batch != 0:
            raise ValueError(
                "SFT data.batch_size is the global optimizer batch size and must be "
                "divisible by data.micro_batch_size * world_size "
                f"({self.config.data.micro_batch_size} * {self.world.world_size})."
            )
        self.accumulation_steps = self.config.data.batch_size // global_micro_batch

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
        if self.world is None:
            raise RuntimeError("World not set up")
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
        if self.model is None or self.optimizer is None or self.dataloader is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")

        self.model.train()
        max_steps = self.config.max_steps or 1000
        status = "completed"

        # self.step counts optimizer steps; progress bar is in optimizer-step units
        progress = tqdm(total=max_steps, disable=not self.world.is_main)
        try:
            while self.step < max_steps:
                for batch in self.dataloader:
                    batch = self._prepare_batch(batch)
                    loss = self._train_step(batch)

                    # _train_step returns None while accumulating; non-None signals
                    # that an optimizer step just fired (loss is ready to log).
                    if loss is None:
                        continue

                    if self.step % self.config.log.log_every == 0:
                        loss_val = loss.item()
                        lr_val = self._get_lr()
                        metrics = {"loss": loss_val, "lr": lr_val}
                        self.monitor.log(metrics, self.step)
                        progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")

                    if self.ckpt_manager is not None:
                        self.ckpt_manager.poll_pending_save()
                        did_save = self.ckpt_manager.save(
                            TrainerState(step=self.step, micro_step=self._micro_step),
                            dataloader=self.dataloader,
                        )
                        if did_save:
                            self.monitor.log_event(
                                "checkpoint_triggered",
                                step=self.step,
                                payload={
                                    "mode": self.config.ckpt.mode
                                    if self.config.ckpt
                                    else "disabled"
                                },
                            )

                    progress.update(1)

                    if self.step >= max_steps:
                        break

            if self.ckpt_manager is not None:
                self.ckpt_manager.wait_for_pending_save()
            self._save_model()
        except Exception:
            status = "failed"
            raise
        finally:
            progress.close()
            self.monitor.finish(status=status, step=self.step)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.world:
            raise RuntimeError("World not set up")
        return {k: v.to(self.world.device, non_blocking=True) for k, v in batch.items()}

    def _train_step(self, batch: dict[str, Tensor]) -> Tensor | None:
        if self.model is None:
            raise RuntimeError("Model not set up")

        # With cat packing all tokens are real (no padding), so attention_mask
        # is all-ones. Passing None lets transformers use is_causal=True which
        # enables the memory-efficient SDPA kernel instead of the O(L²) math backend.
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None and attn_mask.all():
            attn_mask = None

        with self.act_offload_ctx:
            if self.config.loss_impl == "liger_fused":
                # Liger's fused_linear_cross_entropy computes loss internally.
                # Wavelet's labels are pre-shifted (labels[i] = next token at i+1),
                # so we pass them as shift_labels to skip liger's built-in shift.
                # This avoids a double-shift that would compute loss 2 tokens ahead.
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=attn_mask,
                    position_ids=batch["position_ids"],
                    shift_labels=batch["labels"],
                )
                loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=attn_mask,
                    position_ids=batch["position_ids"],
                )
                loss = self.compute_loss(outputs.logits, batch["labels"])

            if torch.isnan(loss):
                logger.warning(f"NaN loss at step {self.step}, skipping backward")
                self._micro_step += 1
                return None

            (loss / self.accumulation_steps).backward()

        self._micro_step += 1
        if self._micro_step % self.accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return loss

        return None

    def compute_loss(self, logits: Tensor, labels: Tensor, chunk: int = 256) -> Tensor:
        """Chunked cross-entropy loss.

        Slices the sequence into chunks of `chunk` tokens and accumulates the
        loss, so we never materialise the full (B*L, V) logit gradient at once.
        With vocab_size=152k and seq_len=2048 that tensor is ~1.2 GB; chunking
        reduces peak gradient memory to ~chunk/seq_len of that.
        """
        B, L, V = logits.shape
        flat_labels = labels.view(-1)
        total_loss = logits.new_zeros(())
        valid = (flat_labels != -100).sum()
        if valid == 0:
            return total_loss

        for start in range(0, B * L, chunk):
            end = min(start + chunk, B * L)
            chunk_logits = logits.view(-1, V)[start:end]
            chunk_labels = flat_labels[start:end]
            chunk_loss = nn.functional.cross_entropy(
                chunk_logits, chunk_labels, ignore_index=-100, reduction="sum"
            )
            total_loss = total_loss + chunk_loss

        del logits
        return total_loss / valid.float()

    def _get_lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def _save_model(self) -> None:
        if not self.world:
            return
        if not self.model or not self.tokenizer:
            return

        from wavelet.trainer.model import export_model_for_save, save_model

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

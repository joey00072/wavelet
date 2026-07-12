from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from wavelet.configs.sft import SFTConfig
from wavelet.data.batch import SFTBatch
from wavelet.data.dataloader import setup_dataloader
from wavelet.trainer.base import BaseTrainer
from wavelet.trainer.lora import sync_hf_tp_lora_replicated_grads
from wavelet.trainer.types import LossOutput, TrainOutput


logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    def __init__(self, config: SFTConfig) -> None:
        super().__init__(config)

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

    def _log_train_output(self, output: TrainOutput, progress: tqdm) -> None:
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self.step % self.config.log.log_every != 0:
            return
        loss = output.loss.loss.item()
        lr = self._get_lr()
        self.monitor.log({"loss": loss, "lr": lr}, self.step)
        progress.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

    def _train_step(self, batch: SFTBatch) -> TrainOutput:
        if self.model is None:
            raise RuntimeError("Model not set up")

        # With cat packing all tokens are real (no padding), so attention_mask
        # is all-ones. Passing None lets transformers use is_causal=True which
        # enables the memory-efficient SDPA kernel instead of the O(L²) math backend.
        attn_mask = batch.attention_mask
        if attn_mask is not None and attn_mask.all():
            attn_mask = None

        with self.act_offload_ctx:
            if self.config.loss_impl == "liger_fused":
                # Liger's fused_linear_cross_entropy computes loss internally.
                # Wavelet's labels are pre-shifted (labels[i] = next token at i+1),
                # so we pass them as shift_labels to skip liger's built-in shift.
                # This avoids a double-shift that would compute loss 2 tokens ahead.
                outputs = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=attn_mask,
                    position_ids=batch.position_ids,
                    shift_labels=batch.labels,
                )
                loss_output = LossOutput(loss=outputs.loss)
            else:
                outputs = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=attn_mask,
                    position_ids=batch.position_ids,
                )
                loss_output = self.compute_loss(outputs.logits, batch.labels)
            loss = loss_output.loss

            if torch.isnan(loss):
                logger.warning(f"NaN loss at step {self.step}, skipping backward")
                self._micro_step += 1
                return TrainOutput(
                    loss=loss_output,
                    stepped=False,
                    step=self.step,
                    micro_step=self._micro_step,
                    skipped=True,
                )

            (loss / self.accumulation_steps).backward()

        self._micro_step += 1
        if self._micro_step % self.accumulation_steps == 0:
            sync_hf_tp_lora_replicated_grads(self.model, self.parallel_dims)
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return TrainOutput(
                loss=loss_output,
                stepped=True,
                step=self.step,
                micro_step=self._micro_step,
                metrics={"loss": float(loss.detach().item())},
            )

        return TrainOutput(
            loss=loss_output,
            stepped=False,
            step=self.step,
            micro_step=self._micro_step,
        )

    def compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        chunk: int = 256,
    ) -> LossOutput:
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
            return LossOutput(loss=total_loss)

        for start in range(0, B * L, chunk):
            end = min(start + chunk, B * L)
            chunk_logits = logits.view(-1, V)[start:end]
            chunk_labels = flat_labels[start:end]
            chunk_loss = nn.functional.cross_entropy(
                chunk_logits, chunk_labels, ignore_index=-100, reduction="sum"
            )
            total_loss = total_loss + chunk_loss

        del logits
        loss = total_loss / valid.float()
        return LossOutput(
            loss=loss,
            metrics={"nll": loss.detach(), "tokens/train": valid.detach()},
        )

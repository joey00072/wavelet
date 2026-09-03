# ruff: noqa: E402, F811

from __future__ import annotations

import contextlib
import gc
import logging
import os
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from wavelet.configs.sft import SFTConfig
from wavelet.data.sft import setup_dataloader
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import (
    ParallelDims,
    World,
    distributed_uses_cuda,
    get_world,
    set_world,
)
from wavelet.trainer.model import sync_hf_tp_lora_replicated_grads
from wavelet.trainer.types import LossOutput, TrainOutput
from wavelet.utils.monitoring import RunMonitor
from wavelet.utils.pathing import resolve_resume_checkpoint

if TYPE_CHECKING:
    from wavelet.configs.rl_config import RLConfig


logger = logging.getLogger(__name__)


def _lora_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return None


class BaseTrainer:
    def __init__(self, config: SFTConfig | "RLConfig") -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self.dataset: IterableDataset | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.dataloader: Any = None
        self.world: World | None = None
        self.parallel_dims: ParallelDims | None = None
        self.output_dir = Path(config.output_dir)
        self.act_offload_ctx: contextlib.AbstractContextManager = (
            contextlib.nullcontext()
        )
        self.monitor: RunMonitor | None = None
        self.step = 0
        self._micro_step = 0
        self.accumulation_steps = 1
        self.ckpt_manager: CheckpointManager | None = None
        self.resume_checkpoint_dir: Path | None = None
        self._run_closed = False

    def setup(self) -> None:
        self._setup_seed()
        self._setup_distributed()
        self._setup_tokenizer()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        self._validate_ready()
        self._setup_run()

    def _validate_ready(self) -> None:
        if (
            self.model is None
            or self.optimizer is None
            or self.dataloader is None
            or self.world is None
        ):
            raise RuntimeError("Trainer not set up. Call setup() first.")

    def _after_resume(self) -> None:
        pass

    def _validate_resume_state(self, state: TrainerState) -> None:
        if state.micro_step != state.step * self.accumulation_steps:
            raise ValueError(
                "Checkpoint micro_step does not match the expected optimizer-step "
                "boundary for this trainer configuration."
            )

    def _checkpoint_dataloader(self) -> StatefulDataLoader | None:
        return self.dataloader

    def _setup_run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = self.config.monitor
        self.monitor = RunMonitor(
            output_dir=self.output_dir,
            checkpoint_dir=self.config.checkpoint_output_dir,
            enabled=config.enabled,
            write_events=config.write_events,
            write_metrics_jsonl=config.write_metrics_jsonl,
            write_metrics_csv=config.write_metrics_csv,
            write_run_metadata=config.write_run_metadata,
            write_heartbeat=config.write_heartbeat,
            log_cuda_memory=config.log_cuda_memory,
            log_disk_usage=config.log_disk_usage,
            wandb=config.wandb,
        )
        assert self.model is not None
        assert self.optimizer is not None
        assert self.world is not None
        self.ckpt_manager = CheckpointManager(
            self.model,
            self.optimizer,
            self.scheduler,
            self.config.ckpt,
            self.config.checkpoint_output_dir,
            self.world,
        )
        if self.config.ckpt is not None and self.config.ckpt.resume_step is not None:
            self.resume_checkpoint_dir = resolve_resume_checkpoint(
                self.config.checkpoint_output_dir,
                self.config.ckpt.resume_step,
            )
            state = self.ckpt_manager.load(
                self.resume_checkpoint_dir,
                dataloader=self._checkpoint_dataloader(),
            )
            self._validate_resume_state(state)
            self.step = state.step
            self._micro_step = state.micro_step
            self._after_resume()
        self.monitor.start_run(
            run_config=self.config.model_dump(mode="json", exclude_none=True),
            world=self.world,
            resumed_from=(
                str(self.resume_checkpoint_dir)
                if self.resume_checkpoint_dir is not None
                else None
            ),
        )

    def _setup_seed(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self) -> None:
        self.train_until(self._compute_total_steps(), finish_run=True)

    def train_until(self, target_step: int, *, finish_run: bool = False) -> None:
        self._validate_ready()
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        if self._run_closed:
            raise RuntimeError("Trainer run has already been finalized.")
        assert self.model is not None
        assert self.world is not None
        self.model.train()
        progress = tqdm(
            total=max(target_step - self.step, 0),
            disable=not self.world.is_main,
        )
        try:
            while self.step < target_step:
                for batch in self.dataloader:
                    output = self._train_step(self._prepare_batch(batch))
                    if not output.stepped:
                        continue
                    self._log_train_output(output, progress)
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

    def _train_step(self, batch: dict[str, torch.Tensor]) -> TrainOutput:
        raise NotImplementedError

    def _log_train_output(self, output: TrainOutput, progress: tqdm) -> None:
        raise NotImplementedError

    def _finish_if_requested(self, finish_run: bool, *, status: str) -> None:
        if not finish_run:
            return
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        self.monitor.finish(status=status, step=self.step)
        self._run_closed = True
        if status == "completed":
            self._save_model()

    def _setup_distributed(self) -> None:
        fsdp_config = getattr(self.config, "fsdp", None)
        should_init_dist = (
            bool(fsdp_config is not None and fsdp_config.enabled)
            or int(os.environ.get("WORLD_SIZE", "1")) > 1
        )

        if should_init_dist and not torch.distributed.is_initialized():
            backend = self._distributed_backend()
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method="env://",
                    timeout=timedelta(minutes=30),
                )
            else:
                rendezvous_file = tempfile.NamedTemporaryFile(
                    prefix="wavelet-dist-",
                    suffix=".init",
                    delete=False,
                )
                rendezvous_file.close()
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method=f"file://{rendezvous_file.name}",
                    rank=0,
                    world_size=1,
                    timeout=timedelta(minutes=30),
                )

        if torch.distributed.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if torch.cuda.is_available() and distributed_uses_cuda():
                torch.cuda.set_device(local_rank)
            world = get_world()
            self.world = world
            self._setup_parallel_dims()
            return

        world = World(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        set_world(world)
        self.world = world
        self._setup_parallel_dims()

    def _distributed_backend(self) -> str:
        fsdp_config = getattr(self.config, "fsdp", None)
        configured_backend = getattr(fsdp_config, "backend", "auto")
        if configured_backend == "auto":
            return "nccl" if torch.cuda.is_available() else "gloo"
        if configured_backend == "hybrid":
            if torch.cuda.is_available():
                return "cpu:gloo,cuda:nccl"
            return "gloo"
        return configured_backend

    def _setup_parallel_dims(self) -> None:
        fsdp_config = getattr(self.config, "fsdp", None)
        if self.world is None:
            raise RuntimeError("World must be set up before parallel dims")
        if fsdp_config is None or not fsdp_config.enabled:
            self.parallel_dims = ParallelDims(world_size=self.world.world_size)
            return
        self.parallel_dims = ParallelDims(
            dp_replicate=fsdp_config.dp_replicate,
            dp_shard=fsdp_config.dp_shard,
            cp=fsdp_config.cp,
            tp=fsdp_config.tp,
            ep=fsdp_config.ep,
            world_size=self.world.world_size,
        )
        if self.parallel_dims.cp_enabled:
            divisor = self.parallel_dims.seq_len_divisor
            if self.config.data.seq_len % divisor != 0:
                raise ValueError(
                    "data.seq_len must be divisible by 2 * fsdp.cp when context "
                    "parallelism is enabled."
                )

    def _setup_tokenizer(self) -> None:
        from wavelet.trainer.model import setup_tokenizer

        self.tokenizer = setup_tokenizer(self.config.model)

    def _setup_model(self) -> None:
        self._setup_model_standard()

    def _setup_model_standard(self) -> None:
        from wavelet.trainer.model import (
            apply_liger_kernel,
            apply_lora,
            prepare_hf_tp_lora_for_training,
            setup_model,
        )

        # Apply Liger kernel patches before from_pretrained so the class methods
        # are in place when model weights are loaded.
        apply_liger_kernel(self.config.loss_impl, self.config.model.name)
        fsdp_config = getattr(self.config, "fsdp", None)
        self._validate_model_execution_mode(fsdp_config)
        model = setup_model(
            self.config.model,
            max_seq_length=self.config.data.seq_len,
            distributed=bool(
                (self.world is not None and self.world.world_size > 1)
                or self.config.model.load_in_4bit
                or (fsdp_config is not None and fsdp_config.enabled)
                or (self.parallel_dims is not None and self.parallel_dims.tp_enabled)
            ),
            parallel_dims=self.parallel_dims,
        )
        # Cast LoRA adapters to match the model's compute dtype so flash
        # attention doesn't have to upcast fp32 LoRA outputs at every layer.
        # "auto" → align adapters to whatever dtype the base weights loaded as.
        cfg_dtype = self.config.model.torch_dtype
        lora_dtype = _lora_dtype(cfg_dtype)
        model = apply_lora(
            model,
            self.config.lora,
            lora_dtype=lora_dtype,
            match_base_dtype=(lora_dtype is None and cfg_dtype == "auto"),
        )
        prepare_hf_tp_lora_for_training(model, self.parallel_dims)
        # Activation offloading: intercepts ALL tensors saved for backward via
        # saved_tensors_hooks and streams them to pinned CPU RAM during forward,
        # fetching them back during backward. Matches TRL's activation_offloading=True.
        if self.config.activation_offloading is not None:
            from wavelet.trainer.optim import maybe_activation_offloading

            self.act_offload_ctx = maybe_activation_offloading(
                self.config.activation_offloading
            )
        self._apply_optional_model_kernels(model)
        self.model = self._wrap_distributed_model(model, fsdp_config)

    def _validate_model_execution_mode(self, fsdp_config: Any) -> None:
        if self.config.model.load_in_4bit and getattr(fsdp_config, "enabled", False):
            raise NotImplementedError(
                "QLoRA training uses replicated DDP in Wavelet. Disable FSDP "
                "for model.load_in_4bit=true."
            )
        if self.config.model.load_in_4bit and self._uses_sleep_colocation():
            raise NotImplementedError(
                "QLoRA does not support colocate_sleep yet because bitsandbytes "
                "4-bit modules cannot be moved between CPU and GPU."
            )

    def _apply_optional_model_kernels(self, model: PreTrainedModel) -> None:
        mconf = self.config.model
        if not any(
            (
                mconf.fused_lora_mlp,
                mconf.fused_lora_qkv,
                mconf.fused_lora_o,
                mconf.smart_gc,
            )
        ):
            return
        from wavelet.kernels.patch import (
            patch_fused_mlp,
            patch_fused_o,
            patch_fused_qkv,
            patch_smart_gc,
        )

        if mconf.fused_lora_mlp:
            patch_fused_mlp(model)
        if mconf.fused_lora_qkv:
            patch_fused_qkv(model)
        if mconf.fused_lora_o:
            patch_fused_o(model)
        if mconf.smart_gc:
            patch_smart_gc(model, seq_len=self.config.data.seq_len)

    def _wrap_distributed_model(
        self,
        model: PreTrainedModel,
        fsdp_config: Any,
    ) -> PreTrainedModel:
        from wavelet.trainer.model import maybe_wrap_ddp, maybe_wrap_fsdp

        if (
            self.world
            and (fsdp_config is None or not fsdp_config.enabled)
            and not (self.parallel_dims is not None and self.parallel_dims.tp_enabled)
            and not self.config.model.load_in_4bit
        ):
            model = model.to(self.world.device)
        tp_enabled = self.parallel_dims is not None and self.parallel_dims.tp_enabled
        fsdp_wrap_enabled = (
            fsdp_config is not None
            and fsdp_config.enabled
            and (
                self.parallel_dims is None
                or self.parallel_dims.fsdp_enabled
                or not tp_enabled
            )
        )
        if self.world and fsdp_wrap_enabled:
            model = maybe_wrap_fsdp(
                model,
                model_config=self.config.model,
                fsdp_config=fsdp_config,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        elif self.world and self.world.world_size > 1 and not tp_enabled:
            model = maybe_wrap_ddp(
                model,
                model_config=self.config.model,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        return model

    def _setup_data(self) -> None:
        from wavelet.data.sft import setup_dataset

        if not self.tokenizer:
            raise RuntimeError("Tokenizer must be set up before data")
        if not self.world:
            raise RuntimeError("World must be set up before data")

        data_rank, data_world_size = self._data_partition()
        self.dataset = setup_dataset(
            self.tokenizer,
            self.config.data,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )

    def _data_partition(self) -> tuple[int, int]:
        if self.world is None:
            raise RuntimeError("World must be set up before data partitioning")
        if self.parallel_dims is None:
            return self.world.rank, self.world.world_size

        data_world_size = self._data_parallel_world_size()
        data_rank = self._rank_in_pipeline_stage() // self._model_parallel_size()
        return data_rank, data_world_size

    def _is_data_parallel_metric_leader(self) -> bool:
        if self.world is None:
            raise RuntimeError("World must be set up before metric partitioning")
        if self.parallel_dims is None:
            return True

        return self._rank_in_pipeline_stage() % self._model_parallel_size() == 0

    def _data_parallel_world_size(self) -> int:
        if self.world is None:
            raise RuntimeError("World must be set up before data partitioning")
        if self.parallel_dims is None:
            return self.world.world_size
        return self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard

    def _model_parallel_size(self) -> int:
        if self.parallel_dims is None:
            return 1
        return self.parallel_dims.cp * self.parallel_dims.tp * self.parallel_dims.ep

    def _rank_in_pipeline_stage(self) -> int:
        if self.world is None:
            raise RuntimeError("World must be set up before data partitioning")
        ranks_per_pipeline_stage = (
            self._data_parallel_world_size() * self._model_parallel_size()
        )
        return self.world.rank % ranks_per_pipeline_stage

    def _setup_optimizer(self) -> None:
        from wavelet.trainer.model import enforce_single_lora_adapter
        from wavelet.trainer.optim import setup_optimizer

        if not self.model:
            raise RuntimeError("Model must be set up before optimizer")

        enforce_single_lora_adapter(self.model)
        self.optimizer = setup_optimizer(
            self.config.optim,
            self.model.named_parameters(),
        )

    def _setup_scheduler(self) -> None:
        from wavelet.trainer.optim import setup_scheduler

        if not self.optimizer:
            raise RuntimeError("Optimizer must be set up before scheduler")

        total_steps = self._compute_total_steps()
        self.scheduler = setup_scheduler(
            self.optimizer,
            self.config.scheduler,
            total_steps=total_steps,
            lr=self.config.optim.lr,
        )

    def prepare_for_training(self) -> None:
        if not self._uses_sleep_colocation():
            return
        if self.model is not None and self.world is not None:
            self.model.to(self.world.device)
            self.model.train()
        self._move_optimizer_state("cuda")
        self._flush_cuda_allocator()

    def offload_after_refit(self) -> None:
        if not self._uses_sleep_colocation():
            return
        if self.model is not None:
            self.model.to("cpu")
            self.model.eval()
        self._move_optimizer_state("cpu")
        self._flush_cuda_allocator()

    def _flush_cuda_allocator(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _uses_sleep_colocation(self) -> bool:
        launcher = getattr(self.config, "launcher", None)
        return getattr(launcher, "mode", None) == "colocate_sleep"

    def _move_optimizer_state(self, device: str) -> None:
        if self.optimizer is None:
            return
        target = torch.device(device)
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(target, non_blocking=target.type == "cuda")

    def _compute_total_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        return self.config.epochs * 1000

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.world is None:
            raise RuntimeError("World not set up")
        return {
            key: value.to(self.world.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _maybe_checkpoint(self) -> None:
        if self.ckpt_manager is None:
            return
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        self.ckpt_manager.poll_pending_save()
        did_save = self.ckpt_manager.save(
            TrainerState(step=self.step, micro_step=self._micro_step),
            dataloader=self._checkpoint_dataloader(),
        )
        if did_save:
            self.monitor.log_event(
                "checkpoint_triggered",
                step=self.step,
                payload={
                    "mode": self.config.ckpt.mode if self.config.ckpt else "disabled"
                },
            )

    def _get_lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def _clip_grad_norm(self) -> float:
        """Clip through the model wrapper when it owns gradient sharding."""
        if self.model is None:
            raise RuntimeError("Model not set up")
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

    def _require_finite_loss(self, loss: Tensor, *, label: str) -> None:
        """Abort every rank together when any rank observes a non-finite loss."""
        local_finite = bool(torch.isfinite(loss.detach()).all().item())
        all_finite = local_finite
        if torch.distributed.is_initialized():
            finite_flag = torch.tensor(
                int(local_finite),
                dtype=torch.int32,
                device=loss.device,
            )
            torch.distributed.all_reduce(
                finite_flag,
                op=torch.distributed.ReduceOp.MIN,
            )
            all_finite = bool(finite_flag.item())
        if all_finite:
            return
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        location = "this rank" if not local_finite else "another rank"
        raise FloatingPointError(
            f"Non-finite {label} detected on {location} at optimizer step "
            f"{self.step}; aborting before backward to keep ranks synchronized."
        )

    def _save_model(self) -> None:
        if self.world is None or self.model is None or self.tokenizer is None:
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
            logger.info("Model saved to %s", self.output_dir)


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

    def _train_step(self, batch: dict[str, Tensor]) -> TrainOutput:
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
                loss_output = LossOutput(loss=outputs.loss)
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=attn_mask,
                    position_ids=batch["position_ids"],
                )
                loss_output = self.compute_loss(outputs.logits, batch["labels"])
            loss = loss_output.loss

            self._require_finite_loss(loss, label="SFT loss")

            (loss / self.accumulation_steps).backward()

        self._micro_step += 1
        if self._micro_step % self.accumulation_steps == 0:
            sync_hf_tp_lora_replicated_grads(self.model, self.parallel_dims)
            if self.config.max_grad_norm > 0:
                self._clip_grad_norm()
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


import os
import sys
import time

from wavelet.configs.sft import SFTConfig
from wavelet.trainer.trainer import SFTTrainer
from wavelet.utils.config import load_config
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml


def _distributed_local_rank() -> int | None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None
    return int(os.environ.get("LOCAL_RANK", "0"))


def _wait_for_main_config(config_path, *, timeout_seconds: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not config_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for rank 0 to write '{config_path}'."
            )
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    config = load_config(SFTConfig, argv)
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    local_rank = _distributed_local_rank()
    config_path = get_config_dir(config.output_dir) / "sft.yaml"
    if local_rank in {None, 0}:
        validate_output_dir(
            config.output_dir,
            resuming=resuming,
            clean=config.clean_output_dir,
            protected_paths=(config.model.adapter_path,),
        )
        if resuming:
            assert config.ckpt is not None
            resolve_resume_checkpoint(
                config.checkpoint_output_dir,
                config.ckpt.resume_step,
            )
        dump_yaml(
            config_path,
            config.model_dump(mode="json", exclude_none=True),
        )
    else:
        _wait_for_main_config(config_path)

    if config.dry_run:
        print("Dry run - configuration loaded successfully")
        print(config.model_dump_json(indent=2))
        return 0

    trainer = SFTTrainer(config)
    trainer.setup()
    trainer.train()

    return 0

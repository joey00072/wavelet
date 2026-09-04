from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from wavelet.configs.sft import SFTConfig
from wavelet.data.sft import Example, load_records, setup_dataloader, setup_dataset
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
from wavelet.utils.config import load_config
from wavelet.utils.monitoring import RunMonitor, setup_config_logger
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml

if TYPE_CHECKING:
    from wavelet.configs.rl_config import RLConfig


logger = logging.getLogger(__name__)


def _lora_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return None


def _unwrap_dataset_state(state: object) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    nested = state.get("dataset")
    if isinstance(nested, dict):
        return _unwrap_dataset_state(nested)
    return state


def _dataloader_progress(
    dataloader: StatefulDataLoader,
    dataset: IterableDataset,
) -> dict[str, Any]:
    loader_state = dataloader.state_dict()
    snapshot = loader_state.get("_snapshot")
    states: list[dict[str, Any]] = []
    if isinstance(snapshot, dict):
        worker_snapshots = snapshot.get("_worker_snapshots")
        if isinstance(worker_snapshots, dict):
            for worker_snapshot in worker_snapshots.values():
                if not isinstance(worker_snapshot, dict):
                    continue
                state = _unwrap_dataset_state(worker_snapshot.get("dataset_state"))
                if state is not None:
                    states.append(state)
    else:
        state = _unwrap_dataset_state(loader_state.get("dataset_state"))
        if state is not None:
            states.append(state)

    if not states:
        stats_fn = getattr(dataset, "stats", None)
        stats = stats_fn() if callable(stats_fn) else {}
        base = getattr(dataset, "base", dataset)
        return {
            "step": int(getattr(base, "step", 0)),
            "epoch": int(getattr(base, "epoch", 0)),
            "num_samples": dict(stats.get("samples", {})),
            "num_tokens": dict(stats.get("tokens", {})),
        }

    num_samples: defaultdict[str, int] = defaultdict(int)
    num_tokens: defaultdict[str, int] = defaultdict(int)
    for state in states:
        for name, value in state.get("num_samples", {}).items():
            num_samples[str(name)] += int(value)
        for name, value in state.get("num_tokens", {}).items():
            num_tokens[str(name)] += int(value)
    return {
        "step": max(int(state.get("step", 0)) for state in states),
        "epoch": max(int(state.get("epoch", 0)) for state in states),
        "num_samples": dict(num_samples),
        "num_tokens": dict(num_tokens),
    }


class BaseTrainer:
    def __init__(self, config: SFTConfig | RLConfig) -> None:
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
            sample_history_size=config.samples.keep_last,
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
            self._before_train_loop()
            while self.step < target_step:
                for batch in self.dataloader:
                    output = self._train_step(self._prepare_batch(batch))
                    if not output.stepped:
                        continue
                    self._after_optimizer_step()
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

    def _before_train_loop(self) -> None:
        pass

    def _after_optimizer_step(self) -> None:
        pass

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
                    timeout=timedelta(seconds=self.config.dist_timeout_seconds),
                )
            else:
                with tempfile.NamedTemporaryFile(
                    prefix="wavelet-dist-", suffix=".init", delete=False
                ) as rendezvous_file:
                    rendezvous_path = rendezvous_file.name
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method=f"file://{rendezvous_path}",
                    rank=0,
                    world_size=1,
                    timeout=timedelta(seconds=self.config.dist_timeout_seconds),
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
            # torch.distributed.checkpoint.async_save requires a CPU backend in
            # the default group, so CUDA runs use the hybrid backend by default.
            return "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
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
        # EP reuses dp_shard/cp ranks for expert sharding, so it does not reduce
        # the number of data-parallel ranks.
        return self.parallel_dims.cp * self.parallel_dims.tp

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
        record_count = self._dataset_record_count()
        if record_count is None:
            raise ValueError(
                "max_steps is required when the dataset cannot report its record "
                "count; epochs alone cannot bound this run."
            )
        if self._dataset_is_packed():
            raise ValueError(
                "max_steps is required for packed datasets: packing changes the "
                "number of optimizer steps per epoch, so epochs alone cannot bound "
                "this run."
            )
        return max(
            math.ceil(record_count * self.config.epochs / self.config.data.batch_size),
            1,
        )

    def _dataset_record_count(self) -> int | None:
        dataset = self.dataset
        if dataset is None:
            return None
        records = getattr(dataset, "records", None)
        if records is None:
            records = getattr(getattr(dataset, "base", None), "records", None)
        if records is None:
            return None
        return len(records)

    def _dataset_is_packed(self) -> bool:
        return bool(getattr(self.config.data, "pack_sequences", False)) or (
            getattr(self.config.data, "pack_function", None) == "cat"
        )

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
        self.val_dataset: IterableDataset | None = None
        self.val_dataloader: StatefulDataLoader | None = None
        self._val_records: list[Example] | None = None
        self._validated_steps: set[int] = set()
        self._step_started_at: float | None = None
        self._step_model_tokens = 0

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
        if self.config.val is not None:
            self._val_records = load_records(self.config.val.data)
            self.val_dataloader = self._build_validation_dataloader()

    def _build_validation_dataloader(self) -> StatefulDataLoader:
        if self.config.val is None or self._val_records is None:
            raise RuntimeError("Validation data is not configured")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be set up before validation data")
        data_rank, data_world_size = self._data_partition()
        self.val_dataset = setup_dataset(
            self.tokenizer,
            self.config.val.data,
            data_rank=data_rank,
            data_world_size=data_world_size,
            records=self._val_records,
            max_epochs_per_iteration=1,
        )
        return setup_dataloader(
            self.val_dataset,
            self.config.val.data,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _before_train_loop(self) -> None:
        self._maybe_run_validation(step=self.step, before_training=True)

    def _after_optimizer_step(self) -> None:
        self._maybe_run_validation(step=self.step, before_training=False)

    def _maybe_run_validation(self, *, step: int, before_training: bool) -> None:
        val_config = self.config.val
        if val_config is None or step in self._validated_steps:
            return
        should_run = (before_training and step == 0 and val_config.eval_on_start) or (
            not before_training and step > 0 and step % val_config.interval == 0
        )
        if not should_run:
            return
        self._run_validation(step)
        self._validated_steps.add(step)

    def _run_validation(self, step: int) -> None:
        if self.model is None or self.world is None:
            raise RuntimeError("Model and world must be set up before validation")
        if self.monitor is None:
            raise RuntimeError("Monitor must be set up before validation")
        if self.val_dataloader is None:
            raise RuntimeError("Validation dataloader is not set up")

        was_training = self.model.training
        total_loss = torch.zeros((), dtype=torch.float32, device=self.world.device)
        total_tokens = torch.zeros((), dtype=torch.int64, device=self.world.device)
        nonfinite_batches = torch.zeros((), dtype=torch.int64, device=self.world.device)
        iterator = iter(self.val_dataloader)
        self.model.eval()
        try:
            with torch.no_grad():
                while True:
                    raw_batch = next(iterator, None)
                    has_batch = torch.tensor(
                        int(raw_batch is not None),
                        dtype=torch.int32,
                        device=self.world.device,
                    )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            has_batch, op=torch.distributed.ReduceOp.MIN
                        )
                    if not bool(has_batch.item()):
                        break
                    assert raw_batch is not None
                    batch = self._prepare_batch(raw_batch)
                    loss_output = self._forward_loss(batch)
                    token_count = (batch["labels"] != -100).sum()
                    if token_count.item() == 0:
                        continue
                    loss = loss_output.loss.detach()
                    if not torch.isfinite(loss).all():
                        nonfinite_batches += 1
                        continue
                    total_loss += loss.float() * token_count
                    total_tokens += token_count

            if torch.distributed.is_initialized():
                for value in (total_loss, total_tokens, nonfinite_batches):
                    torch.distributed.all_reduce(
                        value, op=torch.distributed.ReduceOp.SUM
                    )
            if nonfinite_batches.item() > 0:
                logger.warning(
                    "Validation at step %s skipped %s non-finite batches",
                    step,
                    nonfinite_batches.item(),
                )
            mean_loss = (
                float((total_loss / total_tokens).item())
                if total_tokens.item() > 0
                else float("nan")
            )
            if total_tokens.item() == 0:
                logger.warning(
                    "Validation at step %s had no finite trainable tokens", step
                )
            self.monitor.log({"val/loss": mean_loss}, step)
        finally:
            self.model.train(was_training)
        self.val_dataloader = self._build_validation_dataloader()

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
        metrics = dict(output.metrics)
        metrics.update(self._dataset_progress_metrics())
        metrics.update({"loss": loss, "lr": lr})
        self.monitor.log(metrics, self.step)
        progress.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

    def _train_step(self, batch: dict[str, Tensor]) -> TrainOutput:
        if self.model is None:
            raise RuntimeError("Model not set up")
        if self._step_started_at is None:
            self._step_started_at = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        self._step_model_tokens += int(batch["input_ids"].numel())

        with self.act_offload_ctx:
            loss_output = self._forward_loss(batch)
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
                metrics={
                    "loss": float(loss.detach().item()),
                    **self._finish_step_performance_metrics(),
                },
            )

        return TrainOutput(
            loss=loss_output,
            stepped=False,
            step=self.step,
            micro_step=self._micro_step,
        )

    def _finish_step_performance_metrics(self) -> dict[str, float]:
        if self._step_started_at is None:
            raise RuntimeError("SFT step timer was not started.")
        elapsed = max(time.perf_counter() - self._step_started_at, 1e-9)
        global_tokens = self._step_model_tokens * self._data_parallel_world_size()
        peak_memory_gib = (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        self._step_started_at = None
        self._step_model_tokens = 0
        return {
            "perf/tokens_per_second": global_tokens / elapsed,
            "perf/peak_memory_gib": peak_memory_gib,
        }

    def _dataset_progress_metrics(self) -> dict[str, float]:
        if self.dataset is None or self.dataloader is None:
            return {}
        progress = _dataloader_progress(self.dataloader, self.dataset)
        progress = self._sync_dataset_progress(progress)
        samples = progress["num_samples"]
        tokens = progress["num_tokens"]
        total_samples = sum(samples.values())
        total_tokens = sum(tokens.values())
        metrics = {"progress/epoch": float(progress["epoch"])}
        for source in sorted(set(samples) | set(tokens)):
            metrics[f"progress/{source}/ratio_samples"] = (
                samples.get(source, 0) / total_samples if total_samples else 0.0
            )
            metrics[f"progress/{source}/ratio_tokens"] = (
                tokens.get(source, 0) / total_tokens if total_tokens else 0.0
            )
        return metrics

    def _sync_dataset_progress(self, progress: dict[str, Any]) -> dict[str, Any]:
        if (
            self.world is None
            or self.world.world_size <= 1
            or not torch.distributed.is_initialized()
        ):
            return progress
        payload = progress if self._is_data_parallel_metric_leader() else None
        gathered: list[dict[str, Any] | None] = [None] * self.world.world_size
        torch.distributed.all_gather_object(gathered, payload)
        contributors = [item for item in gathered if item is not None]
        samples: defaultdict[str, int] = defaultdict(int)
        tokens: defaultdict[str, int] = defaultdict(int)
        for item in contributors:
            for name, value in item["num_samples"].items():
                samples[name] += int(value)
            for name, value in item["num_tokens"].items():
                tokens[name] += int(value)
        return {
            "step": max(int(item["step"]) for item in contributors),
            "epoch": max(int(item["epoch"]) for item in contributors),
            "num_samples": dict(samples),
            "num_tokens": dict(tokens),
        }

    def _forward_loss(self, batch: dict[str, Tensor]) -> LossOutput:
        if self.model is None:
            raise RuntimeError("Model not set up")

        # With cat packing all tokens are real (no padding), so attention_mask
        # is all-ones. Passing None lets transformers use is_causal=True which
        # enables the memory-efficient SDPA kernel instead of the O(L²) math backend.
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None and attn_mask.all():
            attn_mask = None

        if self.config.loss_impl == "liger_fused":
            # Wavelet's labels are pre-shifted, so shift_labels avoids a
            # second shift inside Liger's fused linear cross-entropy.
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attn_mask,
                position_ids=batch["position_ids"],
                shift_labels=batch["labels"],
            )
            return LossOutput(loss=outputs.loss)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            position_ids=batch["position_ids"],
        )
        return self.compute_loss(outputs.logits, batch["labels"])

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
            # Keep the zero attached to the graph so backward() still works.
            return LossOutput(loss=logits.sum() * 0.0)

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
        data_paths = (
            list(config.data.path)
            if isinstance(config.data.path, list)
            else [config.data.path]
        )
        validate_output_dir(
            config.output_dir,
            resuming=resuming,
            clean=config.clean_output_dir,
            protected_paths=(config.model.adapter_path, *data_paths),
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

    setup_config_logger("sft", config)
    trainer = SFTTrainer(config)
    trainer.setup()
    trainer.train()

    return 0

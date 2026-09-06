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
from collections.abc import Callable, Mapping, Sequence
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

from wavelet.configs.sft import FSDPConfig, SFTConfig
from wavelet.data.sft import Example, load_records, setup_dataloader, setup_dataset
from wavelet.kernels.patch import (
    patch_fused_mlp,
    patch_fused_o,
    patch_fused_qkv,
    patch_smart_gc,
)
from wavelet.monitor import RunMonitor, setup_config_logger
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.context_parallel import (
    context_parallel_batch,
    prepare_context_parallel_batch,
)
from wavelet.trainer.debug import DEBUG_MODEL_NAME
from wavelet.trainer.distributed import (
    ParallelDims,
    World,
    all_ranks_true,
    clip_grad_norm_across_meshes_,
    distributed_uses_cuda,
    get_world,
    set_world,
    world_is_distributed,
)
from wavelet.trainer.garbage_collection import DeterministicGarbageCollector
from wavelet.trainer.model import (
    TORCH_DTYPES,
    apply_activation_checkpointing,
    apply_liger_kernel,
    apply_lora,
    compile_transformer_layers,
    enforce_single_lora_adapter,
    export_model_for_save,
    load_fsdp2_model_from_hf,
    maybe_wrap_ddp,
    maybe_wrap_fsdp,
    prepare_hf_tp_lora_for_training,
    save_model,
    setup_model,
    setup_tokenizer,
    sync_hf_tp_lora_replicated_grads,
)
from wavelet.trainer.moe import configure_hf_moe_routers, moe_load_balance_metrics
from wavelet.trainer.optim import (
    OffloadActivations,
    enable_optimizer_state_offload,
    setup_optimizer,
    setup_scheduler,
)
from wavelet.trainer.perf import (
    estimate_training_flops_per_token,
    model_compute_dtype,
    training_flop_metrics,
)
from wavelet.trainer.profiling import CudaMemoryProfiler, StepProfiler
from wavelet.trainer.telemetry import (
    gather_rank_telemetry,
    node_metrics,
    rank_rows,
    sample_rank_telemetry,
)
from wavelet.trainer.types import LossOutput, TrainOutput
from wavelet.utils.config import load_config
from wavelet.utils.pathing import (
    get_config_dir,
    resolve_resume_checkpoint_source,
    validate_output_dir,
)
from wavelet.utils.serialization import dump_yaml

if TYPE_CHECKING:
    from wavelet.configs.rl_config import RLConfig


logger = logging.getLogger(__name__)


def _lora_dtype(
    model_dtype: str, optimization_dtype: str = "model"
) -> torch.dtype | None:
    dtype = model_dtype if optimization_dtype == "model" else optimization_dtype
    return TORCH_DTYPES.get(dtype)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _reduce_by_key(
    dicts: Sequence[Mapping[str, float]],
    reduce: Callable[[str, list[float]], float],
    *,
    keep: Callable[[str], bool] = lambda key: True,
    ordered: bool = False,
) -> dict[str, float]:
    """Apply ``reduce(key, values)`` to the values present for each key."""
    keys = {key for metrics in dicts for key in metrics if keep(key)}
    return {
        key: reduce(key, [float(metrics[key]) for metrics in dicts if key in metrics])
        for key in (sorted(keys) if ordered else keys)
    }


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

    if states:
        return _merge_progress_states(states)
    stats_fn = getattr(dataset, "stats", None)
    stats = stats_fn() if callable(stats_fn) else {}
    base = getattr(dataset, "base", dataset)
    return {
        "step": int(getattr(base, "step", 0)),
        "epoch": int(getattr(base, "epoch", 0)),
        "num_samples": dict(stats.get("samples", {})),
        "num_tokens": dict(stats.get("tokens", {})),
    }


def _merge_progress_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-worker/per-rank dataset progress into one progress record."""
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
        self.total_tokens = 0
        self.total_samples = 0
        self._garbage_collector: DeterministicGarbageCollector | None = None
        self._step_profiler: StepProfiler | None = None
        self._memory_profiler: CudaMemoryProfiler | None = None
        self._model_flops_per_token: int | None = None
        self._model_compute_dtype: torch.dtype | None = None
        self._step_model_tokens = 0
        self._latest_rank_telemetry: list[dict[str, Any]] | None = None
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

    def _require_world(self) -> World:
        if self.world is None:
            raise RuntimeError("World not set up. Call setup() first.")
        return self.world

    def _require_monitor(self) -> RunMonitor:
        if self.monitor is None:
            raise RuntimeError("Monitor not set up. Call setup() first.")
        return self.monitor

    def _after_resume(self) -> None:
        pass

    def _validate_resume_state(self, state: TrainerState) -> None:
        self._validate_progress_state(state)
        if state.micro_step != state.step * self.accumulation_steps:
            raise ValueError(
                "Checkpoint micro_step does not match the expected optimizer-step "
                "boundary for this trainer configuration."
            )

    @staticmethod
    def _validate_progress_state(state: TrainerState) -> None:
        if state.total_tokens < 0 or state.total_samples < 0:
            raise ValueError("Checkpoint progress counters must be non-negative.")

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
        checkpoint = self.config.ckpt
        if checkpoint is not None and checkpoint.is_resuming:
            self.resume_checkpoint_dir = resolve_resume_checkpoint_source(
                self.config.checkpoint_output_dir,
                resume_step=checkpoint.resume_step,
                resume_dir=checkpoint.resume_dir,
            )
            state = self.ckpt_manager.load(
                self.resume_checkpoint_dir,
                dataloader=(
                    None
                    if checkpoint.skip_dataloader
                    else self._checkpoint_dataloader()
                ),
                load_optimizer=not checkpoint.skip_optimizer,
                load_scheduler=not checkpoint.skip_scheduler,
                load_progress=not checkpoint.skip_progress,
            )
            if checkpoint.skip_scheduler:
                self._reset_scheduler(completed_steps=state.step)
            self._validate_resume_state(state)
            self.step = state.step
            self._micro_step = state.micro_step
            self.total_tokens = state.total_tokens
            self.total_samples = state.total_samples
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
        self._require_monitor()
        if self._run_closed:
            raise RuntimeError("Trainer run has already been finalized.")
        assert self.model is not None
        assert self.world is not None
        self._ensure_memory_profiler()
        self._ensure_garbage_collector()
        self.model.train()
        progress = tqdm(
            total=max(target_step - self.step, 0),
            disable=not self.world.is_main,
        )
        try:
            self._before_train_loop()
            while self.step < target_step:
                for batch in self.dataloader:
                    self._maybe_start_step_profiler(self.step + 1)
                    prepared_batch = self._prepare_batch(batch)
                    output = self._train_step(prepared_batch)
                    self._record_progress(prepared_batch)
                    if not output.stepped:
                        continue
                    self._after_optimizer_step()
                    self._maybe_collect_garbage()
                    self._maybe_dump_memory_snapshot(self.step)
                    self._maybe_finish_step_profiler(self.step)
                    self._log_train_output(output, progress)
                    self._maybe_checkpoint()
                    progress.update(1)
                    if self.step >= target_step:
                        break
            if self.ckpt_manager is not None:
                if finish_run:
                    self._save_final_checkpoint()
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
        monitor = self._require_monitor()
        self._close_step_profiler()
        self._close_memory_profiler()
        try:
            monitor.finish(status=status, step=self.step)
            self._run_closed = True
            if status == "completed":
                self._save_model()
        finally:
            self._close_garbage_collector()

    def _ensure_garbage_collector(self) -> None:
        if self.config.gc is None or self._garbage_collector is not None:
            return
        self._garbage_collector = DeterministicGarbageCollector(self.config.gc.interval)

    def _maybe_collect_garbage(self) -> None:
        if self._garbage_collector is not None:
            self._garbage_collector.run(self.step)

    def _close_garbage_collector(self) -> None:
        if self._garbage_collector is None:
            return
        self._garbage_collector.close()
        self._garbage_collector = None

    def _maybe_start_step_profiler(self, step: int) -> None:
        config = self.config.profiler
        if config is None:
            return
        if self._step_profiler is None:
            trace_path = config.trace_path or (
                self.output_dir
                / "profiler"
                / f"trace-{config.start_step}-{config.end_step}.json"
            )
            if world_is_distributed(self.world):
                trace_path = trace_path.with_name(
                    f"{trace_path.stem}.rank-{self.world.rank}{trace_path.suffix}"
                )
            self._step_profiler = StepProfiler(
                trace_path,
                start_step=config.start_step,
                end_step=config.end_step,
                record_shapes=config.record_shapes,
                profile_memory=config.profile_memory,
            )
        self._step_profiler.before_step(step)

    def _maybe_finish_step_profiler(self, step: int) -> None:
        if self._step_profiler is not None:
            self._step_profiler.after_step(step)

    def _close_step_profiler(self) -> None:
        if self._step_profiler is None:
            return
        self._step_profiler.close()
        self._step_profiler = None

    def _ensure_memory_profiler(self) -> None:
        config = self.config.memory_profiler
        if config is None or self._memory_profiler is not None:
            return
        self._memory_profiler = CudaMemoryProfiler(
            config.output_dir or (self.output_dir / "memory"),
            rank=self._require_world().rank,
            interval=config.interval,
            max_entries=config.max_entries,
        )

    def _maybe_dump_memory_snapshot(self, step: int) -> None:
        if self._memory_profiler is not None:
            self._memory_profiler.step(step)

    def _close_memory_profiler(self) -> None:
        if self._memory_profiler is None:
            return
        self._memory_profiler.close()
        self._memory_profiler = None

    def _setup_distributed(self) -> None:
        should_init_dist = (
            self.config.fsdp.enabled or int(os.environ.get("WORLD_SIZE", "1")) > 1
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
            self.world = get_world()
        else:
            self.world = World(
                rank=0,
                local_rank=0,
                world_size=1,
                local_world_size=1,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            set_world(self.world)
        self._setup_parallel_dims()

    def _distributed_backend(self) -> str:
        configured_backend = self.config.fsdp.backend
        if configured_backend in ("auto", "hybrid"):
            # torch.distributed.checkpoint.async_save requires a CPU backend in
            # the default group, so CUDA runs use the hybrid backend by default.
            return "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
        return configured_backend

    def _setup_parallel_dims(self) -> None:
        fsdp_config = self.config.fsdp
        world = self._require_world()
        if not fsdp_config.enabled:
            self.parallel_dims = ParallelDims(world_size=world.world_size)
            return
        self.parallel_dims = ParallelDims(
            dp_replicate=fsdp_config.dp_replicate,
            dp_shard=fsdp_config.dp_shard,
            cp=fsdp_config.cp,
            tp=fsdp_config.tp,
            ep=fsdp_config.ep,
            world_size=world.world_size,
        )
        if self.parallel_dims.cp_enabled:
            divisor = self.parallel_dims.seq_len_divisor
            if self.config.data.seq_len % divisor != 0:
                raise ValueError(
                    "data.seq_len must be divisible by 2 * fsdp.cp when context "
                    "parallelism is enabled."
                )

    def _setup_tokenizer(self) -> None:
        self.tokenizer = setup_tokenizer(self.config.model)

    def _setup_model(self) -> None:
        # Apply Liger kernel patches before from_pretrained so the class methods
        # are in place when model weights are loaded.
        apply_liger_kernel(self.config.loss_impl, self.config.model.name)
        fsdp_config = self.config.fsdp
        use_fsdp2_meta_init = self._use_fsdp2_meta_init(fsdp_config)
        self._validate_model_execution_mode(fsdp_config)
        model = setup_model(
            self.config.model,
            max_seq_length=self.config.data.seq_len,
            distributed=bool(
                world_is_distributed(self.world)
                or self.config.model.load_in_4bit
                or fsdp_config.enabled
                or self._tp_enabled()
            ),
            parallel_dims=self.parallel_dims,
            initialize_on_meta=use_fsdp2_meta_init,
        )
        # Keep the compatibility default aligned with the base model while allowing
        # explicit FP32 adapter parameters for higher-fidelity optimizer updates.
        # "auto" + "model" aligns to whatever dtype the base weights loaded as.
        cfg_dtype = self.config.model.torch_dtype
        lora_optimization_dtype = (
            self.config.lora.optimization_dtype
            if self.config.lora is not None
            else "model"
        )
        lora_dtype = _lora_dtype(cfg_dtype, lora_optimization_dtype)
        model = apply_lora(
            model,
            self.config.lora,
            lora_dtype=lora_dtype,
            match_base_dtype=(lora_dtype is None and cfg_dtype == "auto"),
        )
        configure_hf_moe_routers(model, self.config.model)
        self._model_flops_per_token = estimate_training_flops_per_token(
            model,
            seq_len=self.config.data.seq_len,
        )
        self._model_compute_dtype = model_compute_dtype(model)
        prepare_hf_tp_lora_for_training(model, self.parallel_dims)
        # Activation offloading: intercepts ALL tensors saved for backward via
        # saved_tensors_hooks and streams them to pinned CPU RAM during forward,
        # fetching them back during backward. Matches TRL's activation_offloading=True.
        if self.config.activation_offloading is not None:
            self.act_offload_ctx = OffloadActivations(
                use_pin_memory=self.config.activation_offloading.pin_memory
            )
        self._apply_optional_model_kernels(model)
        activation_checkpointing = self.config.model.activation_checkpointing
        if activation_checkpointing is not None and not self.config.model.smart_gc:
            apply_activation_checkpointing(model, activation_checkpointing)
        if self.config.model.compile:
            compile_transformer_layers(
                model,
                fullgraph=self.config.model.compile_fullgraph,
            )
        self.model = self._wrap_distributed_model(model, fsdp_config)
        if use_fsdp2_meta_init:
            load_fsdp2_model_from_hf(
                self.model,
                self.config.model,
                world=self._require_world(),
                cpu_offload=bool(fsdp_config.cpu_offload),
            )

    def _tp_enabled(self) -> bool:
        return self.parallel_dims is not None and self.parallel_dims.tp_enabled

    def _model_forward_context(self) -> contextlib.AbstractContextManager[None]:
        if self.world is None or self.world.device.type != "cuda":
            return contextlib.nullcontext()
        if self.config.model.torch_dtype == "float32":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (
            self.config.lora is not None
            and self.config.lora.optimization_dtype == "float32"
            and self.config.model.torch_dtype in {"bfloat16", "float16"}
        ):
            dtype = _lora_dtype(self.config.model.torch_dtype)
            assert dtype is not None
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _use_fsdp2_meta_init(self, fsdp_config: FSDPConfig) -> bool:
        if not (
            self.config.model.meta_device_init
            and fsdp_config.enabled
            and fsdp_config.impl == "fsdp2"
        ):
            return False
        if self.config.model.name == DEBUG_MODEL_NAME:
            logger.warning(
                "The random debug model has no Hugging Face weights to load; "
                "ignoring model.meta_device_init."
            )
            return False
        if self.config.model.adapter_path is not None:
            logger.warning(
                "FSDP2 meta initialization does not yet support resuming a PEFT "
                "adapter; using the standard Hugging Face load path."
            )
            return False
        if self._tp_enabled():
            logger.warning(
                "FSDP2 meta initialization with tensor parallelism is not yet "
                "validated; using the standard Hugging Face load path."
            )
            return False
        lora_config = self.config.lora
        if lora_config is not None and lora_config.modules_to_save:
            logger.warning(
                "FSDP2 meta initialization does not yet support LoRA "
                "modules_to_save; using the standard Hugging Face load path."
            )
            return False
        return True

    def _validate_model_execution_mode(self, fsdp_config: FSDPConfig) -> None:
        if self.config.model.load_in_4bit and fsdp_config.enabled:
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
        fsdp_config: FSDPConfig,
    ) -> PreTrainedModel:
        tp_enabled = self._tp_enabled()
        if (
            self.world
            and not fsdp_config.enabled
            and not tp_enabled
            and not self.config.model.load_in_4bit
        ):
            model = model.to(self.world.device)
        fsdp_wrap_enabled = fsdp_config.enabled and (
            self.parallel_dims is None
            or self.parallel_dims.fsdp_enabled
            or not tp_enabled
        )
        if self.world and fsdp_wrap_enabled:
            model = maybe_wrap_fsdp(
                model,
                model_config=self.config.model,
                fsdp_config=fsdp_config,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        elif world_is_distributed(self.world) and not tp_enabled:
            model = maybe_wrap_ddp(
                model,
                model_config=self.config.model,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        return model

    def _setup_data(self) -> None:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer must be set up before data")
        data_rank, data_world_size = self._data_partition()
        self.dataset = setup_dataset(
            self.tokenizer,
            self.config.data,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )

    def _data_partition(self) -> tuple[int, int]:
        world = self._require_world()
        if self.parallel_dims is None:
            return world.rank, world.world_size
        data_rank = self._rank_in_pipeline_stage() // self._model_parallel_size()
        return data_rank, self._data_parallel_world_size()

    def _is_data_parallel_metric_leader(self) -> bool:
        self._require_world()
        if self.parallel_dims is None:
            return True
        return self._rank_in_pipeline_stage() % self._model_parallel_size() == 0

    def _data_parallel_world_size(self) -> int:
        world = self._require_world()
        if self.parallel_dims is None:
            return world.world_size
        return self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard

    def _model_parallel_size(self) -> int:
        if self.parallel_dims is None:
            return 1
        # EP reuses dp_shard/cp ranks for expert sharding, so it does not reduce
        # the number of data-parallel ranks.
        return self.parallel_dims.cp * self.parallel_dims.tp

    def _rank_in_pipeline_stage(self) -> int:
        ranks_per_pipeline_stage = (
            self._data_parallel_world_size() * self._model_parallel_size()
        )
        return self._require_world().rank % ranks_per_pipeline_stage

    def _setup_optimizer(self) -> None:
        if not self.model:
            raise RuntimeError("Model must be set up before optimizer")

        enforce_single_lora_adapter(self.model)
        self.optimizer = setup_optimizer(
            self.config.optim,
            self.model.named_parameters(),
        )
        if self.config.optim.cpu_offload:
            enable_optimizer_state_offload(self.optimizer)

    def _setup_scheduler(self, *, total_steps: int | None = None) -> None:
        if not self.optimizer:
            raise RuntimeError("Optimizer must be set up before scheduler")

        if total_steps is None:
            total_steps = self._compute_total_steps()
        self.scheduler = setup_scheduler(
            self.optimizer,
            self.config.scheduler,
            total_steps=total_steps,
            lr=self.config.optim.lr,
        )

    def _reset_scheduler(self, *, completed_steps: int) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be set up before scheduler reset")
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.optim.lr
            group["initial_lr"] = self.config.optim.lr
        remaining_steps = max(self._compute_total_steps() - completed_steps, 1)
        self._setup_scheduler(total_steps=remaining_steps)
        if self.ckpt_manager is not None:
            self.ckpt_manager.scheduler = self.scheduler

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
        offloader = getattr(self.optimizer, "_wavelet_state_offloader", None)
        if offloader is not None:
            if device == "cpu":
                offloader.move_to_cpu()
            elif not self.config.optim.cpu_offload:
                offloader.move_to_parameters()
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
        device = self._require_world().device
        prepared = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        return prepare_context_parallel_batch(
            prepared,
            self.parallel_dims,
            configured_seq_len=self.config.data.seq_len,
        )

    def _context_parallel_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        extra_buffers: list[tuple[torch.Tensor, int]] | None = None,
    ) -> contextlib.AbstractContextManager:
        return context_parallel_batch(
            batch,
            self.parallel_dims,
            extra_buffers=extra_buffers,
        )

    def _maybe_checkpoint(self) -> None:
        if self.ckpt_manager is None:
            return
        monitor = self._require_monitor()
        self.ckpt_manager.poll_pending_save()
        did_save = self.ckpt_manager.save(
            self._trainer_state(),
            dataloader=self._checkpoint_dataloader(),
        )
        if did_save:
            monitor.log_event(
                "checkpoint_triggered",
                step=self.step,
                payload={
                    "mode": self.config.ckpt.mode if self.config.ckpt else "disabled"
                },
            )

    def _save_final_checkpoint(self) -> bool:
        if self.ckpt_manager is None:
            return False
        return self.ckpt_manager.save(
            self._trainer_state(),
            dataloader=self._checkpoint_dataloader(),
            force=True,
        )

    def _trainer_state(self) -> TrainerState:
        return TrainerState(
            step=self.step,
            micro_step=self._micro_step,
            total_tokens=self.total_tokens,
            total_samples=self.total_samples,
        )

    def _record_progress(self, batch: dict[str, torch.Tensor]) -> None:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return
        data_parallel_size = self._data_parallel_world_size()
        self.total_tokens += int(input_ids.numel()) * data_parallel_size
        sample_counts = batch.get("sample_counts")
        local_samples = (
            int(sample_counts.sum().item())
            if sample_counts is not None
            else int(input_ids.shape[0])
        )
        self.total_samples += local_samples * data_parallel_size

    def _progress_metrics(self) -> dict[str, float]:
        return {
            "progress/total_tokens": float(self.total_tokens),
            "progress/total_samples": float(self.total_samples),
        }

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
        elif self.parallel_dims is not None and self.parallel_dims.ep_enabled:
            clipped = clip_grad_norm_across_meshes_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
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
        if all_ranks_true(local_finite, device=loss.device):
            return
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        location = "this rank" if not local_finite else "another rank"
        raise FloatingPointError(
            f"Non-finite {label} detected on {location} at optimizer step "
            f"{self.step}; aborting before backward to keep ranks synchronized."
        )

    def _step_performance_metrics(
        self,
        *,
        local_tokens: int,
        elapsed: float,
        dtype: torch.dtype | None,
    ) -> dict[str, float]:
        """Throughput, MFU, and per-rank telemetry for one optimizer step."""
        elapsed = max(elapsed, 1e-9)
        global_tokens = local_tokens * self._data_parallel_world_size()
        peak_memory_gib = (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        world_size = self.world.world_size if self.world is not None else 1
        metrics = {
            "perf/tokens_per_second": global_tokens / elapsed,
            "perf/peak_memory_gib": peak_memory_gib,
        }
        metrics.update(
            training_flop_metrics(
                flops_per_token=self._model_flops_per_token,
                model_tokens=global_tokens,
                elapsed_seconds=elapsed,
                world_size=world_size,
                dtype=dtype,
            )
        )
        replication = max(world_size // max(self._data_parallel_world_size(), 1), 1)
        sample = sample_rank_telemetry(
            self.world,
            tokens=local_tokens,
            seconds=elapsed,
            replication=replication,
        )
        ranks = gather_rank_telemetry(sample, self.world)
        self._latest_rank_telemetry = rank_rows(ranks)
        metrics.update(node_metrics(ranks))
        return metrics

    def _save_model(self) -> None:
        if self.world is None or self.model is None or self.tokenizer is None:
            return
        saveable_model, state_dict = export_model_for_save(self.model)
        save_model(
            saveable_model,
            self.tokenizer,
            self.output_dir,
            state_dict=state_dict,
            is_main_process=self.world.is_main,
            parallel_dims=self.parallel_dims,
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
        self._sft_moe_metric_accum: list[dict[str, float]] = []

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
        if self.model is None:
            raise RuntimeError("Model must be set up before validation")
        device = self._require_world().device
        monitor = self._require_monitor()
        if self.val_dataloader is None:
            raise RuntimeError("Validation dataloader is not set up")

        was_training = self.model.training
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_tokens = torch.zeros((), dtype=torch.int64, device=device)
        nonfinite_batches = torch.zeros((), dtype=torch.int64, device=device)
        iterator = iter(self.val_dataloader)
        self.model.eval()
        try:
            with torch.no_grad():
                while True:
                    raw_batch = next(iterator, None)
                    if not all_ranks_true(raw_batch is not None, device=device):
                        break
                    assert raw_batch is not None
                    batch = self._prepare_batch(raw_batch)
                    with self._context_parallel_batch(batch):
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
            monitor.log({"val/loss": mean_loss}, step)
        finally:
            self.model.train(was_training)
        self.val_dataloader = self._build_validation_dataloader()

    def _setup_accumulation_steps(self) -> None:
        world_size = self._require_world().world_size
        global_micro_batch = self.config.data.micro_batch_size * world_size
        if self.config.data.batch_size % global_micro_batch != 0:
            raise ValueError(
                "SFT data.batch_size is the global optimizer batch size and must be "
                "divisible by data.micro_batch_size * world_size "
                f"({self.config.data.micro_batch_size} * {world_size})."
            )
        self.accumulation_steps = self.config.data.batch_size // global_micro_batch

    def _log_train_output(self, output: TrainOutput, progress: tqdm) -> None:
        monitor = self._require_monitor()
        if self.step % self.config.log.log_every != 0:
            return
        loss = output.loss.loss.item()
        lr = self._get_lr()
        metrics = dict(output.metrics)
        metrics.update(self._dataset_progress_metrics())
        metrics.update(self._progress_metrics())
        metrics.update({"loss": loss, "lr": lr})
        monitor.log(metrics, self.step, ranks=self._latest_rank_telemetry)
        progress.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

    def _train_step(self, batch: dict[str, Tensor]) -> TrainOutput:
        if self.model is None:
            raise RuntimeError("Model not set up")
        if self._step_started_at is None:
            self._step_started_at = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        self._step_model_tokens += int(batch["input_ids"].numel())

        with self._context_parallel_batch(batch), self.act_offload_ctx:
            loss_output = self._forward_loss(batch)
            loss = loss_output.loss

            self._require_finite_loss(loss, label="SFT loss")

            (loss / self.accumulation_steps).backward()

        self._sft_moe_metric_accum.append(
            {
                key: float(value.detach().item())
                for key, value in loss_output.metrics.items()
                if key.startswith("moe/")
            }
        )

        self._micro_step += 1
        if self._micro_step % self.accumulation_steps == 0:
            sync_hf_tp_lora_replicated_grads(self.model, self.parallel_dims)
            grad_norm = (
                self._clip_grad_norm() if self.config.max_grad_norm > 0 else None
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            moe_metrics = self._aggregate_sft_moe_metrics()
            return TrainOutput(
                loss=loss_output,
                stepped=True,
                metrics={
                    "loss": float(loss.detach().item()),
                    **(
                        {"optim/grad_norm": grad_norm}
                        if grad_norm is not None
                        else {}
                    ),
                    **moe_metrics,
                    **self._finish_step_performance_metrics(),
                },
            )

        return TrainOutput(
            loss=loss_output,
            stepped=False,
        )

    def _aggregate_sft_moe_metrics(self) -> dict[str, float]:
        aggregated = _reduce_by_key(
            self._sft_moe_metric_accum,
            lambda key, values: max(values) if key.endswith("/max") else _mean(values),
        )
        self._sft_moe_metric_accum.clear()
        return aggregated

    def _finish_step_performance_metrics(self) -> dict[str, float]:
        if self._step_started_at is None:
            raise RuntimeError("SFT step timer was not started.")
        metrics = self._step_performance_metrics(
            local_tokens=self._step_model_tokens,
            elapsed=time.perf_counter() - self._step_started_at,
            dtype=self._model_compute_dtype,
        )
        self._step_started_at = None
        self._step_model_tokens = 0
        return metrics

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
            not world_is_distributed(self.world)
            or not torch.distributed.is_initialized()
        ):
            return progress
        payload = progress if self._is_data_parallel_metric_leader() else None
        gathered: list[dict[str, Any] | None] = [None] * self.world.world_size
        torch.distributed.all_gather_object(gathered, payload)
        return _merge_progress_states([item for item in gathered if item is not None])

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
            with self._model_forward_context():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=attn_mask,
                    position_ids=batch["position_ids"],
                    shift_labels=batch["labels"],
                )
            return LossOutput(
                loss=outputs.loss,
                metrics=moe_load_balance_metrics(
                    self.model,
                    outputs,
                    token_mask=batch.get("attention_mask"),
                ),
            )
        with self._model_forward_context():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attn_mask,
                position_ids=batch["position_ids"],
            )
        result = self.compute_loss(outputs.logits, batch["labels"])
        result.metrics.update(
            moe_load_balance_metrics(
                self.model,
                outputs,
                token_mask=batch.get("attention_mask"),
            )
        )
        return result

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
    resuming = config.ckpt is not None and config.ckpt.is_resuming
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
            resolve_resume_checkpoint_source(
                config.checkpoint_output_dir,
                resume_step=config.ckpt.resume_step,
                resume_dir=config.ckpt.resume_dir,
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

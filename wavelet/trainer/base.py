from __future__ import annotations

import contextlib
import os
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from wavelet.configs.sft import SFTConfig
from wavelet.distributed.parallel_dims import ParallelDims
from wavelet.distributed.world import World, distributed_uses_cuda, get_world, set_world

if TYPE_CHECKING:
    from wavelet.configs.rl_config import RLConfig


class BaseTrainer:
    def __init__(self, config: SFTConfig | "RLConfig") -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self.dataset: IterableDataset | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.world: World | None = None
        self.parallel_dims: ParallelDims | None = None
        self.output_dir = Path(config.output_dir)
        self.act_offload_ctx: contextlib.AbstractContextManager = (
            contextlib.nullcontext()
        )

    def setup(self) -> None:
        self._setup_seed()
        self._setup_distributed()
        self._setup_tokenizer()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_seed(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
            maybe_wrap_ddp,
            maybe_wrap_fsdp,
            setup_model,
        )

        # Apply Liger kernel patches before from_pretrained so the class methods
        # are in place when model weights are loaded.
        apply_liger_kernel(self.config.loss_impl, self.config.model.name)
        fsdp_config = getattr(self.config, "fsdp", None)
        model = setup_model(
            self.config.model,
            max_seq_length=self.config.data.seq_len,
            distributed=bool(
                (fsdp_config is not None and fsdp_config.enabled)
                or (self.parallel_dims is not None and self.parallel_dims.tp_enabled)
            ),
            parallel_dims=self.parallel_dims,
        )
        if (
            self.parallel_dims is not None
            and self.parallel_dims.tp_enabled
            and self.config.lora is not None
        ):
            raise NotImplementedError(
                "Tensor parallel training with LoRA/QLoRA adapters is not "
                "implemented yet."
            )
        # Cast LoRA adapters to match the model's compute dtype so flash
        # attention doesn't have to upcast fp32 LoRA outputs at every layer.
        # "auto" → align adapters to whatever dtype the base weights loaded as.
        cfg_dtype = self.config.model.torch_dtype
        if cfg_dtype == "bfloat16":
            lora_dtype: torch.dtype | None = torch.bfloat16
        elif cfg_dtype == "float16":
            lora_dtype = torch.float16
        else:
            lora_dtype = None  # "auto"/"float32": use match_base_dtype instead
        model = apply_lora(
            model,
            self.config.lora,
            lora_dtype=lora_dtype,
            match_base_dtype=(lora_dtype is None and cfg_dtype == "auto"),
        )
        # Activation offloading: intercepts ALL tensors saved for backward via
        # saved_tensors_hooks and streams them to pinned CPU RAM during forward,
        # fetching them back during backward. Matches TRL's activation_offloading=True.
        if self.config.activation_offloading is not None:
            from wavelet.utils.act_offloading import maybe_activation_offloading

            self.act_offload_ctx = maybe_activation_offloading(
                self.config.activation_offloading
            )
        # Optional kernel patches on the standard backend (post-PEFT).
        mconf = self.config.model
        if (
            mconf.fused_lora_mlp
            or mconf.fused_lora_qkv
            or mconf.fused_lora_o
            or mconf.smart_gc
        ):
            from wavelet.kernels.patch import (
                patch_fused_mlp,
                patch_fused_qkv,
                patch_fused_o,
                patch_smart_gc,
            )

            if mconf.fused_lora_mlp:
                patch_fused_mlp(model)
            if mconf.fused_lora_qkv:
                patch_fused_qkv(model)
            # patch_fused_o is standalone; safe to combine with patch_fused_qkv
            # (o_proj.forward becomes the fused closure, called normally by both paths)
            if mconf.fused_lora_o:
                patch_fused_o(model)
            if mconf.smart_gc:
                patch_smart_gc(model, seq_len=self.config.data.seq_len)
        if (
            self.world
            and (fsdp_config is None or not fsdp_config.enabled)
            and not (self.parallel_dims is not None and self.parallel_dims.tp_enabled)
        ):
            model = model.to(self.world.device)
        if self.world and fsdp_config is not None and fsdp_config.enabled:
            model = maybe_wrap_fsdp(
                model,
                model_config=self.config.model,
                fsdp_config=fsdp_config,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        elif self.world and self.world.world_size > 1:
            model = maybe_wrap_ddp(
                model,
                world=self.world,
                parallel_dims=self.parallel_dims,
            )
        self.model = model

    def _setup_data(self) -> None:
        from wavelet.data.dataset import setup_dataset

        if not self.tokenizer:
            raise RuntimeError("Tokenizer must be set up before data")
        if not self.world:
            raise RuntimeError("World must be set up before data")

        self.dataset = setup_dataset(
            self.tokenizer,
            self.config.data,
            data_rank=self.world.rank,
            data_world_size=self.world.world_size,
        )

    def _setup_optimizer(self) -> None:
        from wavelet.trainer.optim import setup_optimizer

        if not self.model:
            raise RuntimeError("Model must be set up before optimizer")

        self.optimizer = setup_optimizer(
            self.config.optim,
            self.model.named_parameters(),
        )

    def _setup_scheduler(self) -> None:
        from wavelet.trainer.scheduler import setup_scheduler

        if not self.optimizer:
            raise RuntimeError("Optimizer must be set up before scheduler")

        total_steps = self._compute_total_steps()
        self.scheduler = setup_scheduler(
            self.optimizer,
            self.config.scheduler,
            total_steps=total_steps,
            lr=self.config.optim.lr,
        )

    def _compute_total_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        return self.config.epochs * 1000

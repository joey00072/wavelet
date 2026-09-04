from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchdata.stateful_dataloader import StatefulDataLoader

from wavelet.configs.sft import CheckpointConfig
from wavelet.trainer.distributed import World, barrier, distributed_uses_cuda
from wavelet.utils.pathing import (
    STABLE_CHECKPOINT_MARKER,
    get_checkpoint_dir,
    is_stable_checkpoint,
    list_checkpoint_steps,
    resolve_resume_checkpoint,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


class AppState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, [self.optimizer]
        )
        state_dict: dict[str, Any] = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            [self.optimizer],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])


@dataclass(slots=True)
class TrainerState:
    step: int
    micro_step: int


@dataclass(slots=True)
class PendingAsyncSave:
    step: int
    checkpoint_dir: Path
    meta: dict[str, Any]
    response: AsyncSaveResponse | Future[Any]
    stager: DefaultStager | None


class CheckpointManager:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        config: CheckpointConfig | None,
        output_dir: Path,
        world: World,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = output_dir
        self.world = world
        self.pending_save: PendingAsyncSave | None = None

    def resolve_resume_checkpoint(self, resume_step: int) -> Path:
        return resolve_resume_checkpoint(self.output_dir, resume_step)

    def save(
        self,
        trainer_state: TrainerState,
        *,
        dataloader: StatefulDataLoader | None = None,
        force: bool = False,
    ) -> bool:
        if self.config is None or self.config.mode == "disabled":
            return False
        if self.config.interval is None:
            return False
        if not force and trainer_state.step % self.config.interval != 0:
            return False

        checkpoint_dir = get_checkpoint_dir(self.output_dir, trainer_state.step)
        if force and (
            is_stable_checkpoint(checkpoint_dir)
            or (
                self.pending_save is not None
                and self.pending_save.checkpoint_dir == checkpoint_dir
            )
        ):
            return False

        self.wait_for_pending_save()

        self._reset_checkpoint_dir(checkpoint_dir)
        self._save_dataloader_state(checkpoint_dir, dataloader)
        meta = self._build_meta(trainer_state)

        trainer_dir = checkpoint_dir / "trainer"
        writer = FileSystemWriter(trainer_dir, overwrite=True)
        state_dict = {"app": AppState(self.model, self.optimizer, self.scheduler)}
        no_dist = not torch.distributed.is_initialized()

        if self.config.mode == "async":
            stager = self._build_async_stager(use_pinned_memory=False)
            response = dcp.async_save(
                state_dict=state_dict,
                storage_writer=writer,
                async_checkpointer_type=AsyncCheckpointerType.THREAD,
                async_stager=stager,
                no_dist=no_dist,
            )
            self.pending_save = PendingAsyncSave(
                step=trainer_state.step,
                checkpoint_dir=checkpoint_dir,
                meta=meta,
                response=response,
                stager=stager,
            )
            self._wait_for_staging(response)
            return True

        if self.config.mode == "async_with_pinned_mem":
            stager = self._build_async_stager(use_pinned_memory=True)
            response = dcp.async_save(
                state_dict=state_dict,
                storage_writer=writer,
                async_checkpointer_type=AsyncCheckpointerType.THREAD,
                async_stager=stager,
                no_dist=no_dist,
            )
            self.pending_save = PendingAsyncSave(
                step=trainer_state.step,
                checkpoint_dir=checkpoint_dir,
                meta=meta,
                response=response,
                stager=stager,
            )
            self._wait_for_staging(response)
            return True

        dcp.save(
            state_dict=state_dict,
            storage_writer=writer,
            no_dist=no_dist,
        )
        self._finalize_checkpoint(checkpoint_dir, meta)
        self._maybe_clean()
        return True

    def load(
        self,
        checkpoint_dir: Path,
        *,
        dataloader: StatefulDataLoader | None = None,
    ) -> TrainerState:
        meta_path = checkpoint_dir / "meta.json"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_dir}'.")
        if not is_stable_checkpoint(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' exists but is not stable."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' is missing metadata."
            )

        metadata = json.loads(meta_path.read_text())
        if int(metadata.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version in '{checkpoint_dir.name}'."
            )
        if int(metadata["world_size"]) != self.world.world_size:
            raise ValueError(
                "Checkpoint world_size does not match the current runtime world_size."
            )

        trainer_dir = checkpoint_dir / "trainer"
        if not trainer_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' is missing trainer state."
            )

        state_dict = {"app": AppState(self.model, self.optimizer, self.scheduler)}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=trainer_dir,
            no_dist=not torch.distributed.is_initialized(),
        )

        self._load_dataloader_state(checkpoint_dir, dataloader)
        return TrainerState(
            step=int(metadata["step"]),
            micro_step=int(metadata["micro_step"]),
        )

    def poll_pending_save(self) -> None:
        self._maybe_finalize_pending(block=False)

    def wait_for_pending_save(self) -> None:
        self._maybe_finalize_pending(block=True)

    def _maybe_finalize_pending(self, *, block: bool) -> None:
        pending = self.pending_save
        if pending is None:
            return

        if not block and not self._all_ranks_done(pending.response):
            return

        try:
            self._wait_for_response(pending.response)
            self._finalize_checkpoint(pending.checkpoint_dir, pending.meta)
            self._maybe_clean()
        finally:
            if pending.stager is not None:
                pending.stager.close()
            self.pending_save = None

    def _build_meta(self, trainer_state: TrainerState) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "step": trainer_state.step,
            "micro_step": trainer_state.micro_step,
            "world_size": self.world.world_size,
            "mode": self.config.mode if self.config is not None else "disabled",
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _save_dataloader_state(
        self,
        checkpoint_dir: Path,
        dataloader: StatefulDataLoader | None,
    ) -> None:
        if dataloader is None:
            return
        dataloader_dir = checkpoint_dir / "dataloader"
        dataloader_dir.mkdir(parents=True, exist_ok=True)
        path = dataloader_dir / f"rank_{self.world.rank}.pt"
        torch.save(dataloader.state_dict(), path)

    def _load_dataloader_state(
        self,
        checkpoint_dir: Path,
        dataloader: StatefulDataLoader | None,
    ) -> None:
        if dataloader is None:
            return
        path = checkpoint_dir / "dataloader" / f"rank_{self.world.rank}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_dir.name}' is missing dataloader state for "
                f"rank {self.world.rank}."
            )
        dataloader.load_state_dict(torch.load(path, map_location="cpu"))

    def _finalize_checkpoint(
        self,
        checkpoint_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        if self.world.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "meta.json").write_text(json.dumps(meta))
            (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
        self._barrier_if_distributed()

    def _reset_checkpoint_dir(self, checkpoint_dir: Path) -> None:
        if self.world.is_main:
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier_if_distributed()

    def _maybe_clean(self) -> None:
        if (
            self.config is None
            or self.config.keep_last is None
            or self.config.keep_last < 1
        ):
            return
        if not self.world.is_main:
            return
        steps = list_checkpoint_steps(self.output_dir, stable_only=True)
        recent_steps = set(steps[-self.config.keep_last :])
        permanent_steps = (
            {step for step in steps if step % self.config.keep_interval == 0}
            if self.config.keep_interval is not None
            else set()
        )
        for step in steps:
            if step in recent_steps or step in permanent_steps:
                continue
            checkpoint_dir = get_checkpoint_dir(self.output_dir, step)
            if (
                self.pending_save is not None
                and checkpoint_dir == self.pending_save.checkpoint_dir
            ):
                continue
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

    def _is_response_done(
        self,
        response: AsyncSaveResponse | Future[Any],
    ) -> bool:
        if isinstance(response, Future):
            return response.done()
        return response.upload_completion.done()

    def _all_ranks_done(
        self,
        response: AsyncSaveResponse | Future[Any],
    ) -> bool:
        """Agree across ranks whether the async save finished.

        Finalization runs a barrier, so every rank must take the same branch;
        a rank that observes its own upload as done may not assume the others
        have finished too.
        """
        done = self._is_response_done(response)
        if not torch.distributed.is_initialized():
            return done
        device = torch.device("cpu")
        if distributed_uses_cuda() and self.world.device.type == "cuda":
            device = self.world.device
        flag = torch.tensor(int(done), dtype=torch.int64, device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _wait_for_response(
        self,
        response: AsyncSaveResponse | Future[Any],
    ) -> None:
        if isinstance(response, Future):
            response.result()
            return
        response.staging_completion.result()
        response.upload_completion.result()

    @staticmethod
    def _wait_for_staging(response: AsyncSaveResponse | Future[Any]) -> None:
        """Block until the live tensors have been copied to the staging buffer.

        The upload keeps running in the background, but the next optimizer step
        may not mutate parameters or optimizer state while staging still reads
        them.
        """
        if isinstance(response, Future):
            return
        response.staging_completion.result()

    def _build_async_stager(self, *, use_pinned_memory: bool) -> DefaultStager:
        accelerator_available = bool(torch.accelerator.is_available())
        return DefaultStager(
            StagingOptions(
                use_pinned_memory=use_pinned_memory and accelerator_available,
                # async_save uses a thread checkpointer, so IPC-backed tensor
                # storage only burns one POSIX shared-memory FD per storage.
                use_shared_memory=False,
                use_async_staging=True,
                use_non_blocking_copy=accelerator_available,
            )
        )

    def _barrier_if_distributed(self) -> None:
        if torch.distributed.is_initialized():
            barrier(self.world)

"""Trainer-owned policy publication and weight-export mechanics."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep

import torch
from peft import PeftModel

from wavelet.contracts.policy_metadata import adapter_artifact_metadata, policy_metadata
from wavelet.contracts.schedule import retained_policy_snapshots
from wavelet.trainer.distributed import barrier
from wavelet.trainer.export_tensors import _iter_layer_state_dicts, broadcast_model
from wavelet.trainer.model import (
    export_model_for_save,
    is_fsdp_model,
    save_lora_adapter_snapshot,
    save_lora_adapter_snapshot_from_fsdp,
    save_model,
)
from wavelet.transport.rollouts.filesystem import (
    POLICY_META_FILENAME,
    STABLE_BATCH_MARKER,
    get_policy_step_dir,
    record_policy_export_event,
    resolve_policy_dir,
    utc_now,
)
from wavelet.transport.weights.handshake import (
    NCCL_READY_MARKER,
    NCCL_UPDATE_INFO_FILENAME,
)
from wavelet.transport.weights.nccl import (
    NCCLWeightBroadcaster,
    _is_reusable_policy_snapshot,
    nccl_world_size,
    prune_policy_snapshots,
    prune_policy_snapshots_beyond,
)


class PolicyExporter:
    """Trainer-side filesystem and NCCL policy publication mechanics.

    Reads a fixed set of attributes off `trainer` (exposed below as explicit
    read-only properties) instead of forwarding every attribute lookup, so a
    typo fails at the read site with a normal `AttributeError` on a named
    property rather than silently reaching into an unrelated object.
    """

    def __init__(self, trainer: object) -> None:
        self._trainer = trainer

    @property
    def config(self) -> object:
        return self._trainer.config

    @property
    def world(self) -> object:
        return self._trainer.world

    @property
    def model(self) -> object:
        return self._trainer.model

    @property
    def step(self) -> int:
        return self._trainer.step

    @property
    def tokenizer(self) -> object:
        return self._trainer.tokenizer

    @property
    def output_dir(self) -> object:
        return self._trainer.output_dir

    @property
    def parallel_dims(self) -> object:
        return self._trainer.parallel_dims

    def offload_after_refit(self) -> None:
        self._trainer.offload_after_refit()

    def _init_policy_transport(self) -> None:
        self._nccl_broadcaster_executor: ThreadPoolExecutor | None = None
        self._nccl_broadcaster_future: Future[NCCLWeightBroadcaster] | None = None

    def _close_policy_transport(self) -> None:
        if self._nccl_broadcaster_executor is not None:
            self._nccl_broadcaster_executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
            self._nccl_broadcaster_executor = None

    def _barrier(self) -> None:
        if self.world.world_size > 1:
            barrier(self.world)

    def should_export_policy(self, step: int) -> bool:
        if step == 0:
            return self.config.policy_transfer.export_initial
        return step % self.config.policy_transfer.export_every_steps == 0

    def export_policy(
        self,
        *,
        step: int | None = None,
        force: bool = False,
    ) -> Path | None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")

        export_step = self.step if step is None else step
        if not force and not self.should_export_policy(export_step):
            return None
        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        if force:
            # A resumed run must not leave newer snapshots from the crashed run
            # visible; inference would otherwise load a policy the trainer never
            # produced (and, for NCCL, wait on a handshake nobody completes).
            if self.world.is_main:
                prune_policy_snapshots_beyond(policy_dir, step=export_step)
            self._barrier()
        if self.config.policy_transfer.type == "nccl":
            return self._export_nccl_policy(export_step)

        step_dir = get_policy_step_dir(policy_dir, export_step)
        if force:
            expected_kind = (
                "adapter"
                if self.config.lora is not None
                and self.config.policy_transfer.lightweight_lora
                else "model"
            )
            if _is_reusable_policy_snapshot(
                step_dir,
                step=export_step,
                expected_kind=expected_kind,
            ):
                self.offload_after_refit()
                return step_dir
        elif (step_dir / STABLE_BATCH_MARKER).is_file():
            raise FileExistsError(
                f"Stable policy step {export_step} already exists at '{step_dir}'."
            )
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")
        self._prepare_export_directory(tmp_dir, step_dir)
        self._barrier()
        saved_path = self._save_filesystem_policy(tmp_dir)
        if self.world.is_main:
            self._write_policy_metadata(
                tmp_dir,
                export_step=export_step,
                kind=saved_path.name,
            )
        self.offload_after_refit()
        self._publish_export_directory(
            tmp_dir,
            step_dir,
            export_step=export_step,
        )
        return step_dir

    def _save_filesystem_policy(self, tmp_dir: Path) -> Path:
        if (
            self.config.lora is not None
            and self.config.policy_transfer.lightweight_lora
            and is_fsdp_model(self.model)
        ):
            return save_lora_adapter_snapshot_from_fsdp(
                self.model,
                tmp_dir,
                is_main_process=self.world.is_main,
                parallel_dims=self.parallel_dims,
            )
        export_dtype = torch.bfloat16 if self.config.lora is None else None
        export_model, state_dict = export_model_for_save(
            self.model,
            state_dict_dtype=export_dtype,
        )
        if self.config.policy_transfer.lightweight_lora and isinstance(
            export_model, PeftModel
        ):
            return save_lora_adapter_snapshot(
                export_model,
                tmp_dir,
                state_dict=state_dict,
                is_main_process=self.world.is_main,
                parallel_dims=self.parallel_dims,
            )
        return save_model(
            export_model,
            self.tokenizer,
            tmp_dir,
            state_dict=state_dict,
            is_main_process=self.world.is_main,
            parallel_dims=self.parallel_dims,
        )

    def _prepare_export_directory(self, tmp_dir: Path, step_dir: Path) -> None:
        if not self.world.is_main:
            return
        for path in (tmp_dir, step_dir):
            if path.exists():
                shutil.rmtree(path)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def _write_policy_metadata(
        self,
        tmp_dir: Path,
        *,
        export_step: int,
        kind: str,
    ) -> None:
        artifact = adapter_artifact_metadata(tmp_dir / "adapter")
        metadata = policy_metadata(
            config=self.config,
            format_version=1,
            step=export_step,
            kind=kind,
            created_at=utc_now(),
            extra={"artifact": artifact} if artifact is not None else None,
        )
        (tmp_dir / POLICY_META_FILENAME).write_text(json.dumps(metadata))

    def _publish_export_directory(
        self,
        tmp_dir: Path,
        step_dir: Path,
        *,
        export_step: int,
    ) -> None:
        self._barrier()
        if self.world.is_main:
            (tmp_dir / STABLE_BATCH_MARKER).touch()
            tmp_dir.replace(step_dir)
            record_policy_export_event(self.config.output_dir, export_step)
        self._barrier()
        if self.world.is_main:
            prune_policy_snapshots(
                step_dir.parent,
                keep_last=retained_policy_snapshots(self.config),
            )

    def _export_nccl_policy(self, export_step: int) -> Path:
        if self.model is None:
            raise RuntimeError("Trainer not set up. Call setup() first.")
        if self.world is None:
            raise RuntimeError("World not set up")
        if self.config.lora is not None:
            raise NotImplementedError(
                "NCCL policy transfer is only implemented for full-model updates. "
                "Use filesystem policy transfer for LoRA adapters."
            )

        policy_dir = resolve_policy_dir(self.output_dir, self.config.policy_transfer)
        step_dir = get_policy_step_dir(policy_dir, export_step)
        tmp_dir = step_dir.with_name(f".{step_dir.name}.tmp")
        self._prepare_export_directory(tmp_dir, step_dir)
        if self.world.is_main:
            self._write_nccl_export(
                tmp_dir,
                step_dir,
                export_step=export_step,
            )

        self._broadcast_nccl_export(
            step_dir,
            export_step=export_step,
        )
        self.offload_after_refit()
        self._barrier()
        if self.world.is_main:
            record_policy_export_event(self.config.output_dir, export_step)
        self._barrier()
        return step_dir

    def _write_nccl_export(
        self,
        tmp_dir: Path,
        step_dir: Path,
        *,
        export_step: int,
    ) -> None:
        update_info = {"protocol": "layerwise_v1"}
        (tmp_dir / NCCL_UPDATE_INFO_FILENAME).write_text(json.dumps(update_info))
        self._write_policy_metadata(
            tmp_dir,
            export_step=export_step,
            kind="nccl",
        )
        (tmp_dir / STABLE_BATCH_MARKER).touch()
        tmp_dir.replace(step_dir)
        self._start_nccl_broadcaster()

    def _broadcast_nccl_export(
        self,
        step_dir: Path,
        *,
        export_step: int,
    ) -> None:
        broadcaster = None
        if self.world.is_main:
            self._wait_for_nccl_ready(step_dir)
            broadcaster = self._nccl_broadcaster()
        self._barrier()
        if broadcaster is not None:
            broadcast_model(broadcaster, self.model)
        else:
            # Non-source ranks still drive every layer gather so the FSDP
            # collectives stay in lockstep with the broadcasting rank.
            for _ in _iter_layer_state_dicts(self.model, self._nccl_wire_dtype()):
                pass

    def _nccl_wire_dtype(self) -> torch.dtype | None:
        name = self.config.policy_transfer.nccl_dtype
        return None if name == "model" else getattr(torch, name)

    def _wait_for_nccl_ready(self, step_dir: Path) -> None:
        ready_path = step_dir / NCCL_READY_MARKER
        deadline = monotonic() + self.config.policy_transfer.nccl_timeout_seconds
        while monotonic() < deadline:
            if ready_path.exists():
                return
            sleep(0.1)
        raise TimeoutError(
            "Timed out waiting for inference workers to enter NCCL policy update "
            f"for {step_dir}."
        )

    def _start_nccl_broadcaster(self) -> None:
        if self._nccl_broadcaster_future is not None:
            return
        if self._nccl_broadcaster_executor is None:
            self._nccl_broadcaster_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="wavelet-nccl-broadcaster",
            )
        self._nccl_broadcaster_future = self._nccl_broadcaster_executor.submit(
            self._create_nccl_broadcaster
        )

    def _nccl_broadcaster(self) -> NCCLWeightBroadcaster:
        if self._nccl_broadcaster_future is None:
            self._start_nccl_broadcaster()
        assert self._nccl_broadcaster_future is not None
        return self._nccl_broadcaster_future.result(
            timeout=self.config.policy_transfer.nccl_timeout_seconds
        )

    def _create_nccl_broadcaster(self) -> NCCLWeightBroadcaster:
        device = (
            self.world.device
            if self.world is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        return NCCLWeightBroadcaster(
            host=self.config.policy_transfer.nccl_host,
            port=self.config.policy_transfer.nccl_port,
            rank=0,
            world_size=nccl_world_size(
                self.config.policy_transfer.nccl_inference_world_size
            ),
            device=device,
            timeout_seconds=self.config.policy_transfer.nccl_timeout_seconds,
            dtype=self._nccl_wire_dtype(),
        )

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from wavelet.configs.config import CheckpointConfig
from wavelet.configs.rl_config import RLConfig
from wavelet.kernels.lora import LoRA_W
from wavelet.kernels.smart_gc import WaveletCheckpointFunction
from wavelet.trainer import model as model_module
from wavelet.trainer.ckpt import CheckpointManager, TrainerState
from wavelet.trainer.distributed import World
from wavelet.transport import policy as policy_module
from wavelet.transport.policy import PolicyExportMixin


def _world(world_size: int = 1) -> World:
    return World(
        rank=0,
        local_rank=0,
        world_size=world_size,
        local_world_size=world_size,
        device=torch.device("cpu"),
    )


# ── async checkpoint staging ──────────────────────────────────────────────────


class _RecordingFuture(Future):
    def __init__(self) -> None:
        super().__init__()
        self.result_calls = 0
        self.set_result(None)

    def result(self, timeout=None):
        self.result_calls += 1
        return super().result(timeout)


def test_async_save_waits_for_staging_before_returning(monkeypatch, tmp_path) -> None:
    model = nn.Linear(2, 2)
    manager = CheckpointManager(
        model,
        torch.optim.AdamW(model.parameters()),
        None,
        CheckpointConfig(mode="async", interval=1),
        tmp_path,
        _world(),
    )
    staging = _RecordingFuture()
    upload = _RecordingFuture()
    monkeypatch.setattr(
        "wavelet.trainer.ckpt.dcp.async_save",
        lambda **kwargs: SimpleNamespace(
            staging_completion=staging, upload_completion=upload
        ),
    )

    assert manager.save(TrainerState(step=1, micro_step=1), dataloader=None)

    # The live tensors may be mutated by the next step only after staging.
    assert staging.result_calls == 1
    assert upload.result_calls == 0
    assert manager.pending_save is not None


# ── forced NCCL export on resume ──────────────────────────────────────────────


class _PolicyExporter(PolicyExportMixin):
    pass


def test_forced_nccl_export_prunes_newer_stable_snapshots(monkeypatch) -> None:
    config = RLConfig(policy_transfer={"export_every_steps": 4})
    config = config.model_copy(
        update={
            "policy_transfer": config.policy_transfer.model_copy(
                update={"type": "nccl"}
            )
        }
    )
    exporter = _PolicyExporter()
    exporter.config = config
    exporter.model = object()
    exporter.tokenizer = object()
    exporter.world = _world()
    exporter.output_dir = Path("outputs/run")
    exporter._export_nccl_policy = Mock(return_value=Path("policy"))
    pruned: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        policy_module,
        "prune_policy_snapshots_beyond",
        lambda policy_dir, *, step: pruned.append((policy_dir, step)),
    )

    assert exporter.export_policy(step=7, force=True) == Path("policy")
    assert exporter.export_policy(step=8) == Path("policy")

    assert [step for _, step in pruned] == [7]
    exporter._export_nccl_policy.assert_any_call(7)


# ── LoRA export key naming ────────────────────────────────────────────────────


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(ids))


def test_lora_parameter_shapes_match_state_dict_keys_for_embeddings() -> None:
    peft_model = get_peft_model(
        _TinyModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["embed", "proj"])
    )

    shapes = model_module._lora_parameter_shapes(peft_model)

    parameters = dict(peft_model.named_parameters())
    assert shapes, "expected LoRA parameters"
    assert any("lora_embedding_A" in key for key in shapes)
    for key, shape in shapes.items():
        assert key in parameters, key
        assert tuple(parameters[key].shape) == shape


# ── smart gradient checkpointing ──────────────────────────────────────────────


def test_smart_gc_recompute_runs_under_forward_autocast_state() -> None:
    observed: list[bool] = []

    def run_function(value: torch.Tensor) -> torch.Tensor:
        observed.append(torch.is_autocast_enabled("cpu"))
        return value * 2

    value = torch.ones(4, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = WaveletCheckpointFunction.apply(run_function, True, value)
    output.sum().backward()

    assert observed == [True, True]
    assert torch.equal(value.grad, torch.full((4,), 2.0))


# ── fused LoRA kernels ────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(5, 4), (2, 3, 4)])
def test_lora_w_backward_accepts_2d_and_3d_inputs(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, requires_grad=True)
    weight = torch.randn(3, 4)
    lora_a = torch.randn(2, 4, requires_grad=True)
    lora_b = torch.randn(3, 2, requires_grad=True)

    out = LoRA_W.apply(x, weight, None, lora_a, lora_b, 1.0)
    out.sum().backward()

    expected = x.detach() @ (weight + lora_b @ lora_a).t()
    assert torch.allclose(out, expected, atol=1e-5)
    assert x.grad is not None and x.grad.shape == x.shape
    assert lora_a.grad is not None and lora_b.grad is not None


# ── resume discards batches the new run regenerates ───────────────────────────


def test_prune_rollout_batches_from_removes_only_later_steps(tmp_path) -> None:
    from wavelet.transport.queue import prune_rollout_batches_from

    queue_dir = tmp_path / "queue"
    for step in (3, 4, 5):
        (queue_dir / f"step-{step:06d}").mkdir(parents=True)
    (queue_dir / ".step-000006.tmp").mkdir()
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    for step in (3, 4):
        (rollouts_dir / f"materialized-step-{step:06d}.jsonl").write_text("{}")

    removed = prune_rollout_batches_from(
        queue_dir, first_step=4, materialized_dir=rollouts_dir
    )

    assert sorted(path.name for path in removed) == [
        "materialized-step-000004.jsonl",
        "step-000004",
        "step-000005",
    ]
    assert (queue_dir / "step-000003").exists()
    assert (rollouts_dir / "materialized-step-000003.jsonl").exists()


def test_resume_discards_streaming_chunks_after_the_checkpoint_step(tmp_path) -> None:
    from wavelet.orchestrator.scheduler import (
        discard_rollout_batches_after_resume,
        first_regenerated_queue_step,
    )

    config = RLConfig(
        output_dir=tmp_path,
        launcher={"mode": "process"},
        orchestrator={
            "verifier_env_id": "reverse-text",
            "examples_per_step": 4,
            "rollouts_per_example": 2,
            "rollout_chunk_examples": 2,
            "max_async_level": 2,
            "max_off_policy_steps": 1,
        },
    )
    # Two chunks per optimizer step: resuming at step 2 keeps chunks 4-5.
    assert first_regenerated_queue_step(config, start_step=2) == 6
    queue_dir = tmp_path / "rollouts"
    for step in range(4, 8):
        (queue_dir / f"step-{step:06d}").mkdir(parents=True)

    discard_rollout_batches_after_resume(config, start_step=2)

    assert sorted(path.name for path in queue_dir.iterdir()) == [
        "step-000004",
        "step-000005",
    ]


def test_resume_at_step_zero_keeps_every_batch(tmp_path) -> None:
    from wavelet.orchestrator.scheduler import discard_rollout_batches_after_resume

    config = RLConfig(output_dir=tmp_path)
    (tmp_path / "rollouts" / "step-000000").mkdir(parents=True)

    discard_rollout_batches_after_resume(config, start_step=0)

    assert (tmp_path / "rollouts" / "step-000000").exists()


# ── modules_to_save in lightweight FSDP exports ───────────────────────────────


def test_lora_parameter_shapes_include_modules_to_save_copies() -> None:
    peft_model = get_peft_model(
        _TinyModel(),
        LoraConfig(
            r=2, lora_alpha=4, target_modules=["embed"], modules_to_save=["proj"]
        ),
    )

    shapes = model_module._lora_parameter_shapes(peft_model)

    parameters = dict(peft_model.named_parameters())
    saved = {key for key in shapes if ".modules_to_save.default." in key}
    assert saved, "modules_to_save copies must be exported with the adapter"
    for key in saved:
        assert key in parameters, key
        assert tuple(parameters[key].shape) == shapes[key]

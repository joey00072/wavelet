from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from wavelet.configs.sft import (
    ActivationCheckpointingConfig,
    LoRAConfig,
    ModelConfig,
    SFTConfig,
)
from wavelet.trainer import model as model_utils
from wavelet.trainer.debug import DEBUG_MODEL_NAME
from wavelet.trainer.distributed import ParallelDims
from wavelet.trainer.model import setup_tokenizer
from wavelet.trainer.trainer import BaseTrainer


class _Tokenizer:
    pad_token = None
    eos_token = "<eos>"
    chat_template = "original"
    padding_side = "right"


def test_pre_download_model_populates_hugging_face_cache(monkeypatch, tmp_path) -> None:
    downloaded = tmp_path / "hub" / "snapshot"
    snapshot_download = Mock(return_value=str(downloaded))
    monkeypatch.setattr(model_utils, "snapshot_download", snapshot_download)

    result = model_utils.pre_download_model("org/model")

    assert result == downloaded
    snapshot_download.assert_called_once_with(repo_id="org/model", repo_type="model")


def test_pre_download_model_skips_local_and_debug_models(monkeypatch, tmp_path) -> None:
    snapshot_download = Mock()
    monkeypatch.setattr(model_utils, "snapshot_download", snapshot_download)

    assert model_utils.pre_download_model(str(tmp_path)) == tmp_path
    assert model_utils.pre_download_model(DEBUG_MODEL_NAME) is None
    snapshot_download.assert_not_called()


def test_setup_tokenizer_falls_back_from_adapter_to_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | str] = []
    tokenizer = _Tokenizer()

    def from_pretrained(source: Path | str, **_: object) -> _Tokenizer:
        calls.append(source)
        if source == Path("outputs/policy/adapter"):
            raise ValueError("adapter does not contain tokenizer artifacts")
        return tokenizer

    monkeypatch.setattr(
        "wavelet.trainer.model.AutoTokenizer.from_pretrained", from_pretrained
    )

    result = setup_tokenizer(
        ModelConfig(
            name="org/base-model",
            adapter_path=Path("outputs/policy/adapter"),
        )
    )

    assert result is tokenizer
    assert calls == [Path("outputs/policy/adapter"), "org/base-model"]
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"


def test_setup_tokenizer_does_not_hide_base_model_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def from_pretrained(source: Path | str, **_: object) -> _Tokenizer:
        raise ValueError(f"cannot load {source}")

    monkeypatch.setattr(
        "wavelet.trainer.model.AutoTokenizer.from_pretrained", from_pretrained
    )

    with pytest.raises(ValueError, match="cannot load org/base-model"):
        setup_tokenizer(ModelConfig(name="org/base-model"))


def test_auto_attention_falls_back_to_sdpa_without_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_is_hopper_gpu", lambda: False)
    monkeypatch.setattr(model_utils, "_flash_attention_available", lambda: False)

    assert model_utils._best_attn_implementation() == "sdpa"


def test_auto_attention_selects_flash_attention_3_on_hopper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_is_hopper_gpu", lambda: True)
    monkeypatch.setattr(model_utils, "_flash_attention_3_available", lambda: True)

    assert model_utils._best_attn_implementation() == "flash_attention_3"


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), True), ((8, 9), False), ((10, 0), False), ((12, 0), False)],
)
def test_hopper_detection_uses_sm_major_version(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.setattr(model_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        model_utils.torch.cuda,
        "get_device_capability",
        lambda: capability,
    )

    assert model_utils._is_hopper_gpu() is expected


def test_auto_attention_falls_back_from_unavailable_flash_attention_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_is_hopper_gpu", lambda: True)
    monkeypatch.setattr(model_utils, "_flash_attention_3_available", lambda: False)
    monkeypatch.setattr(model_utils, "_flash_attention_available", lambda: True)

    assert model_utils._best_attn_implementation() == "flash_attention_2"


def test_explicit_flash_attention_requires_importable_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_flash_attention_available", lambda: False)

    with pytest.raises(ImportError, match="uv sync --extra flash-attn"):
        model_utils._model_load_kwargs(
            ModelConfig(attn_implementation="flash_attention_2"),
            distributed=False,
            parallel_dims=None,
        )


def test_explicit_flash_attention_3_requires_hopper_and_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_utils, "_is_hopper_gpu", lambda: False)
    monkeypatch.setattr(model_utils, "_flash_attention_3_available", lambda: True)

    with pytest.raises(ImportError, match="Hopper GPU"):
        model_utils._model_load_kwargs(
            ModelConfig(attn_implementation="flash_attention_3"),
            distributed=False,
            parallel_dims=None,
        )


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_setup_runtime_applies_configured_matmul_precision(
    monkeypatch: pytest.MonkeyPatch,
    precision: Literal["highest", "high", "medium"],
) -> None:
    applied: list[str] = []
    monkeypatch.setattr(model_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        model_utils.torch,
        "set_float32_matmul_precision",
        applied.append,
    )

    model_utils.setup_runtime(ModelConfig(matmul_precision=precision))

    assert applied == [precision]


def test_removed_allow_tf32_model_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="allow_tf32"):
        ModelConfig.model_validate({"allow_tf32": True})


def test_fsdp2_meta_init_selection_rejects_unsupported_lora_copies() -> None:
    trainer = BaseTrainer(
        SFTConfig(
            model={"meta_device_init": True},
            fsdp={"enabled": True, "impl": "fsdp2"},
            lora=LoRAConfig(modules_to_save=["lm_head"]),
        )
    )
    trainer.parallel_dims = ParallelDims(world_size=1)

    assert trainer._use_fsdp2_meta_init(trainer.config.fsdp) is False


def test_fsdp2_meta_init_selection_accepts_fresh_lora() -> None:
    trainer = BaseTrainer(
        SFTConfig(
            model={"meta_device_init": True},
            fsdp={"enabled": True, "impl": "fsdp2"},
            lora=LoRAConfig(),
        )
    )
    trainer.parallel_dims = ParallelDims(world_size=1)

    assert trainer._use_fsdp2_meta_init(trainer.config.fsdp) is True


def test_fsdp2_meta_init_selection_skips_random_debug_model() -> None:
    trainer = BaseTrainer(
        SFTConfig(
            model={"name": DEBUG_MODEL_NAME, "meta_device_init": True},
            fsdp={"enabled": True, "impl": "fsdp2"},
        )
    )
    trainer.parallel_dims = ParallelDims(world_size=1)

    assert trainer._use_fsdp2_meta_init(trainer.config.fsdp) is False


def test_meta_init_rejects_nonpersistent_buffers_it_cannot_rebuild() -> None:
    model = torch.nn.Module()
    model.register_buffer(
        "attention_cache",
        torch.empty(2, device="meta"),
        persistent=False,
    )

    with pytest.raises(RuntimeError, match="attention_cache"):
        model_utils._validate_meta_model_buffers(model, set(model.state_dict()))


def test_compile_fullgraph_requires_compile_enabled() -> None:
    with pytest.raises(ValueError, match="compile_fullgraph"):
        ModelConfig(compile_fullgraph=True)


def test_compiled_debug_layers_match_eager_loss_with_activation_checkpointing() -> None:
    eager = model_utils.build_debug_model(max_seq_length=64)
    compiled = copy.deepcopy(eager)
    for model in (eager, compiled):
        model.config.use_cache = False
        model.train()

    state_keys = set(compiled.state_dict())
    checkpointed_count = model_utils.apply_activation_checkpointing(
        compiled,
        ActivationCheckpointingConfig(),
    )
    compiled_count = model_utils.compile_transformer_layers(
        compiled,
        fullgraph=False,
        backend="eager",
    )
    input_ids = torch.tensor([[1, 7, 8, 9, 10, 1]])
    eager_loss = eager(input_ids=input_ids, labels=input_ids).loss
    compiled_loss = compiled(input_ids=input_ids, labels=input_ids).loss
    compiled_loss.backward()

    assert checkpointed_count == 2
    assert compiled_count == 2
    assert set(compiled.state_dict()) == state_keys
    assert compiled_loss.item() == pytest.approx(eager_loss.item(), rel=1e-5, abs=1e-6)
    assert any(parameter.grad is not None for parameter in compiled.parameters())


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_activation_checkpointing_wraps_configured_layer_frequency(
    mode: Literal["full", "selective"],
) -> None:
    model = model_utils.build_debug_model(max_seq_length=64)
    model.config.use_cache = False
    state_keys = set(model.state_dict())
    config = ActivationCheckpointingConfig(
        mode=mode,
        freq=2,
        targets=[] if mode == "selective" else None,
    )

    wrapped_count = model_utils.apply_activation_checkpointing(model, config)
    input_ids = torch.tensor([[1, 7, 8, 9, 10, 1]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert wrapped_count == 1
    assert set(model.state_dict()) == state_keys
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_activation_checkpointing_rejects_inert_or_conflicting_settings() -> None:
    with pytest.raises(ValueError, match="targets"):
        ActivationCheckpointingConfig(mode="full", targets=["aten::mm"])
    with pytest.raises(ValueError, match="smart_gc"):
        ModelConfig(smart_gc=True, activation_checkpointing=None)
    with pytest.raises(ValueError, match="smart_gc"):
        ModelConfig(
            smart_gc=True,
            activation_checkpointing={"mode": "selective"},
        )
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        ModelConfig.model_validate({"gradient_checkpointing": True})


def test_selective_checkpoint_policy_matches_namespaces_and_operations() -> None:
    class _Operation:
        def __init__(self, namespace: str, name: str) -> None:
            self.namespace = namespace
            self._name = name

        def name(self) -> str:
            return self._name

    targets = frozenset({"custom", "aten::mm"})

    assert (
        model_utils._selective_checkpoint_policy(
            object(),
            _Operation("custom", "custom::op"),
            targets=targets,
        )
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        model_utils._selective_checkpoint_policy(
            object(),
            _Operation("aten", "aten::mm"),
            targets=targets,
        )
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        model_utils._selective_checkpoint_policy(
            object(),
            _Operation("aten", "aten::relu"),
            targets=targets,
        )
        is CheckpointPolicy.PREFER_RECOMPUTE
    )

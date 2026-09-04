from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from wavelet.configs.config import ModelConfig
from wavelet.trainer.ckpt import AppState
from wavelet.trainer.debug import DEBUG_MODEL_NAME
from wavelet.trainer.model import setup_model
from wavelet.tools.convert_checkpoint import (
    convert_checkpoint,
    resolve_checkpoint_source,
)
from wavelet.utils.pathing import STABLE_CHECKPOINT_MARKER, create_launch_attempt
from wavelet.utils.serialization import dump_yaml


def _write_debug_checkpoint(tmp_path: Path) -> tuple[Path, torch.Tensor]:
    checkpoint_dir = tmp_path / "checkpoint-3"
    trainer_dir = checkpoint_dir / "trainer"
    model_config = ModelConfig(
        name=DEBUG_MODEL_NAME,
        torch_dtype="float32",
        activation_checkpointing=None,
    )
    model = setup_model(model_config, max_seq_length=64)
    with torch.no_grad():
        model.transformer.wte.weight.fill_(0.25)
    expected = model.transformer.wte.weight.detach().to(torch.bfloat16).clone()
    dcp.save(
        state_dict={"app": AppState(model, None, None)},
        checkpoint_id=trainer_dir,
        no_dist=True,
    )
    (checkpoint_dir / STABLE_CHECKPOINT_MARKER).touch()
    attempt = create_launch_attempt(tmp_path)
    dump_yaml(
        attempt.config_dir / "sft.yaml",
        {
            "model": model_config.model_dump(mode="json"),
            "lora": None,
            "data": {"seq_len": 64},
        },
    )
    return checkpoint_dir, expected


def test_convert_checkpoint_writes_hf_safetensors(tmp_path: Path) -> None:
    checkpoint_dir, expected = _write_debug_checkpoint(tmp_path)

    output_dir = convert_checkpoint(checkpoint_dir)

    tensor_path = output_dir / "model.safetensors"
    assert tensor_path.is_file()
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("transformer.wte.weight")
    assert weight.dtype == torch.bfloat16
    assert torch.equal(weight, expected)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["use_cache"] is True
    assert (output_dir / "tokenizer_config.json").is_file()


def test_convert_checkpoint_rejects_lora_config(tmp_path: Path) -> None:
    checkpoint_dir, _ = _write_debug_checkpoint(tmp_path)
    config_path = tmp_path / "configs" / "latest" / "resolved" / "sft.yaml"
    payload = {
        "model": {"name": DEBUG_MODEL_NAME},
        "lora": {"rank": 4},
        "data": {"seq_len": 64},
    }
    dump_yaml(config_path, payload)

    with pytest.raises(ValueError, match="LoRA checkpoints"):
        convert_checkpoint(checkpoint_dir)


def test_resolve_checkpoint_source_rejects_unstable_checkpoint(
    tmp_path: Path,
) -> None:
    trainer_dir = tmp_path / "checkpoint-1" / "trainer"
    trainer_dir.mkdir(parents=True)
    (trainer_dir / ".metadata").touch()

    with pytest.raises(ValueError, match="STABLE"):
        resolve_checkpoint_source(trainer_dir)


def test_convert_checkpoint_refuses_nonempty_destination(tmp_path: Path) -> None:
    checkpoint_dir, _ = _write_debug_checkpoint(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        convert_checkpoint(checkpoint_dir, output_dir)

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"

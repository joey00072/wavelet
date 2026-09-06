"""Convert a stable Wavelet DCP checkpoint to Hugging Face safetensors."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed.checkpoint as dcp
from accelerate import init_empty_weights
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from wavelet.configs.config import ModelConfig
from wavelet.trainer.ckpt import AppState
from wavelet.trainer.debug import DEBUG_MODEL_NAME
from wavelet.trainer.model import setup_model, setup_tokenizer
from wavelet.utils.pathing import get_config_dir, is_stable_checkpoint
from wavelet.utils.serialization import load_yaml

CONFIG_NAMES = ("rl_trainer.yaml", "sft.yaml", "rl.yaml")
OUTPUT_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True, slots=True)
class CheckpointSource:
    step_dir: Path
    dcp_dir: Path


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    model: ModelConfig
    seq_len: int


def resolve_checkpoint_source(checkpoint: Path) -> CheckpointSource:
    """Resolve a stable checkpoint step or its trainer DCP directory."""
    checkpoint = checkpoint.expanduser()
    if (checkpoint / "trainer" / ".metadata").is_file():
        step_dir = checkpoint
        dcp_dir = checkpoint / "trainer"
    elif checkpoint.name == "trainer" and (checkpoint / ".metadata").is_file():
        step_dir = checkpoint.parent
        dcp_dir = checkpoint
    else:
        raise FileNotFoundError(
            f"No DCP trainer checkpoint found at '{checkpoint}'. Expected a "
            "checkpoint-N directory containing trainer/.metadata, or that trainer "
            "directory itself."
        )
    if not is_stable_checkpoint(step_dir):
        raise ValueError(
            f"Checkpoint '{step_dir}' is incomplete because it has no STABLE marker."
        )
    return CheckpointSource(step_dir=step_dir, dcp_dir=dcp_dir)


def resolve_conversion_config(
    source: CheckpointSource,
    config_path: Path | None = None,
) -> ConversionConfig:
    """Load model details from an explicit or run-resolved config file."""
    if config_path is None:
        config_dir = get_config_dir(source.step_dir.parent)
        candidates = [config_dir / name for name in CONFIG_NAMES]
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            names = ", ".join(CONFIG_NAMES)
            raise FileNotFoundError(
                f"No resolved trainer config found for '{source.step_dir}'. "
                f"Expected one of {names} under '{config_dir}', or pass --config."
            )
    else:
        config_path = config_path.expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found at '{config_path}'.")

    payload = load_yaml(config_path)
    raw_model = payload.get("model")
    if not isinstance(raw_model, dict):
        raise TypeError(f"Config '{config_path}' has no model mapping.")
    model_config = ModelConfig.model_validate(raw_model)
    if payload.get("lora", object()) is not None:
        raise ValueError(
            "LoRA checkpoints are not supported by convert-checkpoint; export the "
            "adapter policy instead."
        )
    if model_config.adapter_path is not None:
        raise ValueError(
            "Adapter-backed checkpoints are not supported by convert-checkpoint."
        )
    if model_config.load_in_4bit:
        raise ValueError(
            "4-bit checkpoints are not supported by convert-checkpoint; only full "
            "model fine-tunes can be converted."
        )
    raw_data = payload.get("data", {})
    seq_len = raw_data.get("seq_len", 128) if isinstance(raw_data, dict) else 128
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError(f"Config '{config_path}' has an invalid data.seq_len.")
    return ConversionConfig(model=model_config, seq_len=seq_len)


def _check_checkpoint_is_full_model(dcp_dir: Path) -> None:
    metadata = FileSystemReader(dcp_dir).read_metadata()
    lora_markers = ("lora_A", "lora_B", ".base_layer.")
    lora_key = next(
        (
            key
            for key in metadata.state_dict_metadata
            if any(marker in key for marker in lora_markers)
        ),
        None,
    )
    if lora_key is not None:
        raise ValueError(
            "Checkpoint contains LoRA state and cannot be converted as a full "
            f"model (found '{lora_key}')."
        )


def _portable_model_config(
    config: ModelConfig,
    *,
    dtype: Literal["bfloat16", "float16", "float32"],
) -> ModelConfig:
    return config.model_copy(
        update={
            "activation_checkpointing": None,
            "attn_implementation": "eager",
            "compile": False,
            "compile_fullgraph": False,
            "fused_lora_mlp": False,
            "fused_lora_o": False,
            "fused_lora_qkv": False,
            "fused_lm_head_token_chunk_size": "disabled",
            "smart_gc": False,
            "torch_dtype": dtype,
        }
    )


def _validate_output_dir(output_dir: Path, source: CheckpointSource) -> Path:
    output_dir = output_dir.expanduser()
    resolved_output = output_dir.resolve()
    resolved_dcp = source.dcp_dir.resolve()
    if resolved_output == source.step_dir.resolve() or resolved_output.is_relative_to(
        resolved_dcp
    ):
        raise ValueError(
            "Output directory cannot replace the checkpoint or be inside its "
            "trainer state."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path '{output_dir}' is not a directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty; choose a new directory."
        )
    return output_dir


def convert_checkpoint(
    checkpoint: Path,
    output_dir: Path | None = None,
    *,
    config_path: Path | None = None,
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
    max_shard_size: str = "5GB",
) -> Path:
    """Load model state from DCP and write an inference-ready HF directory."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "convert-checkpoint currently runs in one process; invoke it with "
            "'uv run wavelet convert-checkpoint', not torchrun."
        )
    source = resolve_checkpoint_source(checkpoint)
    conversion = resolve_conversion_config(source, config_path)
    _check_checkpoint_is_full_model(source.dcp_dir)
    if dtype not in OUTPUT_DTYPES:
        choices = ", ".join(OUTPUT_DTYPES)
        raise ValueError(
            f"Unsupported output dtype '{dtype}'; choose one of {choices}."
        )
    output_dir = source.step_dir / "weights" if output_dir is None else output_dir
    output_dir = _validate_output_dir(output_dir, source)

    model_config = _portable_model_config(conversion.model, dtype=dtype)
    model = _setup_checkpoint_model(
        model_config,
        max_seq_length=conversion.seq_len,
        dtype=OUTPUT_DTYPES[dtype],
    )
    dcp.load(
        state_dict={"app": AppState(model, None, None)},
        checkpoint_id=source.dcp_dir,
        no_dist=True,
    )
    model.to(dtype=OUTPUT_DTYPES[dtype])
    model.config.use_cache = True
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_hf_artifacts(
        model,
        model_config=model_config,
        output_dir=output_dir,
        max_shard_size=max_shard_size,
    )
    return output_dir


def _setup_checkpoint_model(
    model_config: ModelConfig,
    *,
    max_seq_length: int,
    dtype: torch.dtype,
) -> PreTrainedModel:
    """Allocate checkpoint targets without loading the base model weights."""
    if model_config.name == DEBUG_MODEL_NAME:
        return setup_model(model_config, max_seq_length=max_seq_length).to(dtype=dtype)

    hf_config = AutoConfig.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=model_config.trust_remote_code,
            dtype=dtype,
            attn_implementation="eager",
        )
    model.to_empty(device="cpu")
    model.tie_weights()
    return model


def _save_hf_artifacts(
    model: PreTrainedModel,
    *,
    model_config: ModelConfig,
    output_dir: Path,
    max_shard_size: str,
) -> None:
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    setup_tokenizer(model_config).save_pretrained(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="stable checkpoint-N directory or its trainer subdirectory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="destination (default: <checkpoint>/weights)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="resolved run config when it cannot be discovered beside the checkpoint",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(OUTPUT_DTYPES),
        default="bfloat16",
        help="floating-point output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="maximum Hugging Face safetensors shard size (default: 5GB)",
    )
    args = parser.parse_args(argv)
    output_dir = convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        config_path=args.config,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
    )
    print(f"Converted checkpoint to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

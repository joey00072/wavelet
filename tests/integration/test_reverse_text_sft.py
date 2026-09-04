from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavelet.trainer.debug import DEBUG_MODEL_NAME
from wavelet.utils.serialization import dump_yaml


def _run_sft(config_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    result = subprocess.run(
        [sys.executable, "-m", "wavelet", "sft", "@", str(config_path)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"SFT subprocess failed with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _config(
    *,
    data_path: Path,
    output_dir: Path,
    max_steps: int,
    resume_step: int | None,
) -> dict[str, object]:
    return {
        "model": {
            "name": DEBUG_MODEL_NAME,
            "torch_dtype": "float32",
            "attn_implementation": "eager",
            "activation_checkpointing": None,
        },
        "data": {
            "source": "local",
            "path": str(data_path),
            "batch_size": 2,
            "micro_batch_size": 2,
            "seq_len": 32,
            "shuffle": False,
            "pin_memory": False,
        },
        "optim": {
            "type": "adamw",
            "implementation": "for-loop",
            "lr": 0.02,
            "weight_decay": 0.0,
        },
        "scheduler": {"type": "constant"},
        "lora": None,
        "gc": None,
        "ckpt": {
            "mode": "async",
            "interval": 1,
            "keep_last": 3,
            "resume_step": resume_step,
        },
        "max_steps": max_steps,
        "seed": 7,
        "monitor": {
            "log_cuda_memory": False,
            "log_disk_usage": False,
            "wandb": {"enabled": False},
        },
        "log": {"log_every": 1},
        "output_dir": str(output_dir),
        "clean_output_dir": resume_step is None,
    }


@pytest.mark.integration
@pytest.mark.slow
def test_reverse_text_sft_subprocess_improves_and_resumes(tmp_path: Path) -> None:
    data_path = tmp_path / "reverse_text.jsonl"
    rows = [
        {"prompt": value, "completion": value[::-1]}
        for value in ("abc", "wave", "train", "token")
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    first_config = tmp_path / "first.yaml"
    resumed_config = tmp_path / "resumed.yaml"
    dump_yaml(
        first_config,
        _config(
            data_path=data_path,
            output_dir=output_dir,
            max_steps=1,
            resume_step=None,
        ),
    )
    dump_yaml(
        resumed_config,
        _config(
            data_path=data_path,
            output_dir=output_dir,
            max_steps=2,
            resume_step=-1,
        ),
    )

    _run_sft(first_config)
    _run_sft(resumed_config)

    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    training_rows = [row for row in metrics if "loss" in row]
    assert [row["step"] for row in training_rows] == [1, 2]
    assert training_rows[-1]["loss"] < training_rows[0]["loss"]
    assert (
        training_rows[-1]["progress/total_tokens"]
        > training_rows[0]["progress/total_tokens"]
    )
    assert (output_dir / "checkpoint-2" / "STABLE").is_file()
    run_metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert run_metadata["resumed_from"].endswith("checkpoint-1")

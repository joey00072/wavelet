from __future__ import annotations

from wavelet.configs.rl_config import RLConfig
from wavelet.orchestrator.policy_metadata import policy_metadata, precision_metadata


def test_precision_metadata_records_train_and_serve_settings() -> None:
    config = RLConfig(
        model={"torch_dtype": "bfloat16", "load_in_4bit": True},
        optim={"type": "paged_adamw_8bit"},
        inference={
            "vllm": {
                "dtype": "float16",
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
                "tensor_parallel_size": 2,
            }
        },
    )

    metadata = precision_metadata(config)

    assert metadata["trainer"]["torch_dtype"] == "bfloat16"
    assert metadata["trainer"]["load_in_4bit"] is True
    assert metadata["trainer"]["optimizer"] == "paged_adamw_8bit"
    assert metadata["adapter"]["enabled"] is True
    assert metadata["adapter"]["rank"] == config.lora.rank
    assert metadata["inference"]["dtype"] == "float16"
    assert metadata["inference"]["quantization"] == "bitsandbytes"
    assert metadata["inference"]["tensor_parallel_size"] == 2
    assert metadata["train_serve_dtype_match"] is False
    assert metadata["train_serve_low_precision_match"] is True


def test_precision_metadata_marks_intentional_low_precision_mismatch() -> None:
    config = RLConfig(
        model={"load_in_4bit": False},
        inference={"vllm": {"quantization": "bitsandbytes"}},
    )

    metadata = precision_metadata(config)

    assert metadata["train_serve_low_precision_match"] is False


def test_policy_metadata_wraps_precision_contract() -> None:
    config = RLConfig(model={"torch_dtype": "float32"})

    metadata = policy_metadata(
        config=config,
        format_version=1,
        step=3,
        kind="adapter",
        created_at="2026-06-04T00:00:00+00:00",
        extra={"source_adapter_path": "/tmp/adapter"},
    )

    assert metadata["format_version"] == 1
    assert metadata["step"] == 3
    assert metadata["kind"] == "adapter"
    assert metadata["created_at"] == "2026-06-04T00:00:00+00:00"
    assert metadata["source_adapter_path"] == "/tmp/adapter"
    assert metadata["precision"]["trainer"]["torch_dtype"] == "float32"

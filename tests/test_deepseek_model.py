import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wavelet.trainer.models.deepseek_v4 import DeepseekV4Config
from wavelet.trainer.models.deepseek_v4.conversion import (
    convert_checkpoint,
    raw_to_wavelet,
    wavelet_to_raw,
)
from wavelet.trainer.models.deepseek_v4.modeling import DeepseekV4ForCausalLM
from wavelet.transport.policy import PolicyExportMixin


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        partial_rotary_factor=1.0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        sliding_window=4,
        o_groups=2,
        o_lora_rank=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        moe_intermediate_size=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_hash_layers=1,
        hc_mult=2,
        max_position_embeddings=16,
    )


def test_native_model_all_attention_types_hash_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = DeepseekV4ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        expected = model(input_ids).logits

    model.save_pretrained(tmp_path)
    assert model.config.wavelet_checkpoint_format == "deepseek_v4_native_v1"
    restored = DeepseekV4ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.shape == (1, 8, 16)

    trainable = DeepseekV4ForCausalLM(_tiny_config())
    loss = trainable(input_ids, labels=input_ids).loss
    assert loss is not None
    loss.backward()
    assert trainable.model.embed_tokens.weight.grad is not None


def test_raw_checkpoint_conversion_roundtrips_strict_native_keys() -> None:
    model = DeepseekV4ForCausalLM(_tiny_config())
    native = model.state_dict()
    raw = wavelet_to_raw(native)
    converted = raw_to_wavelet(raw)
    assert set(converted) == set(native)
    restored = DeepseekV4ForCausalLM(_tiny_config())
    restored.load_state_dict(converted, strict=True)
    for key, value in native.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_bfloat16_forward_backward_and_fp32_transfer_policy() -> None:
    model = DeepseekV4ForCausalLM(_tiny_config()).to(dtype=torch.bfloat16)
    for name in (
        "model.layers.0.self_attn.sinks",
        "model.layers.1.self_attn.compressor.position_bias",
        "model.layers.1.mlp.router.selection_bias",
        "model.layers.0.attn_hc.fn",
        "model.norm.weight",
    ):
        assert model.keep_in_fp32_for_weight_transfer(name)
    assert not model.keep_in_fp32_for_weight_transfer("lm_head.weight")

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    result = model(input_ids, labels=input_ids)
    assert result.loss is not None and torch.isfinite(result.loss)
    result.loss.backward()
    assert model.model.embed_tokens.weight.grad is not None


def test_liger_fused_rejects_native_deepseek_checkpoint(tmp_path: Path) -> None:
    from wavelet.trainer.model import apply_liger_kernel

    DeepseekV4ForCausalLM(_tiny_config()).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='loss_impl="torch"'):
        apply_liger_kernel("liger_fused", str(tmp_path))


def test_native_raw_native_converter_preserves_logits(tmp_path: Path) -> None:
    model = DeepseekV4ForCausalLM(_tiny_config()).eval()
    source = tmp_path / "native"
    raw = tmp_path / "raw"
    roundtrip = tmp_path / "roundtrip"
    model.save_pretrained(source)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        expected = model(input_ids).logits

    convert_checkpoint(source, raw, to_raw=True)
    convert_checkpoint(raw, roundtrip)
    from wavelet.configs.config import ModelConfig
    from wavelet.trainer.model import setup_model

    restored = setup_model(
        ModelConfig(
            name=str(roundtrip), torch_dtype="float32", activation_checkpointing=None
        )
    ).eval()
    with torch.no_grad():
        actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_converter_rejects_corrupt_native_checkpoint_before_target(
    tmp_path: Path, corruption: str
) -> None:
    from safetensors.torch import load_file, save_file

    source = tmp_path / "native"
    target = tmp_path / "raw"
    DeepseekV4ForCausalLM(_tiny_config()).save_pretrained(source)
    weights = load_file(source / "model.safetensors")
    if corruption == "missing":
        del weights["model.embed_tokens.weight"]
    else:
        weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"][:1]
    save_file(weights, source / "model.safetensors")

    with pytest.raises((ValueError, RuntimeError), match="DeepSeek|shape|missing"):
        convert_checkpoint(source, target, to_raw=True)
    assert not target.exists()


def test_ddp_native_policy_export_writes_raw_fp32_checkpoint(tmp_path: Path) -> None:
    rendezvous = tmp_path / "rendezvous"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    try:
        model = DeepseekV4ForCausalLM(_tiny_config())
        with torch.no_grad():
            model.model.layers[0].self_attn.sinks.fill_(0.123456789)
            model.model.layers[1].mlp.router.selection_bias.fill_(0.314159265)
        wrapped = DistributedDataParallel(model)

        class Tokenizer:
            def save_pretrained(self, path: Path) -> None:
                (path / "tokenizer.json").write_text("{}")

        trainer = SimpleNamespace(
            model=wrapped,
            config=SimpleNamespace(
                lora=None,
                policy_transfer=SimpleNamespace(lightweight_lora=False),
            ),
            world=SimpleNamespace(is_main=True),
            parallel_dims=None,
            tokenizer=Tokenizer(),
        )
        output = PolicyExportMixin._save_filesystem_policy(trainer, tmp_path / "export")
        from safetensors.torch import load_file

        raw = load_file(output / "model.safetensors")
        assert "layers.0.attn.attn_sink" in raw
        assert "layers.1.ffn.gate.bias" in raw
        assert raw["layers.0.attn.attn_sink"].dtype == torch.float32
        torch.testing.assert_close(
            raw["layers.0.attn.attn_sink"],
            torch.full_like(raw["layers.0.attn.attn_sink"], 0.123456789),
            atol=0,
            rtol=0,
        )
        config = DeepseekV4Config.from_pretrained(output)
        assert getattr(config, "wavelet_checkpoint_format", None) is None
        assert getattr(config, "quantization_config", None) is None
    finally:
        dist.destroy_process_group()


def test_setup_model_rejects_raw_deepseek_policy(tmp_path: Path) -> None:
    from wavelet.configs.config import ModelConfig
    from wavelet.trainer.model import setup_model

    raw = tmp_path / "raw"
    DeepseekV4ForCausalLM(_tiny_config()).save_pretrained(raw)
    # A raw export is identified by the absent native marker before any weights are loaded.
    config_path = raw / "config.json"
    config_data = json.loads(config_path.read_text())
    config_data.pop("wavelet_checkpoint_format", None)
    config_path.write_text(json.dumps(config_data))
    with pytest.raises(ValueError, match="converted native checkpoint"):
        setup_model(ModelConfig(name=str(raw), torch_dtype="float32"))


def test_bfloat16_checkpoint_load_preserves_sensitive_fp32_values(
    tmp_path: Path,
) -> None:
    model = DeepseekV4ForCausalLM(_tiny_config())
    sensitive = {
        "model.layers.0.self_attn.sinks": 0.123456789,
        "model.layers.1.self_attn.compressor.position_bias": -0.987654321,
        "model.layers.1.mlp.router.selection_bias": 0.314159265,
        "model.norm.weight": 1.123456789,
    }
    expected: dict[str, torch.Tensor] = {}
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, value in sensitive.items():
        target = parameters.get(name, buffers.get(name))
        assert target is not None
        target.data = torch.full_like(target, value, dtype=torch.float32)
        expected[name] = target.detach().clone()

    model.save_pretrained(tmp_path)
    restored = DeepseekV4ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    restored_parameters = dict(restored.named_parameters())
    restored_buffers = dict(restored.named_buffers())
    for name, value in expected.items():
        actual = restored_parameters.get(name, restored_buffers.get(name))
        assert actual is not None
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, value, atol=0, rtol=0)
    for module in restored.modules():
        usage = getattr(module, "tokens_per_expert", None)
        if usage is not None:
            assert torch.count_nonzero(usage) == 0

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    result = restored(input_ids, labels=input_ids)
    assert result.loss is not None and torch.isfinite(result.loss)
    result.loss.backward()

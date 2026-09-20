from __future__ import annotations

import pytest
import torch
from transformers import (
    NemotronHConfig,
    NemotronHForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
)

from wavelet.configs.config import ModelConfig
from wavelet.trainer.model import setup_model
from wavelet.trainer.moe import configure_hf_moe_routers


def _qwen35() -> Qwen3_5MoeForCausalLM:
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
    )
    return Qwen3_5MoeForCausalLM(config)


def _nemotron() -> NemotronHForCausalLM:
    return NemotronHForCausalLM(
        NemotronHConfig(
            vocab_size=64,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            n_routed_experts=4,
            moe_intermediate_size=8,
            num_experts_per_tok=2,
            num_hidden_layers=3,
            layers_block_type=["mamba", "attention", "moe"],
            ssm_state_size=4,
            mamba_num_heads=2,
            mamba_head_dim=8,
            n_groups=1,
            conv_kernel=2,
            use_mamba_kernels=False,
        )
    )


@pytest.mark.parametrize("factory", [_qwen35, _nemotron], ids=["qwen35", "nemotron"])
def test_full_new_moe_family_production_load_forward_backward(
    factory, tmp_path
) -> None:
    torch.manual_seed(11)
    source = factory()
    source.save_pretrained(tmp_path)
    config = ModelConfig(
        name=str(tmp_path), torch_dtype="float32", experts_implementation="eager"
    )
    loaded = setup_model(config)
    configure_hf_moe_routers(
        loaded,
        ModelConfig(
            name=str(tmp_path), moe_router_dtype="float32", freeze_moe_router=True
        ),
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = source(input_ids=inputs, use_cache=False).logits
        actual = loaded(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    loss = loaded(input_ids=inputs, labels=inputs, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in loaded.parameters()
        if parameter.requires_grad
    )

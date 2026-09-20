import torch

from wavelet.trainer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb_interleaved,
)
from wavelet.trainer.models.deepseek_v4.dequantize import dequantize_weight
from wavelet.trainer.models.deepseek_v4.moe import DeepseekV4MoE, DeepseekV4Router
from wavelet.trainer.moe import install_moe_load_balance_hook, update_moe_selection_bias


def test_dual_rotary_and_inverse_rotation() -> None:
    config = DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        moe_intermediate_size=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_hash_layers=0,
        partial_rotary_factor=1.0,
        layer_types=["sliding_attention"],
        max_position_embeddings=16,
    )
    rotary = DeepseekV4RotaryEmbedding(config)
    positions = torch.arange(3).view(1, 3)
    cos, sin = rotary(positions, "main", dtype=torch.float32)
    value = torch.randn(1, 1, 3, 4)
    rotated = apply_rotary_pos_emb_interleaved(value, cos, sin)
    restored = apply_rotary_pos_emb_interleaved(rotated, cos, -sin)
    torch.testing.assert_close(value, restored, atol=1e-5, rtol=1e-5)
    assert not torch.equal(cos, rotary(positions, "compress", dtype=torch.float32)[0])


def test_mhc_streams_and_head_are_differentiable() -> None:
    config = DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        moe_intermediate_size=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_hash_layers=0,
        layer_types=["sliding_attention"],
        hc_mult=2,
        hc_sinkhorn_iters=5,
    )
    connection = DeepseekV4HyperConnection(config)
    head = DeepseekV4HyperHead(config)
    connection.init_weights(0.02)
    head.init_weights(0.02)
    states = torch.randn(2, 4, 2, 8, requires_grad=True)
    post, combine, collapsed = connection(states)
    assert combine.shape == (2, 4, 2, 2)
    assert torch.allclose(combine.sum(-1), torch.ones_like(combine.sum(-1)), atol=1e-4)
    loss = (
        head(connection.update_states(post, combine, collapsed, states)).square().mean()
    )
    loss.backward()
    assert states.grad is not None


def test_mxfp4_nibbles_and_block_scales() -> None:
    packed = torch.tensor([[0x10, -22]], dtype=torch.int8)
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    actual = dequantize_weight(packed, scale)
    torch.testing.assert_close(
        actual, torch.tensor([[0.0, 1.0, -2.0, -8.0]], dtype=torch.bfloat16)
    )


def test_hash_router_uses_persistent_token_table_and_backpropagates() -> None:
    config = DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        moe_intermediate_size=8,
        n_routed_experts=3,
        num_experts_per_tok=1,
        num_hash_layers=1,
        vocab_size=16,
        layer_types=["sliding_attention"],
    )
    moe = DeepseekV4MoE(config, 0)
    moe.router.tid2eid.copy_(torch.arange(16).view(-1, 1) % 3)
    x = torch.randn(2, 4, 8, requires_grad=True)
    loss = moe(x, torch.arange(8).view(2, 4)).square().mean()
    loss.backward()
    assert x.grad is not None
    assert "router.tid2eid" in dict(moe.named_buffers())


def test_router_selection_bias_changes_choice_not_gate_weight() -> None:
    router = DeepseekV4Router(4, 3, 1, selection_bias=True)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.selection_bias.copy_(torch.tensor([0.0, 3.0, 0.0]))
    weights, indices, _, _ = router(torch.zeros(1, 2, 4))
    assert indices.tolist() == [[[1], [1]]]
    assert torch.allclose(weights, torch.ones_like(weights))


def _tiny_moe(*, hash_layer: bool = False) -> DeepseekV4MoE:
    config = DeepseekV4Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=4,
        q_lora_rank=2,
        moe_intermediate_size=6,
        n_routed_experts=3,
        num_experts_per_tok=1,
        num_hash_layers=1 if hash_layer else 0,
        vocab_size=8,
        layer_types=["sliding_attention"],
    )
    return DeepseekV4MoE(config, 0)


def test_moe_usage_counts_are_training_only_and_nonpersistent() -> None:
    moe = _tiny_moe()
    moe.train()
    x = torch.randn(2, 3, 4, requires_grad=True)
    moe(x)
    assert moe.tokens_per_expert.sum() == 6
    assert "tokens_per_expert" not in moe.state_dict()
    before = moe.tokens_per_expert.clone()
    with torch.no_grad():
        moe(x.detach())
    assert torch.equal(moe.tokens_per_expert, before)
    moe.eval()
    moe(x.detach())
    assert torch.equal(moe.tokens_per_expert, before)


def test_selection_bias_update_is_centered_and_resets_counts() -> None:
    moe = _tiny_moe()
    model = torch.nn.Module()
    model.moe = moe
    with torch.no_grad():
        moe.tokens_per_expert.copy_(torch.tensor([8.0, 2.0, 2.0]))
    assert update_moe_selection_bias(model) == 1
    torch.testing.assert_close(
        moe.router.selection_bias,
        torch.tensor([-0.0013333333, 0.0006666667, 0.0006666667]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(moe.tokens_per_expert) == 0


def test_selection_bias_optimizer_hook_handles_empty_and_hash_layers() -> None:
    ordinary = _tiny_moe()
    hashed = _tiny_moe(hash_layer=True)
    model = torch.nn.Module()
    model.ordinary = ordinary
    model.hashed = hashed
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    handle = install_moe_load_balance_hook(optimizer, model)
    try:
        optimizer.step()
        assert torch.equal(ordinary.router.selection_bias, torch.zeros(3))
        assert hashed.router.selection_bias is None
        assert torch.equal(hashed.tokens_per_expert, torch.zeros(3))
    finally:
        handle.remove()


def test_mhc_gate_projections_keep_fp32_under_autocast() -> None:
    config = DeepseekV4Config(hidden_size=8, hc_mult=2)
    connection = DeepseekV4HyperConnection(config)
    head = DeepseekV4HyperHead(config)
    connection.init_weights(0.2)
    head.init_weights(0.2)
    states = torch.randn(1, 3, 2, 8)
    expected = connection(states)
    expected_head = head(states)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = connection(states)
        actual_head = head(states)
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.dtype == torch.float32
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_head, expected_head, atol=0, rtol=0)

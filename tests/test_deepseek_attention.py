import pytest
import torch

from wavelet.trainer.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4RotaryEmbedding,
)
from wavelet.trainer.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    PackedContext,
    RMSNorm,
    _eager_sparse_attention,
)


def _config(layer_type: str) -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        partial_rotary_factor=1.0,
        layer_types=[layer_type],
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
        num_hash_layers=0,
        max_position_embeddings=16,
    )


@pytest.mark.parametrize(
    "layer_type",
    [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ],
)
def test_attention_packed_documents_isolate_and_backpropagate(layer_type: str) -> None:
    torch.manual_seed(7)
    config = _config(layer_type)
    rotary = DeepseekV4RotaryEmbedding(config)
    packed = PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor([3, 5]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    attention = DeepseekV4Attention(config, 0)
    hidden = torch.randn(1, 8, config.hidden_size, requires_grad=True)
    output, _ = attention(hidden, packed)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()

    # Each packed document must match running that document by itself. This checks both
    # directions of isolation and catches compressed layouts that accidentally use global
    # token positions.
    with torch.no_grad():
        packed_output = attention(hidden.detach(), packed)[0]
        first_context = PackedContext.build(
            rotary_emb=rotary,
            seq_lens=torch.tensor([3]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        second_context = PackedContext.build(
            rotary_emb=rotary,
            seq_lens=torch.tensor([5]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        first = attention(hidden.detach()[:, :3], first_context)[0]
        second = attention(hidden.detach()[:, 3:], second_context)[0]
        torch.testing.assert_close(packed_output[:, :3], first, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(packed_output[:, 3:], second, atol=2e-5, rtol=2e-5)

        # Changing one document must leave the other document unchanged.
        changed = hidden.detach().clone()
        changed[:, 3:] += 100.0
        changed_output = attention(changed, packed)[0]
    torch.testing.assert_close(
        packed_output[:, :3], changed_output[:, :3], atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_sparse_attention_invalid_slots_do_not_read_nonfinite_values(nonfinite):
    query = torch.tensor([[[[0.5, -0.25]]]], requires_grad=True)
    values = torch.tensor(
        [[[[nonfinite, nonfinite]], [[2.0, -1.0]]]], requires_grad=True
    )
    indices = torch.tensor([[[[-1, 1]]]])
    sinks = torch.tensor([0.0], requires_grad=True)

    output = _eager_sparse_attention(query, values, indices, sinks, scaling=0.5)
    expected_probability = torch.sigmoid(torch.tensor(0.625))
    expected = expected_probability * torch.tensor([[[[2.0, -1.0]]]])
    torch.testing.assert_close(output, expected)

    output.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(sinks.grad).all()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad[:, 0], torch.zeros_like(values.grad[:, 0]))


def test_rmsnorm_keeps_bfloat16_activations_with_float32_weights():
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.001, 0.999, 1.003, 0.997]))
    value = torch.tensor([[0.125, -0.75, 2.0, 0.375]], dtype=torch.bfloat16)
    value.requires_grad_()

    actual = norm(value)
    expected = (
        value.float()
        * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.eps)
        * norm.weight
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.float().square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(norm.weight.grad).all()

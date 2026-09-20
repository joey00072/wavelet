"""DeepSeek-V4 routing and clamped SwiGLU, adapted from Apache-2.0 PrimeRL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ClampedSwiGLU:
    def __init__(self, limit: float) -> None:
        self.limit = limit

    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.silu(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)


class DeepseekV4Router(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        route_scale: float = 1.0,
        selection_bias: bool = False,
        route_norm: bool = True,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts, self.top_k, self.route_scale = num_experts, top_k, route_scale
        self.route_norm = route_norm
        if selection_bias:
            self.register_buffer(
                "selection_bias", torch.zeros(num_experts), persistent=True
            )
        else:
            self.selection_bias = None

    def forward(
        self, x: Tensor, routed_experts: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scores = F.softplus(self.gate(x).float()).sqrt()
        if routed_experts is None:
            selection_scores = (
                scores if self.selection_bias is None else scores + self.selection_bias
            )
            indices = selection_scores.topk(self.top_k, dim=-1, sorted=True).indices
        else:
            indices = routed_experts
        selected = scores.gather(-1, indices)
        selected_probability_mass = (
            selected / (scores.sum(-1, keepdim=True) + 1e-20)
        ).sum()
        weights = selected
        if self.route_norm:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts)
        # Keep routing weights in FP32 through expert scoring.  The reference
        # applies the score after the expert matmul and casts the final result,
        # avoiding BF16 rounding of fractional top-k weights.
        return weights, indices, counts, selected_probability_mass.detach()


class DeepseekV4HashRouter(DeepseekV4Router):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        vocab_size: int,
        route_scale: float = 1.0,
        route_norm: bool = True,
    ) -> None:
        super().__init__(
            dim,
            num_experts,
            top_k,
            route_scale,
            selection_bias=False,
            route_norm=route_norm,
        )
        self.register_buffer(
            "tid2eid", torch.zeros(vocab_size, top_k, dtype=torch.long), persistent=True
        )


class DeepseekV4Experts(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, num_experts: int, swiglu_limit: float
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.activation = ClampedSwiGLU(swiglu_limit)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, x: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        result = torch.zeros_like(x)
        flat_x, flat_i, flat_w = (
            x.reshape(-1, x.shape[-1]),
            indices.reshape(-1, indices.shape[-1]),
            weights.reshape(-1, weights.shape[-1]),
        )
        for expert in range(self.gate_proj.shape[0]):
            rows, slots = (flat_i == expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            hidden = self.activation(
                F.linear(flat_x[rows], self.gate_proj[expert]),
                F.linear(flat_x[rows], self.up_proj[expert]),
            )
            result.view(-1, result.shape[-1]).index_add_(
                0,
                rows,
                (
                    F.linear(hidden, self.down_proj[expert]) * flat_w[rows, slots, None]
                ).to(result.dtype),
            )
        return result


class DeepseekV4MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False
        )
        self.activation = ClampedSwiGLU(config.swiglu_limit)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.gate_proj(x), self.up_proj(x)))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        if config.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek-V4 only supports scoring_func='sqrtsoftplus'.")
        if config.hidden_act != "silu":
            raise ValueError("DeepSeek-V4 only supports hidden_act='silu'.")
        if config.mlp_bias:
            raise ValueError("DeepSeek-V4 does not support biased MoE projections.")
        self.layer_idx = layer_idx
        self.is_hash = layer_idx < config.num_hash_layers
        self.router = (
            DeepseekV4HashRouter(
                config.hidden_size,
                config.n_routed_experts,
                config.num_experts_per_tok,
                config.vocab_size,
                config.routed_scaling_factor,
                route_norm=True,
            )
            if self.is_hash
            else DeepseekV4Router(
                config.hidden_size,
                config.n_routed_experts,
                config.num_experts_per_tok,
                config.routed_scaling_factor,
                selection_bias=True,
                route_norm=True,
            )
        )
        self.experts = DeepseekV4Experts(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            config.swiglu_limit,
        )
        self.shared_expert = DeepseekV4MLP(config) if config.n_shared_experts else None
        # Usage is optimizer-step state, rather than model state.  Keeping this
        # buffer non-persistent avoids checkpointing stale accumulation and lets
        # the load-balance hook consume counts across gradient accumulation.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.load_balance_coeff = None if self.is_hash else 1e-3

    def forward(
        self,
        x: Tensor,
        input_ids: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        if self.is_hash and routed_experts is None:
            if input_ids is None:
                raise ValueError("hash-routed DeepSeek-V4 layer requires input_ids")
            routed_experts = self.router.tid2eid[input_ids]
        weights, indices, _, _ = self.router(x, routed_experts)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.tokens_per_expert.add_(
                    torch.bincount(
                        indices.reshape(-1), minlength=self.tokens_per_expert.numel()
                    ).to(self.tokens_per_expert.dtype)
                )
        output = self.experts(x, indices, weights)
        return (
            output + self.shared_expert(x) if self.shared_expert is not None else output
        )

# Adapted from PrimeRL (Apache-2.0); see LICENSE in this directory.
# Fused mHC kernels replaced with the upstream eager reference formulas.
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_deepseek_v4 import DeepseekV4Config


def sinkhorn(logits: torch.Tensor, iterations: int, eps: float) -> torch.Tensor:
    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


class DeepseekV4UnweightedRMSNorm(nn.Module):
    """RMS normalization without a learnable gain, computed in fp32."""

    def __init__(self, eps: float = 1e-6, out_dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(
            out_dtype
        )


class DeepseekV4HyperConnection(nn.Module):
    """Manifold-constrained hyper-connection (mHC) around one sublayer.

    The residual, `mhc_states`, is `hc_mult` parallel streams shaped `(B, S, hc_mult, hidden_size)`.
    A single projection of the normalized, flattened streams produces three gates:

    - `pre`: weights that collapse the streams into the single sequence fed to the
      sublayer. Returned already applied, as `collapsed`.
    - `post`: weights in `[0, 2]` that broadcast the sublayer output back over the
      streams. Returned for the caller to apply.
    - `comb`: an `hc_mult x hc_mult` matrix that remixes the streams. It is projected
      onto the doubly-stochastic manifold by Sinkhorn-Knopp (alternating row and column
      normalization), which is what makes signal propagation non-expansive across depth.

    The projection and the Sinkhorn iterations run in fp32; only `collapsed` is cast
    back to the input dtype.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(
            eps=config.rms_norm_eps, out_dtype=torch.float32
        )
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One scale per gate: `pre`, `post`, `comb`.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, mhc_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(mhc_states.flatten(start_dim=2))
        with torch.autocast(device_type=mhc_states.device.type, enabled=False):
            pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
                [hc, hc, hc * hc], dim=-1
            )
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(
            *comb_w.shape[:-1], hc, hc
        ) * comb_scale + comb_b.view(hc, hc)
        comb = sinkhorn(comb_logits, self.hc_sinkhorn_iters, self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * mhc_states).sum(dim=2).to(mhc_states.dtype)
        return post, comb, collapsed

    def update_states(
        self,
        post: torch.Tensor,
        comb: torch.Tensor,
        sublayer_out: torch.Tensor,
        mhc_states: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast the sublayer output over the streams via `post` and remix them via `comb`."""
        dtype = mhc_states.dtype
        return post.to(dtype).unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), mhc_states
        )

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


class DeepseekV4HyperHead(nn.Module):
    """Final collapse of the `hc_mult` residual streams, before the model's last norm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(
            eps=config.rms_norm_eps, out_dtype=torch.float32
        )
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, mhc_states: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(mhc_states.flatten(2))
        with torch.autocast(device_type=mhc_states.device.type, enabled=False):
            mixes = F.linear(flat, self.hc_fn.float())
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.eps
        )
        return (pre.unsqueeze(-1) * mhc_states).sum(dim=2).to(mhc_states.dtype)

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.hc_fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)


__all__ = [
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4UnweightedRMSNorm",
]

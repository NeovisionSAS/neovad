import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    The normalizer used across the Llama / DeepSeek / Qwen families: no mean
    subtraction and no bias, so it is cheaper than LayerNorm while matching its
    quality. Operates over the last dimension and is identical frame-by-frame, so
    it is trivially streaming-safe (no cross-time state).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class RMSNormGated(nn.Module):
    """Gated RMSNorm used as Mamba-2's output gate.

    Applies the SiLU gate *before* normalizing (``norm_before_gate=False`` in the
    reference Mamba-2), i.e. ``rmsnorm(x * silu(gate))``. Pointwise over time, so it
    carries no streaming state.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        x = (x * F.silu(gate.float())).float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)

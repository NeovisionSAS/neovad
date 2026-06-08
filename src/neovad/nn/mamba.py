import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from neovad.nn.conv import CausalDepthwiseConv1d, ConvState
from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.norm import RMSNormGated


class Mamba2State(ModuleState):
    conv: ConvState  # short-conv receptive-field tail
    ssm: Tensor  # [B, n_heads, headdim, d_state] recurrent SSM state


class Mamba2Config(MixerConfig):
    kind: Literal["mamba2"] = "mamba2"
    expand: int = 2
    headdim: int = 64
    d_state: int = 64
    n_groups: int = 1
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_floor: float = 1e-4


class Mamba2Mixer(StreamingMixer):
    """Mamba-2 selective state-space block (Dao & Gu, 2024), pure PyTorch.

    The headline stateful backbone: a constant-size hidden state implicitly tracks
    *who the dominant speaker is*, which is exactly the foreground-locking behaviour
    the project needs and the reason it stays O(1) per step over a multi-minute call
    (attention's KV cache grows with time; this does not).

    Two equivalent paths over one set of weights:

    * ``forward`` uses the **dual quadratic SSD form** — the structured-masked-attention
      view ``scores[t,s] = (C_t . B_s) * exp(cumdecay_t - cumdecay_s)`` (causal) — which
      is exact and fully parallel, ideal for training on short VAD clips.
    * ``step`` uses the linear recurrence ``h = h*exp(dt*A) + (dt*x) (x) B; y = C.h``.

    We deliberately avoid the official ``mamba_ssm`` CUDA/Triton kernels: they do not
    run on CPU, which is the deployment target.
    """

    kind = "mamba2"

    def __init__(self, dim: int, cfg: Mamba2Config, depth: int = 1):
        super().__init__(dim)
        d_inner = cfg.expand * dim
        if d_inner % cfg.headdim != 0:
            raise ValueError(f"expand*dim ({d_inner}) not divisible by headdim ({cfg.headdim})")
        self.d_inner = d_inner
        self.headdim = cfg.headdim
        self.n_heads = d_inner // cfg.headdim
        self.n_groups = cfg.n_groups
        self.d_state = cfg.d_state
        self.conv_dim = d_inner + 2 * cfg.n_groups * cfg.d_state

        self.in_proj = nn.Linear(
            dim, 2 * d_inner + 2 * cfg.n_groups * cfg.d_state + self.n_heads, bias=False
        )
        self.conv = CausalDepthwiseConv1d(self.conv_dim, cfg.d_conv, bias=True)
        self.norm = RMSNormGated(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def split_projection(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, xBC, dt = self.in_proj(x).split([self.d_inner, self.conv_dim, self.n_heads], dim=-1)
        return z, xBC, dt

    def split_xBC(self, xBC: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gn = self.n_groups * self.d_state
        x, b, c = xBC.split([self.d_inner, gn, gn], dim=-1)
        x = rearrange(x, "b t (h p) -> b t h p", h=self.n_heads)
        b = self.heads(rearrange(b, "b t (g n) -> b t g n", g=self.n_groups))
        c = self.heads(rearrange(c, "b t (g n) -> b t g n", g=self.n_groups))
        return x, b, c

    def heads(self, gbc: Tensor) -> Tensor:
        # [B, T, G, N] -> [B, T, H, N] (each group shared across its head block)
        if self.n_groups == self.n_heads:
            return gbc
        return gbc.repeat_interleave(self.n_heads // self.n_groups, dim=2)

    def forward(self, x: Tensor) -> Tensor:
        z, xBC, dt_raw = self.split_projection(x)
        xBC = F.silu(self.conv(xBC))
        xi, b, c = self.split_xBC(xBC)  # xi:[B,T,H,P]; b,c:[B,T,H,N]
        dt = F.softplus(dt_raw + self.dt_bias)  # [B, T, H]
        a = -torch.exp(self.A_log)  # [H], negative
        decay = dt * a  # [B, T, H]
        cum = decay.cumsum(dim=1)  # [B, T, H] cumulative log-decay

        cb = torch.einsum("bthn,bshn->bhts", c, b)  # [B, H, T, T]
        cum_h = cum.permute(0, 2, 1)  # [B, H, T]
        rel = cum_h[..., :, None] - cum_h[..., None, :]  # [B, H, T, T] = cum_t - cum_s
        t = x.shape[1]
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = cb * torch.exp(rel) * mask
        vals = dt[..., None] * xi  # [B, T, H, P]
        y = torch.einsum("bhts,bshp->bthp", scores, vals)  # [B, T, H, P]
        y = y + self.D[None, None, :, None] * xi
        y = rearrange(y, "b t h p -> b t (h p)")
        return self.out_proj(self.norm(y, z))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Mamba2State:
        return Mamba2State(
            conv=self.conv.init_state(batch, device, dtype),
            ssm=torch.zeros(
                batch, self.n_heads, self.headdim, self.d_state, device=device, dtype=dtype
            ),
        )

    def step(self, x: Tensor, state: Mamba2State) -> Tensor:
        z, xBC, dt_raw = self.split_projection(x)
        xBC = F.silu(self.conv.step(xBC, state.conv))
        xi, b, c = self.split_xBC(xBC)
        xi, b, c = xi[:, 0], b[:, 0], c[:, 0]  # drop time -> [B,H,P], [B,H,N]
        dt = F.softplus(dt_raw[:, 0] + self.dt_bias)  # [B, H]
        a = torch.exp(dt * -torch.exp(self.A_log))  # [B, H] discrete decay
        state.ssm = (
            state.ssm * a[:, :, None, None] + (dt[:, :, None] * xi)[..., None] * b[:, :, None, :]
        )
        y = (state.ssm * c[:, :, None, :]).sum(-1) + self.D[None, :, None] * xi  # [B, H, P]
        y = rearrange(y, "b h p -> b () (h p)")
        return self.out_proj(self.norm(y, z))

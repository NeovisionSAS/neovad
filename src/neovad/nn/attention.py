import math
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.norm import RMSNorm
from neovad.nn.rope import RotaryEmbedding


class AttentionMixer(StreamingMixer):
    """Shared machinery for causal, sliding-window self-attention mixers.

    Owns the rotary embedding, the scaled-softmax reduction, and the
    causal + sliding-window mask. Subclasses (:class:`GQAMixer`,
    :class:`MLAMixer`) only decide how ``q/k/v`` are produced and what is cached for
    streaming. A finite ``window`` bounds both compute and the streaming cache, which
    is exactly what keeps a long phone call from growing unbounded state.
    """

    def __init__(self, dim: int, window: int, scale: float):
        super().__init__(dim)
        self.window = window
        self.scale = scale

    def causal_window_mask(self, t: int, device: torch.device) -> Tensor:
        i = torch.arange(t, device=device)[:, None]
        j = torch.arange(t, device=device)[None, :]
        return (j <= i) & (i - j < self.window)  # [T, T] bool, True = attend

    def reduce(self, scores: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
        # scores: [B, H, Tq, Tk]; value: [B, H, Tk, Dv]; mask: [Tq, Tk] bool or None
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return attn @ value


class GQAState(ModuleState):
    k: Tensor  # [B, n_kv, L, head_dim] (already rotary-applied)
    v: Tensor  # [B, n_kv, L, head_dim]
    pos: int


class GQAConfig(MixerConfig):
    kind: Literal["gqa"] = "gqa"
    n_heads: int = 4
    n_kv_heads: int = 1
    window: int = 128
    rope_base: float = 10000.0


class GQAMixer(AttentionMixer):
    """Grouped-query attention with sliding window + RoPE.

    ``n_kv_heads < n_heads`` shrinks the KV cache (and CPU memory traffic) by sharing
    each key/value head across a group of query heads (Ainslie et al., 2023) — the
    standard small-model attention since Llama-2/Mistral.
    """

    kind = "gqa"

    def __init__(self, dim: int, cfg: GQAConfig, depth: int = 1):
        if dim % cfg.n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {cfg.n_heads}")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError(f"n_heads {cfg.n_heads} not divisible by n_kv_heads {cfg.n_kv_heads}")
        head_dim = dim // cfg.n_heads
        super().__init__(dim, cfg.window, scale=head_dim**-0.5)
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(dim, cfg.n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, cfg.n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, cfg.n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * head_dim, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim, base=cfg.rope_base)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(self.wk(x), "b t (h d) -> b h t d", h=self.n_kv)
        v = rearrange(self.wv(x), "b t (h d) -> b h t d", h=self.n_kv)
        return q, k, v

    def expand_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.groups == 1:
            return k, v
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        return k, v

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        q, k, v = self.project(x)
        q = self.rope.apply_rotary(q, pos)
        k = self.rope.apply_rotary(k, pos)
        k, v = self.expand_kv(k, v)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        out = self.reduce(scores, v, self.causal_window_mask(t, x.device))
        return self.wo(rearrange(out, "b h t d -> b t (h d)"))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> GQAState:
        empty = torch.zeros(batch, self.n_kv, 0, self.head_dim, device=device, dtype=dtype)
        return GQAState(k=empty, v=empty.clone(), pos=0)

    def step(self, x: Tensor, state: GQAState) -> Tensor:
        pos = torch.tensor([state.pos], device=x.device)
        q, k, v = self.project(x)
        q = self.rope.apply_rotary(q, pos)
        k = self.rope.apply_rotary(k, pos)
        state.k = torch.cat([state.k, k], dim=2)[:, :, -self.window :]
        state.v = torch.cat([state.v, v], dim=2)[:, :, -self.window :]
        state.pos += 1
        k_e, v_e = self.expand_kv(state.k, state.v)
        scores = (q @ k_e.transpose(-1, -2)) * self.scale
        out = self.reduce(scores, v_e, mask=None)
        return self.wo(rearrange(out, "b h t d -> b t (h d)"))


class MLAState(ModuleState):
    c_kv: Tensor  # [B, L, kv_lora_rank] compressed latent
    k_rope: Tensor  # [B, L, qk_rope_dim] decoupled rotary key (rotary-applied)
    pos: int


class MLAConfig(MixerConfig):
    kind: Literal["mla"] = "mla"
    n_heads: int = 4
    qk_nope_dim: int = 0  # per head; 0 -> dim // n_heads
    qk_rope_dim: int = 16  # decoupled rotary key dim (even), shared across heads
    v_head_dim: int = 0  # 0 -> dim // n_heads
    kv_lora_rank: int = 0  # 0 -> max(32, dim // 2)
    window: int = 128
    rope_base: float = 10000.0


class MLAMixer(AttentionMixer):
    """Multi-head Latent Attention (DeepSeek-V2/V3).

    Keys and values are jointly compressed into a low-rank latent ``c_kv`` plus a
    small head-shared rotary key ``k_rope``; only those two are cached at inference,
    making the streaming KV state several times smaller than GQA's for the same head
    count. Per-head keys/values are reconstructed on demand from the latent.
    """

    kind = "mla"

    def __init__(self, dim: int, cfg: MLAConfig, depth: int = 1):
        head_dim = dim // cfg.n_heads
        dn = cfg.qk_nope_dim or head_dim
        dv = cfg.v_head_dim or head_dim
        dr = cfg.qk_rope_dim
        r = cfg.kv_lora_rank or max(32, dim // 2)
        if dr % 2 != 0:
            raise ValueError(f"qk_rope_dim must be even, got {dr}")
        super().__init__(dim, cfg.window, scale=(dn + dr) ** -0.5)
        self.n_heads = cfg.n_heads
        self.dn, self.dv, self.dr, self.r = dn, dv, dr, r
        self.wq = nn.Linear(dim, cfg.n_heads * (dn + dr), bias=False)
        self.w_dkv = nn.Linear(dim, r, bias=False)  # down-project to latent
        self.w_kr = nn.Linear(dim, dr, bias=False)  # decoupled rotary key (shared)
        self.w_uk = nn.Linear(r, cfg.n_heads * dn, bias=False)  # latent -> per-head nope key
        self.w_uv = nn.Linear(r, cfg.n_heads * dv, bias=False)  # latent -> per-head value
        self.wo = nn.Linear(cfg.n_heads * dv, dim, bias=False)
        self.rope = RotaryEmbedding(dr, base=cfg.rope_base)

    def split_query(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.n_heads)
        q_nope, q_rope = q.split([self.dn, self.dr], dim=-1)
        return q_nope, self.rope.apply_rotary(q_rope, pos)

    def reconstruct_kv(self, c_kv: Tensor) -> tuple[Tensor, Tensor]:
        k_nope = rearrange(self.w_uk(c_kv), "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(self.w_uv(c_kv), "b t (h d) -> b h t d", h=self.n_heads)
        return k_nope, v

    def combine(self, q_nope: Tensor, q_rope: Tensor, k_nope: Tensor, k_rope: Tensor) -> Tensor:
        # k_rope: [B, Tk, dr] shared across heads -> broadcast
        s_nope = q_nope @ k_nope.transpose(-1, -2)  # [B, H, Tq, Tk]
        s_rope = torch.einsum("bhqd,bkd->bhqk", q_rope, k_rope)
        return (s_nope + s_rope) * self.scale

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        q_nope, q_rope = self.split_query(x, pos)
        c_kv = self.w_dkv(x)
        k_rope = self.rope.apply_rotary(self.w_kr(x), pos)
        k_nope, v = self.reconstruct_kv(c_kv)
        scores = self.combine(q_nope, q_rope, k_nope, k_rope)
        out = self.reduce(scores, v, self.causal_window_mask(t, x.device))
        return self.wo(rearrange(out, "b h t d -> b t (h d)"))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> MLAState:
        return MLAState(
            c_kv=torch.zeros(batch, 0, self.r, device=device, dtype=dtype),
            k_rope=torch.zeros(batch, 0, self.dr, device=device, dtype=dtype),
            pos=0,
        )

    def step(self, x: Tensor, state: MLAState) -> Tensor:
        pos = torch.tensor([state.pos], device=x.device)
        q_nope, q_rope = self.split_query(x, pos)
        c_kv = self.w_dkv(x)
        k_rope = self.rope.apply_rotary(self.w_kr(x), pos)
        state.c_kv = torch.cat([state.c_kv, c_kv], dim=1)[:, -self.window :]
        state.k_rope = torch.cat([state.k_rope, k_rope], dim=1)[:, -self.window :]
        state.pos += 1
        k_nope, v = self.reconstruct_kv(state.c_kv)
        scores = self.combine(q_nope, q_rope, k_nope, state.k_rope)
        out = self.reduce(scores, v, mask=None)
        return self.wo(rearrange(out, "b h t d -> b t (h d)"))


class DiffAttnState(ModuleState):
    k: Tensor  # [B, 2*n_heads, L, head_dim] (rotary-applied)
    v: Tensor  # [B, n_heads, L, 2*head_dim]
    pos: int


class DiffAttnConfig(MixerConfig):
    kind: Literal["diffattn"] = "diffattn"
    n_heads: int = 2  # differential heads; each consumes two softmax maps
    window: int = 128
    rope_base: float = 10000.0


class DiffAttnMixer(AttentionMixer):
    """Differential Attention (Ye et al., 2024, "Differential Transformer").

    The attention map is the difference of two softmaxes,
    ``softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)``. Like a differential amplifier,
    the common mode shared by both maps — diffuse attention to background noise and
    secondary voices — cancels, while the foreground signal that only the first map
    locks onto survives. That is precisely this project's failure mode (leaked
    background voices), which is why this is the primary attention backbone.
    """

    kind = "diffattn"

    def __init__(self, dim: int, cfg: DiffAttnConfig, depth: int = 1):
        if dim % (2 * cfg.n_heads) != 0:
            raise ValueError(f"dim {dim} not divisible by 2*n_heads {2 * cfg.n_heads}")
        head_dim = dim // (2 * cfg.n_heads)
        if head_dim % 2 != 0:
            raise ValueError(f"diffattn head_dim {head_dim} must be even for RoPE")
        super().__init__(dim, cfg.window, scale=head_dim**-0.5)
        self.n_heads = cfg.n_heads
        self.head_dim = head_dim
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))
        self.wq = nn.Linear(dim, dim, bias=False)  # 2*n_heads*head_dim
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)  # n_heads*(2*head_dim)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(head_dim, base=cfg.rope_base)
        self.subnorm = RMSNorm(2 * head_dim)
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim) * 0.1)

    def lambda_value(self) -> Tensor:
        l1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        l2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        return l1 - l2 + self.lambda_init

    def project(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=2 * self.n_heads)
        k = rearrange(self.wk(x), "b t (h d) -> b h t d", h=2 * self.n_heads)
        v = rearrange(self.wv(x), "b t (h d) -> b h t d", h=self.n_heads)
        return self.rope.apply_rotary(q, pos), self.rope.apply_rotary(k, pos), v

    def differential(self, scores: Tensor, v: Tensor) -> Tensor:
        # scores: [B, 2H, Tq, Tk] (already masked); v: [B, H, Tk, 2*head_dim]
        attn = scores.softmax(dim=-1)
        attn = rearrange(attn, "b (h two) tq tk -> b h two tq tk", two=2)
        attn = attn[:, :, 0] - self.lambda_value() * attn[:, :, 1]  # [B, H, Tq, Tk]
        out = attn @ v  # [B, H, Tq, 2*head_dim]
        out = self.subnorm(out) * (1.0 - self.lambda_init)
        return self.wo(rearrange(out, "b h t d -> b t (h d)"))

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        q, k, v = self.project(x, pos)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        mask = self.causal_window_mask(t, x.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        return self.differential(scores, v)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> DiffAttnState:
        k = torch.zeros(batch, 2 * self.n_heads, 0, self.head_dim, device=device, dtype=dtype)
        v = torch.zeros(batch, self.n_heads, 0, 2 * self.head_dim, device=device, dtype=dtype)
        return DiffAttnState(k=k, v=v, pos=0)

    def step(self, x: Tensor, state: DiffAttnState) -> Tensor:
        pos = torch.tensor([state.pos], device=x.device)
        q, k, v = self.project(x, pos)
        state.k = torch.cat([state.k, k], dim=2)[:, :, -self.window :]
        state.v = torch.cat([state.v, v], dim=2)[:, :, -self.window :]
        state.pos += 1
        scores = (q @ state.k.transpose(-1, -2)) * self.scale
        return self.differential(scores, state.v)

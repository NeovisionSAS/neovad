import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021), the de-facto positional scheme
    in modern transformers (Llama / Qwen / DeepSeek).

    Frequencies are computed on the fly from an explicit ``positions`` tensor rather
    than a precomputed table, so a streaming decoder can keep advancing the absolute
    frame index without any maximum-length ceiling. Because RoPE is relative by
    construction (a query-key dot product depends only on the position *difference*),
    the same code serves both the parallel ``forward`` path (``positions = 0..T-1``)
    and the recurrent ``step`` path (``positions = [t]``).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x: Tensor, positions: Tensor) -> Tensor:
        # x: [..., T, dim]; positions: [T] (int). Returns x rotated per position.
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [T, dim/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, dim]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        return (x * cos + self.rotate_half(x) * sin).to(x.dtype)

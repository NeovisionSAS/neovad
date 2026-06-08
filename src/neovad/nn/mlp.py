import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """Gated MLP with a SiLU gate (Shazeer, 2020), as used in Llama / DeepSeek.

    Beats a plain ``Linear -> GELU -> Linear`` FFN at equal parameter count. The
    hidden width is set so the three projections roughly match a ``4 * dim`` plain
    FFN: ``hidden = round(mult * dim * 2/3)`` snapped to a multiple of ``round_to``.
    Pointwise over time, hence streaming-safe with no state.
    """

    def __init__(self, dim: int, mult: float = 4.0, round_to: int = 32):
        super().__init__()
        hidden = int(mult * dim * 2 / 3)
        hidden = round_to * ((hidden + round_to - 1) // round_to)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

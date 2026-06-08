import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neovad.nn.mixer import ModuleState


class ConvState(ModuleState):
    tail: Tensor  # [B, kernel-1, C] the receptive-field tail carried across chunks


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise 1-D conv with a left-only receptive field and a streaming tail buffer.

    One module concentrates the "cache the last ``kernel-1`` frames" logic reused by
    the front-end stem, the Mamba-2 short conv, and the optional separable-conv stage.
    ``forward`` left-pads the whole sequence; ``step`` prepends the carried tail to the
    incoming frame so a single-frame call sees the same left context — giving identical
    results to ``forward`` on the matching prefix.
    """

    def __init__(self, channels: int, kernel: int, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        y = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(y).transpose(1, 2)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> ConvState:
        tail = torch.zeros(batch, self.kernel - 1, self.channels, device=device, dtype=dtype)
        return ConvState(tail=tail)

    def step(self, x: Tensor, state: ConvState) -> Tensor:
        # x: [B, n, C] for n >= 1 frames
        ctx = torch.cat([state.tail, x], dim=1)  # [B, kernel-1+n, C]
        y = self.conv(ctx.transpose(1, 2)).transpose(1, 2)  # valid conv -> [B, n, C]
        state.tail = ctx[:, -(self.kernel - 1) :] if self.kernel > 1 else state.tail
        return y

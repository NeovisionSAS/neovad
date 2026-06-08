import torch
from torch import Tensor, nn

from neovad.nn.block import ResidualBlock
from neovad.nn.mixer import MixerConfig, ModuleState
from neovad.nn.norm import RMSNorm


class Backbone(nn.Module):
    """A stack of identical :class:`ResidualBlock` s plus a final norm.

    Every block is built from the same ``mixer_cfg`` but receives its 1-based depth so
    schedule-dependent mixers (Differential Attention) initialise per layer. The
    backbone's streaming state is just the ordered list of its blocks' states.
    """

    def __init__(self, dim: int, depth: int, mixer_cfg: MixerConfig, mlp_mult: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim, mixer_cfg, depth=i + 1, mlp_mult=mlp_mult) for i in range(depth)]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[ModuleState]:
        return [block.init_state(batch, device, dtype) for block in self.blocks]

    def step(self, x: Tensor, states: list[ModuleState]) -> Tensor:
        for block, state in zip(self.blocks, states, strict=True):
            x = block.step(x, state)
        return self.norm(x)

import torch
from torch import Tensor, nn

from neovad.nn.mixer import MixerConfig, ModuleState
from neovad.nn.mlp import SwiGLU
from neovad.nn.norm import RMSNorm


class ResidualBlock(nn.Module):
    """The single block shape shared by every backbone: pre-norm sequence mixer with a
    residual, then pre-norm SwiGLU FFN with a residual (the Llama / DeepSeek block).

    The only thing that varies between backbones is which ``StreamingMixer`` the
    ``mixer_cfg`` builds; norms and FFN are identical and stateless, so the block's
    entire streaming state is the mixer's state.
    """

    def __init__(self, dim: int, mixer_cfg: MixerConfig, depth: int, mlp_mult: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = mixer_cfg.build(dim, depth)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> ModuleState:
        return self.mixer.init_state(batch, device, dtype)

    def step(self, x: Tensor, state: ModuleState) -> Tensor:
        x = x + self.mixer.step(self.norm1(x), state)
        return x + self.mlp(self.norm2(x))

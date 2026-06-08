from typing import Literal

import torch
from torch import Tensor, nn

from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer


class GRUState(ModuleState):
    h: Tensor  # [num_layers, B, hidden]


class GRUConfig(MixerConfig):
    kind: Literal["gru"] = "gru"
    hidden: int = 128
    num_layers: int = 1


class GRUMixer(StreamingMixer):
    """Causal GRU temporal mixer — the proven-simple baseline (Silero / Personal-VAD).

    The control every modern backbone is measured against on the same harness:
    O(1)-per-step constant hidden state, trivial CPU cost, clean int8 export.
    forward and step are exactly equivalent because the GRU is unidirectional and
    its recurrence is the same in both paths.
    """

    kind = "gru"

    def __init__(self, dim: int, cfg: GRUConfig, depth: int = 1):
        super().__init__(dim)
        self.hidden = cfg.hidden
        self.num_layers = cfg.num_layers
        self.gru = nn.GRU(dim, cfg.hidden, cfg.num_layers, batch_first=True)
        self.out = nn.Linear(cfg.hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.gru(x)
        return self.out(y)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> GRUState:
        h = torch.zeros(self.num_layers, batch, self.hidden, device=device, dtype=dtype)
        return GRUState(h=h)

    def step(self, x: Tensor, state: GRUState) -> Tensor:
        y, h = self.gru(x, state.h)
        state.h = h
        return self.out(y)

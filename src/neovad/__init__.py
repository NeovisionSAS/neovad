"""neovad — small, streaming, CPU-friendly voice activity detection with pluggable
modern backbones and foreground-speaker gating.

Public surface:
    VADModel        the model (frontend + backbone + head); train and stream on one set of weights
    StreamingVAD    frame-by-frame inference wrapper with a hysteresis gate
    NeoVADConfig    the full typed config (model / data / train)
    train(cfg)      fit a model (requires the optional `train` extra: lightning)
"""

from neovad.config import DataConfig, ModelConfig, NeoVADConfig, TrainConfig
from neovad.infer.stream import HysteresisGate, StreamingVAD
from neovad.models.vad import VADModel, VADState
from neovad.nn.head import SpeechClass

__version__ = "0.1.0"


def train(cfg: NeoVADConfig) -> VADModel:
    # Lightning is an optional, heavy dependency (the `train` extra) — import on demand.
    from neovad.train.lit import NeoVADLit

    return NeoVADLit.run(cfg)


__all__ = [
    "DataConfig",
    "HysteresisGate",
    "ModelConfig",
    "NeoVADConfig",
    "SpeechClass",
    "StreamingVAD",
    "TrainConfig",
    "VADModel",
    "VADState",
    "train",
]

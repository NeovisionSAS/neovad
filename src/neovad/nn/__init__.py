"""Neural building blocks. Importing this package registers every mixer and head
into their registries (the import side effect is what populates ``StreamingMixer`` and
``VADHead``), so config-driven construction can look them up by ``kind``."""

from neovad.nn.attention import (
    DiffAttnConfig,
    DiffAttnMixer,
    GQAConfig,
    GQAMixer,
    MLAConfig,
    MLAMixer,
)
from neovad.nn.block import ResidualBlock
from neovad.nn.conv import CausalDepthwiseConv1d
from neovad.nn.gru import GRUConfig, GRUMixer
from neovad.nn.head import (
    AttractorHead,
    AttractorHeadConfig,
    HeadConfig,
    LinearHead,
    LinearHeadConfig,
    SpeechClass,
    VADHead,
)
from neovad.nn.mamba import Mamba2Config, Mamba2Mixer
from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.mlp import SwiGLU
from neovad.nn.norm import RMSNorm, RMSNormGated
from neovad.nn.rope import RotaryEmbedding

__all__ = [
    "AttractorHead",
    "AttractorHeadConfig",
    "CausalDepthwiseConv1d",
    "DiffAttnConfig",
    "DiffAttnMixer",
    "GQAConfig",
    "GQAMixer",
    "GRUConfig",
    "GRUMixer",
    "HeadConfig",
    "LinearHead",
    "LinearHeadConfig",
    "MLAConfig",
    "MLAMixer",
    "Mamba2Config",
    "Mamba2Mixer",
    "MixerConfig",
    "ModuleState",
    "RMSNorm",
    "RMSNormGated",
    "ResidualBlock",
    "RotaryEmbedding",
    "SpeechClass",
    "StreamingMixer",
    "SwiGLU",
    "VADHead",
]

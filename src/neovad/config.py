from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field

from neovad.frontend.mel import FrontendConfig
from neovad.nn.attention import DiffAttnConfig, GQAConfig, MLAConfig
from neovad.nn.gru import GRUConfig
from neovad.nn.head import AttractorHeadConfig, LinearHeadConfig
from neovad.nn.mamba import Mamba2Config

# The single pluggable axis: a typed discriminated union over every registered mixer.
# Selecting a backbone is setting `mixer.kind`; nothing else in the model changes.
MixerCfg = Annotated[
    GRUConfig | GQAConfig | MLAConfig | DiffAttnConfig | Mamba2Config,
    Field(discriminator="kind"),
]
HeadCfg = Annotated[LinearHeadConfig | AttractorHeadConfig, Field(discriminator="kind")]


class ModelConfig(BaseModel):
    dim: int = 128
    depth: int = 4
    mlp_mult: float = 3.0
    frontend: FrontendConfig = FrontendConfig()
    mixer: MixerCfg = Mamba2Config()
    head: HeadCfg = AttractorHeadConfig()


class DataConfig(BaseModel):
    root: str = "/disk/manual"
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    batch_size: int = 32
    num_workers: int = 8
    steps_per_epoch: int = 2000  # the synth stream is infinite; this bounds an epoch
    # on-the-fly mixing
    speech_sources: list[str] = ["librispeech"]
    noise_sources: list[str] = ["musan", "dns5"]
    min_interferers: int = 0
    max_interferers: int = 3
    interferer_gain: tuple[float, float] = (0.1, 0.8)
    snr_db: tuple[float, float] = (-5.0, 20.0)
    p_noise: float = 0.8
    p_interferer: float = 0.7
    p_rir: float = 0.5
    p_telephony: float = 0.5  # 8 kHz mu-law round-trip
    label_db_threshold: float = -40.0  # primary-reference energy gate for frame labels
    label_smooth_frames: int = 5  # median smoothing of the derived labels


class LossConfig(BaseModel):
    # Cost-sensitive: upweight PRIMARY frames so the model is conservative about
    # firing on anyone but the locked foreground speaker (Personal-VAD spirit).
    class_weights: list[float] = [1.0, 2.0, 1.0]  # [non-speech, primary, secondary]
    label_smoothing: float = 0.0


class TrainConfig(BaseModel):
    output_dir: str = "runs"
    max_epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16-mixed"
    devices: int = 1
    accumulate: int = 1
    val_interval: float = 1.0
    seed: int = 0
    loss: LossConfig = LossConfig()


class NeoVADConfig(BaseModel):
    name: str = "neovad"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))

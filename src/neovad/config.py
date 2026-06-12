from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field

from neovad.frontend.mel import FrontendConfig
from neovad.nn.attention import DiffAttnConfig, GQAConfig, MLAConfig
from neovad.nn.convmixer import ConvMixerConfig
from neovad.nn.deltanet import GatedDeltaNetConfig
from neovad.nn.gru import GRUConfig
from neovad.nn.head import AttractorHeadConfig, LinearHeadConfig
from neovad.nn.linear_rnn import MinGRUConfig, RGLRUConfig
from neovad.nn.mamba import Mamba2Config
from neovad.nn.specaug import SpecAugmentConfig

# The single pluggable axis: a typed discriminated union over every registered mixer.
# Selecting a backbone is setting `mixer.kind`; nothing else in the model changes.
MixerCfg = Annotated[
    GRUConfig
    | GQAConfig
    | MLAConfig
    | DiffAttnConfig
    | Mamba2Config
    | RGLRUConfig
    | MinGRUConfig
    | ConvMixerConfig
    | GatedDeltaNetConfig,
    Field(discriminator="kind"),
]
HeadCfg = Annotated[LinearHeadConfig | AttractorHeadConfig, Field(discriminator="kind")]


class ModelConfig(BaseModel):
    dim: int = 128
    depth: int = 4
    mlp_mult: float = 3.0
    frontend: FrontendConfig = FrontendConfig()
    mixer: MixerCfg = Mamba2Config()
    # Optional per-layer pattern cycled over depth (hybrid stacks, Samba/Zamba style);
    # when set it overrides `mixer`, e.g. [{kind: mamba2}, {kind: diffattn}].
    mixers: list[MixerCfg] | None = None
    head: HeadCfg = AttractorHeadConfig()
    spec_augment: SpecAugmentConfig = SpecAugmentConfig()

    @property
    def layer_mixers(self) -> list[MixerCfg]:
        return self.mixers or [self.mixer]


class DiarizedSource(BaseModel):
    """A HuggingFace diarization dataset (audio + timestamps_start/end + speakers) used
    as real human-labelled speech/non-speech audio. Splits must be disjoint from any
    used for evaluation."""

    repo: str
    config: str | None = None
    split: str = "train"


class DataConfig(BaseModel):
    root: str = "/disk/manual"
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    batch_size: int = 32
    num_workers: int = 8
    steps_per_epoch: int = 2000  # the synth stream is infinite; this bounds an epoch
    # on-the-fly mixing. MUSAN supplies noise + music + babble-speech; interferer
    # voices are drawn from speech_sources. Heavier real-noise/overlap sets
    # (DNS5, AMI, FSD50K) are optional opt-ins documented in docs/ARCHITECTURE.md.
    speech_sources: list[str] = ["librispeech"]
    # Pool for interfering voices; None -> speech_sources. Keep crosstalk-y corpora
    # (AMI meetings) HERE and out of speech_sources: a "primary" clip containing
    # background voices poisons the foreground labels, but an interferer containing
    # crosstalk is just more realistic interference.
    interferer_sources: list[str] | None = None
    noise_sources: list[str] = ["musan"]
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
    # Real human-labelled audio mixed into training as an any-speech objective; 0
    # disables. Closes the synthetic->real domain gap the neutral eval exposes. Sources
    # are diverse + representative: broadcast conversation (VoxConverse dev), meetings
    # (AMI train, close + distant mic), and native-8kHz medical phone calls (simsamu).
    real_windows: int = 0
    real_cache: str = "/disk/manual/real_labeled.pt"
    real_sources: list[DiarizedSource] = [
        DiarizedSource(repo="diarizers-community/voxconverse", split="dev"),
        DiarizedSource(repo="diarizers-community/ami", config="ihm", split="train"),
        DiarizedSource(repo="diarizers-community/ami", config="sdm", split="train"),
        DiarizedSource(repo="diarizers-community/simsamu", split="train"),
    ]
    # deployment-condition augmentation applied to real windows at train time, so the
    # real-audio objective sees phone-like audio, not just clean wideband recordings.
    real_p_telephony: float = 0.5
    real_p_noise: float = 0.6


class LossConfig(BaseModel):
    # Cost-sensitive: upweight PRIMARY frames so the model is conservative about
    # firing on anyone but the locked foreground speaker (Personal-VAD spirit).
    class_weights: list[float] = [1.0, 2.0, 1.0]  # [non-speech, primary, secondary]
    label_smoothing: float = 0.0
    real_weight: float = 0.5  # weight of the real-audio any-speech BCE term


class LogConfig(BaseModel):
    # Controls the rich TensorBoard analysis logging (audio, mel, label-vs-pred).
    media: bool = True  # log audio + spectrogram + prediction figures
    media_every_n_epochs: int = 1
    n_media_samples: int = 6  # examples rendered per media log
    histograms: bool = True  # weight/grad histograms under hist/


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
    log: LogConfig = LogConfig()


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

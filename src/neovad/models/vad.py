from importlib import resources
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn

from neovad.config import ModelConfig, NeoVADConfig
from neovad.frontend.mel import FrontendState, MelFrontend
from neovad.models.backbone import Backbone
from neovad.nn.mixer import ModuleState


class VADState(ModuleState):
    frontend: FrontendState
    layers: list[ModuleState]


class VADModel(nn.Module):
    """The full VAD: ``frontend -> input projection -> backbone -> per-frame head``.

    One set of weights, two execution paths held equivalent by the streaming contract:
    ``forward(waveform)`` for parallel training and ``step(chunk, state)`` for
    frame-by-frame serving. The head emits per-frame logits over
    ``{non-speech, primary, secondary}``; ``speech_probability`` extracts the
    foreground-gating signal.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = MelFrontend(cfg.frontend)
        self.input_proj = nn.Linear(cfg.frontend.n_mels, cfg.dim)
        self.backbone = Backbone(cfg.dim, cfg.depth, cfg.mixer, cfg.mlp_mult)
        self.head = cfg.head.build(cfg.dim)

    @classmethod
    def from_config(cls, cfg: ModelConfig | NeoVADConfig | str | Path) -> Self:
        if isinstance(cfg, str | Path):
            cfg = NeoVADConfig.load(cfg)
        if isinstance(cfg, NeoVADConfig):
            cfg = cfg.model
        return cls(cfg)

    @classmethod
    def from_backbone(cls, name: str) -> Self:
        # Default-everything model with the named mixer — for quick benchmarking.
        return cls(ModelConfig(mixer={"kind": name}))

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> Self:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ModelConfig.model_validate(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model

    @staticmethod
    def pretrained_names() -> list[str]:
        weights = resources.files("neovad").joinpath("weights")
        if not weights.is_dir():
            return []
        return sorted(f.name[:-3] for f in weights.iterdir() if f.name.endswith(".pt"))

    @classmethod
    def from_pretrained(cls, name: str = "mamba2", map_location: str = "cpu") -> Self:
        """Load weights bundled with the installed package — `pip install` then
        `VADModel.from_pretrained()` gives a ready-to-use model, no download."""
        path = resources.files("neovad").joinpath("weights", f"{name}.pt")
        if not path.is_file():
            raise FileNotFoundError(
                f"no bundled weights {name!r}; available: {cls.pretrained_names()}"
            )
        with resources.as_file(path) as real:
            return cls.load(real, map_location=map_location)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        torch.save({"config": self.cfg.model_dump(), "state_dict": self.state_dict()}, path)

    def forward(self, wav: Tensor) -> Tensor:
        # wav: [B, S] -> logits [B, n_frames, n_classes]
        h = self.input_proj(self.frontend(wav))
        return self.head(self.backbone(h))

    def speech_probability(self, logits: Tensor) -> Tensor:
        return self.head.speech_probability(logits)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> VADState:
        return VADState(
            frontend=self.frontend.init_state(batch, device, dtype),
            layers=self.backbone.init_state(batch, device, dtype),
        )

    def step(self, chunk: Tensor, state: VADState) -> Tensor:
        # chunk: [B, n_samples] -> logits [B, n_new_frames, n_classes]
        mels = self.frontend.step(chunk, state.frontend)
        if mels.shape[1] == 0:
            return torch.zeros(
                chunk.shape[0], 0, self.head.n_classes, device=chunk.device, dtype=chunk.dtype
            )
        frames = [
            self.head(self.backbone.step(self.input_proj(mels[:, t : t + 1]), state.layers))
            for t in range(mels.shape[1])
        ]
        return torch.cat(frames, dim=1)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from neovad.models.vad import VADModel


class HysteresisGate:
    """Two-threshold speech gate with onset debounce and a hang-over tail.

    Kept deliberately OUTSIDE the model weights: the network emits an honest per-frame
    foreground probability with zero look-ahead, and the latency/stability trade-off
    (Silero's weakness) is tuned here, not baked into training. Onset needs
    ``min_speech_frames`` consecutive frames above ``on``; offset needs
    ``hang_frames`` below ``off`` — so a single noisy frame neither opens nor closes
    the gate.
    """

    def __init__(self, on: float, off: float, min_speech_frames: int, hang_frames: int):
        self.on = on
        self.off = off
        self.min_speech_frames = min_speech_frames
        self.hang_frames = hang_frames
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._on_count = 0
        self._hang = 0

    def update(self, prob: float) -> bool:
        if self.speaking:
            self._hang = self._hang + 1 if prob < self.off else 0
            if self._hang >= self.hang_frames:
                self.speaking = False
                self._on_count = 0
        else:
            self._on_count = self._on_count + 1 if prob >= self.on else 0
            if self._on_count >= self.min_speech_frames:
                self.speaking = True
                self._hang = 0
        return self.speaking


class StreamingVAD:
    """Frame-by-frame foreground-speech detector over a single audio stream.

    Feed arbitrary-length sample chunks via :meth:`push`; it carries the model's
    streaming state across calls, returns the per-frame foreground probability, and
    maintains a hysteresis gate (:attr:`is_speaking`). Input not at the model's sample
    rate is linearly resampled (adequate for 8 kHz telephony upsampling).
    """

    def __init__(
        self,
        model: VADModel,
        input_sample_rate: int | None = None,
        on: float = 0.5,
        off: float = 0.35,
        min_speech_frames: int = 3,
        hang_frames: int = 8,
    ):
        self.model = model.eval()
        self.model_sr = model.cfg.frontend.sample_rate
        self.input_sr = input_sample_rate or self.model_sr
        self.gate = HysteresisGate(on, off, min_speech_frames, hang_frames)
        self.device = next(model.parameters()).device
        self.dtype = torch.float32
        self.reset()

    def reset(self) -> None:
        self.state = self.model.init_state(1, self.device, self.dtype)
        self.gate.reset()

    def resample(self, x: Tensor) -> Tensor:
        if self.input_sr == self.model_sr:
            return x
        n = round(x.shape[-1] * self.model_sr / self.input_sr)
        return F.interpolate(x[None, None], size=n, mode="linear", align_corners=False)[0, 0]

    @torch.no_grad()
    def push(self, samples: np.ndarray | Tensor) -> np.ndarray:
        x = torch.as_tensor(samples, dtype=self.dtype, device=self.device).reshape(-1)
        x = self.resample(x)
        logits = self.model.step(x[None], self.state)  # [1, k, C]
        probs = self.model.speech_probability(logits)[0]  # [k]
        for p in probs.tolist():
            self.gate.update(p)
        return probs.cpu().numpy()

    @property
    def is_speaking(self) -> bool:
        return self.gate.speaking

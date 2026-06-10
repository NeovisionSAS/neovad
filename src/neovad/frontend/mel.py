import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor, nn

from neovad.nn.mixer import ModuleState


class FrontendConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400  # 25 ms at 16 kHz
    hop_length: int = 160  # 10 ms at 16 kHz -> 100 frames/s (3.2x finer than Silero)
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    pcen_time_constant: float = 0.4
    pcen_eps: float = 1e-6
    pcen_gain: float = 0.98  # alpha
    pcen_bias: float = 2.0  # delta
    pcen_power: float = 0.5  # r
    pcen_trainable: bool = True
    dft_matmul: bool = False  # compute the spectrum via a real DFT matmul (ONNX-exportable)
    # instead of torch.fft.rfft (which the ONNX exporter rejects)


class FrontendState(ModuleState):
    buf: Tensor  # [B, win-hop] trailing-sample context for the next chunk
    pcen_m: Tensor | None = None  # [B, n_mels] carried PCEN smoother state


class MelFrontend(nn.Module):
    """Strictly-causal log-mel front-end with streaming PCEN.

    PCEN (per-channel energy normalization) replaces the usual log compression with a
    learnable AGC whose smoother is a first-order IIR. That AGC suppresses stationary
    background and normalizes loudness — directly the "leaked background that the
    denoiser misses" complaint from production experience — for a few multiplies per
    band, and it streams losslessly because its only state is the per-band smoother.

    ``forward`` left-pads by ``win-hop`` and frames the whole signal; ``step`` carries
    the trailing ``win-hop`` samples plus the PCEN smoother across chunks, so a single
    10 ms hop produces the same mel frame the parallel path would. ``center=False``
    means zero look-ahead — required to beat Silero's transition-decision latency.

    The mel filterbank uses the HTK formula (verified against ``torchaudio`` in tests)
    rather than a hand-typed table; PCEN constants are the librosa defaults.
    """

    def __init__(self, cfg: FrontendConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_left = cfg.win_length - cfg.hop_length
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
        fb = self.mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.fmin, cfg.fmax)
        self.register_buffer("mel_fb", fb, persistent=False)  # [n_mels, n_freqs]
        if cfg.dft_matmul:
            cos, sin = self.dft_basis(cfg.n_fft)
            self.register_buffer("dft_cos", cos, persistent=False)  # [n_fft, n_freqs]
            self.register_buffer("dft_sin", sin, persistent=False)

        t_frames = cfg.pcen_time_constant * cfg.sample_rate / cfg.hop_length
        self.b = (math.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
        self.eps = cfg.pcen_eps
        gain = torch.full((cfg.n_mels,), cfg.pcen_gain)
        bias = torch.full((cfg.n_mels,), cfg.pcen_bias)
        power = torch.full((cfg.n_mels,), cfg.pcen_power)
        if cfg.pcen_trainable:
            self.gain, self.bias, self.power = (
                nn.Parameter(gain),
                nn.Parameter(bias),
                nn.Parameter(power),
            )
        else:
            self.register_buffer("gain", gain)
            self.register_buffer("bias", bias)
            self.register_buffer("power", power)

    @property
    def n_mels(self) -> int:
        return self.cfg.n_mels

    @staticmethod
    def hz_to_mel(hz: Tensor) -> Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel: Tensor) -> Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def mel_filterbank(cls, n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> Tensor:
        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sr / 2, n_freqs)
        edges = cls.mel_to_hz(
            torch.linspace(
                cls.hz_to_mel(torch.tensor(fmin)), cls.hz_to_mel(torch.tensor(fmax)), n_mels + 2
            )
        )
        fb = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            lower, center, upper = edges[i], edges[i + 1], edges[i + 2]
            up = (freqs - lower) / (center - lower)
            down = (upper - freqs) / (upper - center)
            fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
        return fb

    @staticmethod
    def dft_basis(n_fft: int) -> tuple[Tensor, Tensor]:
        # Real DFT bases so |rfft|^2 == (x @ cos)^2 + (x @ sin)^2 — exportable to ONNX.
        n = torch.arange(n_fft)[:, None]
        k = torch.arange(n_fft // 2 + 1)[None, :]
        ang = 2 * torch.pi * n * k / n_fft
        return torch.cos(ang), -torch.sin(ang)

    def spectral_mel(self, frames: Tensor) -> Tensor:
        # frames: [B, T, win_length] -> mel power [B, T, n_mels]
        windowed = frames * self.window
        windowed = F.pad(windowed, (0, self.cfg.n_fft - self.cfg.win_length))
        if self.cfg.dft_matmul:
            power = (windowed @ self.dft_cos) ** 2 + (windowed @ self.dft_sin) ** 2
        else:
            spec = torch.fft.rfft(windowed, dim=-1)
            power = spec.real**2 + spec.imag**2
        return power @ self.mel_fb.t()

    def apply_pcen(self, mel: Tensor, m_prev: Tensor | None) -> tuple[Tensor, Tensor]:
        out = []
        m = m_prev
        for t in range(mel.shape[1]):
            e = mel[:, t]
            m = e if m is None else (1.0 - self.b) * m + self.b * e
            smooth = (self.eps + m) ** (-self.gain)
            out.append((e * smooth + self.bias) ** self.power - self.bias**self.power)
        return torch.stack(out, dim=1), m

    def frame(self, wav: Tensor) -> Tensor:
        return wav.unfold(dimension=1, size=self.cfg.win_length, step=self.cfg.hop_length)

    def forward(self, wav: Tensor) -> Tensor:
        # wav: [B, S] -> [B, n_frames, n_mels]
        padded = F.pad(wav, (self.pad_left, 0))
        mel = self.spectral_mel(self.frame(padded))
        return self.apply_pcen(mel, None)[0]

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> FrontendState:
        return FrontendState(buf=torch.zeros(batch, self.pad_left, device=device, dtype=dtype))

    def step(self, chunk: Tensor, state: FrontendState) -> Tensor:
        data = torch.cat([state.buf, chunk], dim=1)
        win, hop = self.cfg.win_length, self.cfg.hop_length
        frames = []
        while data.shape[1] >= win:
            frames.append(data[:, :win])
            data = data[:, hop:]
        state.buf = data
        if not frames:
            return torch.zeros(
                chunk.shape[0], 0, self.cfg.n_mels, device=chunk.device, dtype=chunk.dtype
            )
        mel = self.spectral_mel(torch.stack(frames, dim=1))
        pcen, state.pcen_m = self.apply_pcen(mel, state.pcen_m)
        return pcen

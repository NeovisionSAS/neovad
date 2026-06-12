import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

try:
    from livekit import rtc
    from livekit.agents import utils as lk_utils
    from livekit.agents import vad as lk_vad
except ImportError as err:  # optional extra — fail with the remedy, not a bare trace
    raise ImportError(
        "neovad.livekit requires the LiveKit Agents framework: "
        "pip install 'neovad[livekit]' (or livekit-agents>=1.0)"
    ) from err

from neovad.models.vad import VADModel

INT16_MAX = float(np.iinfo(np.int16).max)


@dataclass
class _Options:
    gate: Literal["primary", "any_speech"]
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    deactivation_threshold: float
    chunk_hops: int
    smoothing_alpha: float


class VAD(lk_vad.VAD):
    """LiveKit Agents VAD backed by neovad — a drop-in for ``livekit-plugins-silero``.

    ``AgentSession(vad=neovad.livekit.VAD.load(), ...)`` is the whole integration.
    Options mirror the Silero plugin so swapping is config-only; the one neovad-specific
    knob is ``gate``: ``"primary"`` (default) drives turn-taking from the *foreground
    speaker only* — background voices and noise do not open or hold the gate — while
    ``"any_speech"`` reproduces a classic speaker-agnostic VAD.
    """

    def __init__(self, *, model: VADModel, opts: _Options):
        sr = model.cfg.frontend.sample_rate
        hop = model.cfg.frontend.hop_length
        super().__init__(
            capabilities=lk_vad.VADCapabilities(update_interval=hop * opts.chunk_hops / sr)
        )
        self._model = model.eval()
        self._opts = opts

    @classmethod
    def load(
        cls,
        *,
        model: str | VADModel = "gru",
        gate: Literal["primary", "any_speech"] = "primary",
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        deactivation_threshold: float | None = None,
        chunk_hops: int = 3,  # 3 x 10 ms hops = 30 ms inference cadence
        force_cpu: bool = True,
    ) -> "VAD":
        if isinstance(model, str):
            model = VADModel.from_pretrained(model)
        if force_cpu:
            model = model.cpu()
        opts = _Options(
            gate=gate,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold
            if deactivation_threshold is not None
            else max(activation_threshold - 0.15, 0.01),
            chunk_hops=chunk_hops,
            smoothing_alpha=0.35,
        )
        return cls(model=model, opts=opts)

    @property
    def model(self) -> str:
        return f"neovad-{self._model.cfg.mixer.kind}"

    @property
    def provider(self) -> str:
        return "neovision"

    def stream(self) -> "VADStream":
        return VADStream(self, self._opts, self._model)


class VADStream(lk_vad.VADStream):
    """One audio stream: buffers ``rtc.AudioFrame`` s, resamples to the model rate,
    advances neovad's streaming state per chunk, and runs the Silero-compatible
    hysteresis state machine (activation/deactivation thresholds, min-speech /
    min-silence accumulation, prefix-padded utterance buffer)."""

    def __init__(self, vad: VAD, opts: _Options, model: VADModel):
        self._opts = opts
        self._model = model
        self._sr = model.cfg.frontend.sample_rate
        self._chunk = model.cfg.frontend.hop_length * opts.chunk_hops
        self._state = model.init_state(1, torch.device("cpu"), torch.float32)
        self._filter = lk_utils.ExpFilter(alpha=opts.smoothing_alpha)
        self._resampler: rtc.AudioResampler | None = None
        self._input_rate = 0
        self._pending = np.zeros(0, dtype=np.int16)  # model-rate samples awaiting a chunk
        self._speech_buf: np.ndarray | None = None  # input-rate utterance buffer
        self._buf_end = 0
        self._padding_samples = 0
        super().__init__(vad)

    def _reset_state(self) -> None:
        self._state = self._model.init_state(1, torch.device("cpu"), torch.float32)
        self._filter.reset()
        self._pending = np.zeros(0, dtype=np.int16)
        self._buf_end = 0
        if self._input_rate:
            self._resampler = self._make_resampler()

    def _make_resampler(self) -> rtc.AudioResampler | None:
        if self._input_rate == self._sr:
            return None
        return rtc.AudioResampler(
            self._input_rate, self._sr, quality=rtc.AudioResamplerQuality.QUICK
        )

    def _probability(self, chunk_i16: np.ndarray) -> float:
        wav = torch.from_numpy(chunk_i16.astype(np.float32) / INT16_MAX)[None]
        with torch.no_grad():
            logits = self._model.step(wav, self._state)
        if logits.shape[1] == 0:
            return 0.0
        if self._opts.gate == "primary":
            probs = self._model.speech_probability(logits)
        else:
            probs = self._model.any_speech_probability(logits)
        return float(probs[0].mean())

    def _buffer_write(self, samples: np.ndarray) -> None:
        free = self._speech_buf.shape[0] - self._buf_end
        n = min(free, samples.shape[0])
        if n > 0:
            self._speech_buf[self._buf_end : self._buf_end + n] = samples[:n]
            self._buf_end += n

    def _buffer_keep_padding(self) -> None:
        if self._buf_end > self._padding_samples:
            self._speech_buf[: self._padding_samples] = self._speech_buf[
                self._buf_end - self._padding_samples : self._buf_end
            ]
            self._buf_end = self._padding_samples

    def _buffer_frame(self) -> rtc.AudioFrame:
        data = self._speech_buf[: self._buf_end]
        return rtc.AudioFrame(
            data=data.tobytes(),
            sample_rate=self._input_rate,
            num_channels=1,
            samples_per_channel=int(data.shape[0]),
        )

    async def _main_task(self) -> None:
        loop = asyncio.get_event_loop()
        speaking = False
        speech_acc = silence_acc = 0.0
        pub_speech = pub_silence = pub_timestamp = 0.0
        pub_samples = 0
        chunk_s = self._chunk / self._sr

        async for item in self._input_ch:
            if isinstance(item, lk_vad.VADStream._FlushSentinel):
                self._reset_state()
                speaking = False
                speech_acc = silence_acc = 0.0
                continue
            if self._input_rate == 0:
                self._input_rate = item.sample_rate
                self._padding_samples = int(self._opts.prefix_padding_duration * self._input_rate)
                size = int(self._opts.max_buffered_speech * self._input_rate)
                self._speech_buf = np.zeros(size + self._padding_samples, dtype=np.int16)
                self._resampler = self._make_resampler()
            elif item.sample_rate != self._input_rate:
                continue  # mixed-rate input is a caller bug; mirror silero (skip frame)

            in_samples = np.frombuffer(item.data, dtype=np.int16)
            self._buffer_write(in_samples)
            if self._resampler is not None:
                resampled = [
                    np.frombuffer(f.data, dtype=np.int16) for f in self._resampler.push(item)
                ]
                if resampled:
                    self._pending = np.concatenate([self._pending, *resampled])
            else:
                self._pending = np.concatenate([self._pending, in_samples])

            while self._pending.shape[0] >= self._chunk:
                chunk, self._pending = self._pending[: self._chunk], self._pending[self._chunk :]
                t0 = time.perf_counter()
                p = self._filter.apply(
                    exp=1.0, sample=await loop.run_in_executor(None, self._probability, chunk)
                )
                inference_s = time.perf_counter() - t0
                pub_samples += self._chunk
                pub_timestamp += chunk_s

                active = p >= self._opts.activation_threshold or (
                    speaking and p > self._opts.deactivation_threshold
                )
                if active:
                    speech_acc += chunk_s
                    silence_acc = 0.0
                    pub_speech = pub_speech + chunk_s if speaking else pub_speech
                    if not speaking and speech_acc >= self._opts.min_speech_duration:
                        speaking = True
                        pub_speech, pub_silence = speech_acc, 0.0
                        self._emit(
                            lk_vad.VADEventType.START_OF_SPEECH,
                            pub_samples,
                            pub_timestamp,
                            pub_speech,
                            pub_silence,
                            [self._buffer_frame()],
                            p,
                            inference_s,
                            True,
                            speech_acc,
                            silence_acc,
                        )
                else:
                    silence_acc += chunk_s
                    speech_acc = 0.0
                    pub_silence = pub_silence + chunk_s if not speaking else pub_silence
                    if not speaking:
                        self._buffer_keep_padding()
                    if speaking and silence_acc >= self._opts.min_silence_duration:
                        speaking = False
                        pub_silence = silence_acc
                        self._emit(
                            lk_vad.VADEventType.END_OF_SPEECH,
                            pub_samples,
                            pub_timestamp,
                            max(0.0, pub_speech - silence_acc),
                            pub_silence,
                            [self._buffer_frame()],
                            p,
                            inference_s,
                            False,
                            speech_acc,
                            silence_acc,
                        )
                        pub_speech = 0.0
                        self._buffer_keep_padding()
                self._emit(
                    lk_vad.VADEventType.INFERENCE_DONE,
                    pub_samples,
                    pub_timestamp,
                    pub_speech,
                    pub_silence,
                    [],
                    p,
                    inference_s,
                    speaking,
                    speech_acc,
                    silence_acc,
                )

    def _emit(
        self,
        type_,
        samples,
        timestamp,
        speech_s,
        silence_s,
        frames,
        p,
        infer_s,
        speaking,
        raw_speech,
        raw_silence,
    ) -> None:
        self._event_ch.send_nowait(
            lk_vad.VADEvent(
                type=type_,
                samples_index=samples,
                timestamp=timestamp,
                speech_duration=speech_s,
                silence_duration=silence_s,
                frames=frames,
                probability=p,
                inference_duration=infer_s,
                speaking=speaking,
                raw_accumulated_speech=raw_speech,
                raw_accumulated_silence=raw_silence,
            )
        )

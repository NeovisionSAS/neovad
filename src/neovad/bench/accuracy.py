import io

import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel

from neovad.config import DataConfig
from neovad.data.sources import Datasets
from neovad.data.synth import MixtureSynthesizer
from neovad.frontend.mel import FrontendConfig
from neovad.models.vad import VADModel


class AccuracyResult(BaseModel):
    name: str
    roc_auc: float  # threshold-free, the field-standard VAD metric
    f1: float  # best-threshold frame F1
    threshold: float  # threshold achieving that F1
    frames: int
    # decision latency at that same threshold: ms from a true speech boundary to the
    # first frame whose thresholded score reflects it (median / p90 over boundaries)
    onset_ms: float = float("nan")
    onset_p90_ms: float = float("nan")
    offset_ms: float = float("nan")
    offset_p90_ms: float = float("nan")


class AccuracyBenchmark:
    """Apples-to-apples speech/non-speech accuracy: neovad and a reference VAD scored on
    the *same* audio against the *same* per-frame ground truth, on neovad's 10 ms grid.

    neovad is scored by its any-speech probability (1 - P(non-speech)) so it is judged on
    the generic task a speech/non-speech VAD does, not its foreground advantage. The
    headline metric is ROC-AUC (threshold-free, so neither model is favoured by threshold
    tuning); frame-F1 at the best threshold is reported alongside.

    A clip is ``(waveform_16k [S], speech_label_per_10ms_frame [T])``.
    """

    SILERO_SR = 16000
    SILERO_WINDOW = 512  # samples (~32 ms) — Silero's fixed chunk

    def __init__(self, hop_ms: float = 10.0, sample_rate: int = 16000):
        self.hop = int(sample_rate * hop_ms / 1000)
        self.sr = sample_rate

    @staticmethod
    def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        # Mann-Whitney U form: P(score(pos) > score(neg)), ties counted as 0.5.
        pos = labels == 1
        n_pos, n_neg = int(pos.sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # tie-aware average ranks (1-based), then the Mann-Whitney U identity
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        avg_rank = (ends - counts + 1 + ends) / 2  # mean of the rank span of each value
        ranks = avg_rank[inv]
        return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    @staticmethod
    def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = scores >= t
            tp = float((pred & (labels == 1)).sum())
            fp = float((pred & (labels == 0)).sum())
            fn = float((~pred & (labels == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_f1, best_t

    @torch.no_grad()
    def neovad_scores(self, model: VADModel, wav: torch.Tensor) -> np.ndarray:
        prob = model.any_speech_probability(model(wav[None]))[0]
        return prob.float().cpu().numpy()  # [T] at 10 ms

    @torch.no_grad()
    def silero_scores(self, silero, wav: torch.Tensor, n_frames: int) -> np.ndarray:
        # one prob per 512-sample window, then mapped onto the 10 ms frame grid
        win = self.SILERO_WINDOW
        probs = []
        for i in range(0, wav.shape[0] - win + 1, win):
            probs.append(float(silero(wav[i : i + win], self.SILERO_SR)))
        probs = np.array(probs) if probs else np.zeros(1)
        idx = np.minimum((np.arange(n_frames) * self.hop) // win, len(probs) - 1)
        return probs[idx]

    @staticmethod
    def transition_delays(
        scores: np.ndarray, labels: np.ndarray, threshold: float, max_lag: int = 100
    ) -> tuple[list[int], list[int]]:
        """Per-boundary detection lag, in frames, for one clip.

        Onset: first frame at/after a true speech start where the thresholded score is
        ON. Offset: first frame at/after a true speech end where it is OFF. Boundaries
        not reflected within ``max_lag`` frames count as missed (excluded; rate is
        visible through F1). The metric Silero's published AUCs hide — turn-taking
        latency in a live agent is exactly this lag.
        """
        lab = labels.astype(bool)
        pred = scores >= threshold
        prev = np.roll(lab, 1)
        prev[0] = lab[0]
        onsets, offsets = [], []
        for idx in np.flatnonzero(lab & ~prev):
            window = pred[idx : idx + max_lag]
            hits = np.flatnonzero(window)
            if hits.size:
                onsets.append(int(hits[0]))
        for idx in np.flatnonzero(~lab & prev):
            window = pred[idx : idx + max_lag]
            misses = np.flatnonzero(~window)
            if misses.size:
                offsets.append(int(misses[0]))
        return onsets, offsets

    def evaluate(self, model: VADModel, clips: list, silero=None) -> list[AccuracyResult]:
        model = model.eval()
        per_model: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"neovad": []}
        if silero is not None:
            per_model["silero-v6"] = []
        for wav, labels in clips:
            y = labels.cpu().numpy()
            ns = self.neovad_scores(model, wav)
            n = min(len(ns), len(y))
            per_model["neovad"].append((ns[:n], y[:n]))
            if silero is not None:
                per_model["silero-v6"].append((self.silero_scores(silero, wav, n), y[:n]))
        return [self._score(name, pairs) for name, pairs in per_model.items()]

    def _score(self, name: str, pairs: list[tuple[np.ndarray, np.ndarray]]) -> AccuracyResult:
        scores = np.concatenate([s for s, _ in pairs])
        labels = np.concatenate([y for _, y in pairs])
        f1, t = self.best_f1(scores, labels)
        onsets, offsets = [], []
        for s, y in pairs:  # boundaries never span clips
            on, off = self.transition_delays(s, y, t)
            onsets += on
            offsets += off
        hop_ms = self.hop / self.sr * 1000
        onset = np.array(onsets, dtype=float) * hop_ms
        offset = np.array(offsets, dtype=float) * hop_ms
        return AccuracyResult(
            name=name,
            roc_auc=self.roc_auc(scores, labels),
            f1=f1,
            threshold=t,
            frames=len(labels),
            onset_ms=float(np.median(onset)) if onset.size else float("nan"),
            onset_p90_ms=float(np.percentile(onset, 90)) if onset.size else float("nan"),
            offset_ms=float(np.median(offset)) if offset.size else float("nan"),
            offset_p90_ms=float(np.percentile(offset, 90)) if offset.size else float("nan"),
        )

    @staticmethod
    def synthetic_clips(root: str, n: int, seconds: float = 6.0, seed: int = 900000) -> list:
        # Held-out clips from the SAME synthesis pipeline -> in-distribution for neovad.
        fe = FrontendConfig()
        sp = Datasets.files("speech", root)
        ns = Datasets.files("noise", root) + Datasets.files("music", root)
        rir = Datasets.files("rir", root)
        synth = MixtureSynthesizer(DataConfig(clip_seconds=seconds), fe, sp, ns, rir, seed=seed)
        clips = []
        for i in range(n):
            synth.reseed(seed + i)
            mix = synth.mix_components()
            clips.append((mix.mix, (mix.labels > 0).long()))  # speech = primary OR secondary
        return clips

    @staticmethod
    def diarized_clips(
        repo: str,
        config: str | None,
        split: str,
        n: int,
        window_s: int = 30,
        sr: int = 16000,
        hop: int = 160,
    ) -> list:
        """Stream a HuggingFace diarization dataset (audio + timestamps_start/end) and
        slice each recording into ``window_s`` windows with per-``hop`` speech labels
        (speech = union of any speaker segment). Resamples to ``sr``. The shared
        extractor for both the neutral eval sets and the real-audio training corpus."""
        from datasets import Audio, load_dataset  # optional (the `train` extra)

        ds = load_dataset(repo, config, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        win = window_s * sr
        clips = []
        for ex in ds:
            raw = ex["audio"]["bytes"] or open(ex["audio"]["path"], "rb").read()
            data, src_sr = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(1)
            if src_sr != sr:
                m = round(len(data) * sr / src_sr)
                data = np.interp(np.linspace(0, len(data) - 1, m), np.arange(len(data)), data)
                data = data.astype(np.float32)
            for i in range(0, len(data) - win + 1, win):
                nf = win // hop
                y = np.zeros(nf, dtype=np.int64)
                for s, e in zip(ex["timestamps_start"], ex["timestamps_end"], strict=False):
                    a, b = int((s - i / sr) * sr / hop), int(np.ceil((e - i / sr) * sr / hop))
                    y[max(0, a) : min(nf, b)] = 1
                clips.append((torch.from_numpy(data[i : i + win].copy()), torch.from_numpy(y)))
                if len(clips) >= n:
                    return clips
        return clips

    @classmethod
    def voxconverse_clips(cls, n: int, split: str = "test", **kw) -> list:
        # split="test": the neutral eval benchmark — NEVER trained on.
        return cls.diarized_clips("diarizers-community/voxconverse", None, split, n, **kw)

    @staticmethod
    def report(results: list[AccuracyResult]) -> None:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="speech/non-speech accuracy (same audio, same labels)")
        cols = [
            "model",
            "ROC-AUC",
            "frame-F1",
            "@thr",
            "onset ms (p90)",
            "offset ms (p90)",
            "frames",
        ]
        for col in cols:
            table.add_column(col, justify="right")
        for r in results:
            table.add_row(
                r.name,
                f"{r.roc_auc:.4f}",
                f"{r.f1:.4f}",
                f"{r.threshold:.2f}",
                f"{r.onset_ms:.0f} ({r.onset_p90_ms:.0f})",
                f"{r.offset_ms:.0f} ({r.offset_p90_ms:.0f})",
                str(r.frames),
            )
        Console().print(table)

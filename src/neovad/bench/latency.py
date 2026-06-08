import io
from time import perf_counter

import torch
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from neovad.models.vad import VADModel


class LatencyResult(BaseModel):
    name: str
    params_m: float
    size_mb: float
    chunk_ms: float  # wall-clock per chunk this model consumes
    chunk_audio_ms: float  # audio duration that chunk represents
    rtf: float  # compute_time / audio_time — the fair cross-model metric


class LatencyBenchmark:
    """Streaming-latency harness. The headline comparison the project is judged on:
    real-time factor (RTF), per-chunk wall-time, model size on disk — neovad backbones
    against Silero VAD on the same single CPU thread.

    Developer one-shot utility (never a user-facing surface): RTF measured on random
    audio is input-independent, so no dataset is needed for the latency numbers; the
    accuracy comparison uses the labelled eval sets (see docs/ARCHITECTURE.md).
    """

    def __init__(
        self, seconds: float = 30.0, threads: int = 1, device: str = "cpu", warmup: int = 30
    ):
        self.seconds = seconds
        self.threads = threads
        self.device = torch.device(device)
        self.warmup = warmup

    @staticmethod
    def size_mb(state_dict) -> float:
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return len(buf.getvalue()) / 1e6

    def time_stream(self, fn, n: int) -> float:
        for _ in range(self.warmup):
            fn()
        t0 = perf_counter()
        for _ in range(n):
            fn()
        return perf_counter() - t0

    @torch.no_grad()
    def bench_neovad(self, model: VADModel, name: str = "neovad") -> LatencyResult:
        torch.set_num_threads(self.threads)
        model = model.eval().to(self.device)
        hop = model.cfg.frontend.hop_length
        sr = model.cfg.frontend.sample_rate
        n = int(self.seconds * sr / hop)
        chunk = torch.randn(1, hop, device=self.device)
        state = model.init_state(1, self.device, torch.float32)
        dt = self.time_stream(lambda: model.step(chunk, state), n)
        return LatencyResult(
            name=name,
            params_m=model.param_count / 1e6,
            size_mb=self.size_mb(model.state_dict()),
            chunk_ms=dt / n * 1000,
            chunk_audio_ms=hop / sr * 1000,
            rtf=dt / self.seconds,
        )

    @torch.no_grad()
    def bench_silero(self) -> LatencyResult:
        from silero_vad import load_silero_vad  # optional dependency (neovad[bench])

        torch.set_num_threads(self.threads)
        model = load_silero_vad()
        sr, frame = 16000, 512  # silero 16 kHz fixed chunk
        n = int(self.seconds * sr / frame)
        chunk = torch.randn(frame)
        dt = self.time_stream(lambda: model(chunk, sr), n)
        params = sum(p.numel() for p in model.parameters())
        return LatencyResult(
            name="silero-v6",
            params_m=params / 1e6,
            size_mb=self.size_mb(model.state_dict()),
            chunk_ms=dt / n * 1000,
            chunk_audio_ms=frame / sr * 1000,
            rtf=dt / self.seconds,
        )

    def report(self, results: list[LatencyResult]) -> None:
        table = Table(
            title=f"streaming latency ({self.threads} CPU thread, {self.seconds:.0f}s audio)"
        )
        for col in [
            "model",
            "params (M)",
            "size (MB)",
            "chunk (ms)",
            "per chunk audio (ms)",
            "RTF",
        ]:
            table.add_column(col, justify="right")
        for r in results:
            table.add_row(
                r.name,
                f"{r.params_m:.2f}",
                f"{r.size_mb:.2f}",
                f"{r.chunk_ms:.3f}",
                f"{r.chunk_audio_ms:.1f}",
                f"{r.rtf:.5f}",
            )
        Console().print(table)

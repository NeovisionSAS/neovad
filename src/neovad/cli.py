from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import typer
from rich.console import Console

from neovad.bench.latency import LatencyBenchmark
from neovad.data.sources import Datasets
from neovad.export import ModelExporter
from neovad.infer.stream import HysteresisGate
from neovad.models.vad import VADModel
from neovad.nn.head import VADHead
from neovad.nn.mixer import StreamingMixer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="neovad — small streaming VAD")
console = Console()


def resolve_model(checkpoint: Path | None, config: Path | None, backbone: str | None) -> VADModel:
    if checkpoint is not None:
        return VADModel.load(checkpoint)
    if config is not None:
        return VADModel.from_config(config)
    return VADModel.from_backbone(backbone or "mamba2")


@app.command("list-backbones")
def list_backbones():
    """List the pluggable sequence-mixer backbones and the available heads."""
    console.print(f"[bold]backbones[/]: {', '.join(StreamingMixer.names())}")
    console.print(f"[bold]heads[/]: {', '.join(VADHead.names())}")


@app.command()
def download(
    root: str = typer.Option("/disk/manual", help="where datasets are stored"),
    only: list[str] = typer.Option(None, help="restrict to these dataset names"),
):
    """Download training/eval datasets into ROOT (LibriSpeech, MUSAN, RIRs)."""
    for name in only or Datasets.names():
        spec = Datasets.spec(name)
        console.print(f"[bold]{name}[/] (~{spec.approx_gb} GB, {spec.license})")
        Datasets.download(name, root)


@app.command()
def train(config: Path = typer.Argument(..., help="path to a NeoVADConfig YAML")):
    """Train a model from a YAML config."""
    from neovad.config import NeoVADConfig  # local: keeps lightning out of light imports
    from neovad.train.lit import NeoVADLit

    NeoVADLit.run(NeoVADConfig.load(config))


@app.command()
def bench(
    checkpoint: Path = typer.Option(None),
    config: Path = typer.Option(None),
    backbone: str = typer.Option(None, help="bench a default model with this backbone"),
    silero: bool = typer.Option(True, help="also benchmark Silero VAD if installed"),
    seconds: float = typer.Option(30.0),
    threads: int = typer.Option(1),
    all_backbones: bool = typer.Option(False, help="bench every neovad backbone"),
):
    """Measure streaming latency / size / RTF and compare against Silero VAD."""
    harness = LatencyBenchmark(seconds=seconds, threads=threads)
    results = []
    if all_backbones:
        results = [
            harness.bench_neovad(VADModel.from_backbone(n), name=n) for n in StreamingMixer.names()
        ]
    else:
        model = resolve_model(checkpoint, config, backbone)
        results.append(harness.bench_neovad(model, name=backbone or "neovad"))
    if silero:
        try:
            results.append(harness.bench_silero())
        except (ImportError, ModuleNotFoundError):
            console.print(
                "[yellow]silero-vad not installed; skipping (pip install neovad[bench])[/]"
            )
    harness.report(results)


@app.command()
def infer(
    audio: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(None),
    config: Path = typer.Option(None),
    backbone: str = typer.Option(None),
    on: float = typer.Option(0.5),
    off: float = typer.Option(0.35),
):
    """Run a (trained) model over an audio file and print foreground-speech segments."""
    model = resolve_model(checkpoint, config, backbone).eval()
    sr = model.cfg.frontend.sample_rate
    data, file_sr = sf.read(str(audio), dtype="float32", always_2d=False)
    wav = torch.as_tensor(data).float()
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if file_sr != sr:
        wav = F.interpolate(
            wav[None, None],
            size=round(wav.shape[-1] * sr / file_sr),
            mode="linear",
            align_corners=False,
        )[0, 0]
    with torch.no_grad():
        probs = model.speech_probability(model(wav[None]))[0]
    gate = HysteresisGate(on, off, min_speech_frames=3, hang_frames=8)
    frame_sec = model.cfg.frontend.hop_length / sr
    start = None
    for i, p in enumerate(probs.tolist()):
        was = gate.speaking
        now = gate.update(p)
        if now and not was:
            start = i * frame_sec
        elif was and not now and start is not None:
            console.print(f"speech {start:7.2f}s -> {i * frame_sec:7.2f}s")
            start = None
    if start is not None:
        console.print(f"speech {start:7.2f}s -> {len(probs) * frame_sec:7.2f}s")


@app.command()
def export(
    out: Path = typer.Argument(..., help="output path"),
    checkpoint: Path = typer.Option(None),
    config: Path = typer.Option(None),
    backbone: str = typer.Option(None),
    fmt: str = typer.Option("onnx", help="onnx | jit | int8"),
    quantize: bool = typer.Option(
        False, help="int8-quantize the ONNX graph (not advised for tiny RNNs)"
    ),
    seconds: float = typer.Option(2.0, help="fixed clip length for jit/onnx tracing"),
):
    """Compress/export a trained model for CPU deployment (torch int8 / TorchScript / ONNX)."""
    model = resolve_model(checkpoint, config, backbone).eval()
    before = ModelExporter.size_mb(model)
    if fmt == "int8":
        torch.save(ModelExporter.quantize_dynamic(model), out)
    elif fmt == "jit":
        ModelExporter.jit_trace(model, seconds).save(str(out))
    elif fmt == "onnx":
        out = ModelExporter.onnx(model, out, seconds, quantize)
    else:
        raise typer.BadParameter(f"unknown format {fmt!r}; use onnx | jit | int8")
    console.print(
        f"[bold]{fmt}[/] {out} — {ModelExporter.size_mb(out):.2f} MB (fp32 was {before:.2f} MB)"
    )


if __name__ == "__main__":
    app()

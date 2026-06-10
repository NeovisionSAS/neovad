import io
from pathlib import Path

import torch
from torch import nn

from neovad.models.vad import VADModel


class ModelExporter:
    """Post-training compression / export for CPU deployment.

    The research-grounded reality (see docs/ARCHITECTURE.md): Silero ships **fp32**
    JIT+ONNX — its speed is operator fusion, not int8 — and ``torch.fft.rfft`` is not
    ONNX-exportable, so ONNX needs the DFT-matmul frontend. Three paths:

    * ``quantize_dynamic`` — int8 on Linear/GRU, ~4x smaller on those weights, keeps the
      exact streaming ``step`` API in pure PyTorch. Simplest; re-check accuracy after.
    * ``jit_trace`` — TorchScript fused graph for offline forward (Silero's fp32 recipe).
    * ``onnx`` — an ONNX Runtime artifact via the DFT-matmul frontend, optional int8.
      Fixed time length (the PCEN IIR unrolls when traced); batch axis stays dynamic.
    """

    QUANTIZABLE: set[type[nn.Module]] = {nn.Linear, nn.GRU}

    @staticmethod
    def size_mb(obj: nn.Module | bytes | Path) -> float:
        if isinstance(obj, str | Path):
            return Path(obj).stat().st_size / 1e6
        if isinstance(obj, bytes | bytearray):
            return len(obj) / 1e6
        buf = io.BytesIO()
        torch.save(obj.state_dict(), buf)
        return len(buf.getvalue()) / 1e6

    @classmethod
    def quantize_dynamic(cls, model: VADModel) -> nn.Module:
        return torch.ao.quantization.quantize_dynamic(
            model.eval().cpu(), cls.QUANTIZABLE, dtype=torch.qint8
        )

    @staticmethod
    def to_exportable(model: VADModel) -> VADModel:
        # Rebuild with the ONNX-friendly DFT-matmul frontend, sharing all trained weights
        # (the DFT/mel/window buffers are non-persistent, so the state dicts line up).
        cfg = model.cfg.model_copy(deep=True)
        cfg.frontend = cfg.frontend.model_copy(update={"dft_matmul": True})
        export_model = VADModel(cfg)
        export_model.load_state_dict(model.state_dict(), strict=False)
        return export_model.eval()

    @classmethod
    def jit_trace(cls, model: VADModel, seconds: float = 2.0) -> torch.jit.ScriptModule:
        model = model.eval().cpu()
        n = int(seconds * model.cfg.frontend.sample_rate)
        with torch.no_grad():
            return torch.jit.trace(model, torch.randn(1, n))

    @classmethod
    def onnx(
        cls, model: VADModel, path: str | Path, seconds: float = 2.0, quantize: bool = False
    ) -> Path:
        export_model = cls.to_exportable(model)
        n = int(seconds * model.cfg.frontend.sample_rate)
        path = Path(path)
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                torch.randn(1, n),
                str(path),
                input_names=["waveform"],
                output_names=["logits"],
                dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=18,
            )
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            qpath = path.with_suffix(".int8.onnx")
            quantize_dynamic(str(path), str(qpath), weight_type=QuantType.QInt8)
            return qpath
        return path

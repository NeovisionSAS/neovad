import io

import pytest
import torch

from neovad.config import ModelConfig
from neovad.export import ModelExporter
from neovad.models.vad import VADModel


def serialized_size(module) -> int:
    buf = io.BytesIO()
    torch.save(module, buf)
    return len(buf.getvalue())


def test_dft_frontend_matches_rfft():
    # The ONNX-friendly DFT-matmul spectrum must equal the rfft spectrum.
    torch.manual_seed(0)
    model = VADModel(ModelConfig(dim=32, depth=2, mixer={"kind": "gqa"})).eval()
    exportable = ModelExporter.to_exportable(model)
    wav = torch.randn(1, 8000)
    with torch.no_grad():
        assert torch.allclose(model(wav), exportable(wav), atol=1e-3)


def test_jit_trace_matches_eager():
    torch.manual_seed(0)
    model = VADModel(ModelConfig(dim=32, depth=2, mixer={"kind": "mamba2"})).eval()
    wav = torch.randn(1, 8000)
    traced = ModelExporter.jit_trace(model, seconds=0.5)
    with torch.no_grad():
        assert torch.allclose(model(wav), traced(wav), atol=1e-4)


def test_dynamic_int8_runs_and_shrinks():
    model = VADModel(ModelConfig(dim=128, depth=2, mixer={"kind": "gru"})).eval()
    quant = ModelExporter.quantize_dynamic(model)
    with torch.no_grad():
        out = quant(torch.randn(1, 8000))
    assert out.shape[-1] == model.head.n_classes
    assert serialized_size(quant) < serialized_size(model.state_dict())


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import numpy as np
    import onnxruntime as ort

    # mamba2 is Linear-only (no nn.GRU) so it exports to ONNX faithfully.
    torch.manual_seed(0)
    model = VADModel(ModelConfig(dim=32, depth=2, mixer={"kind": "mamba2"})).eval()
    wav = torch.randn(1, 8000)
    with torch.no_grad():
        ref = model(wav).numpy()
    path = ModelExporter.onnx(model, tmp_path / "m.onnx", seconds=0.5)
    out = ort.InferenceSession(str(path)).run(None, {"waveform": wav.numpy()})[0]
    assert np.abs(ref - out).max() < 1e-2

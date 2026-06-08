import numpy as np
import torch

from neovad.infer.stream import StreamingVAD
from neovad.models.vad import VADModel


def test_save_load_roundtrip(tmp_path, make_model):
    model = make_model("gqa").eval()
    wav = torch.randn(1, 4000)
    with torch.no_grad():
        before = model(wav)
    path = tmp_path / "model.pt"
    model.save(path)
    reloaded = VADModel.load(path).eval()
    with torch.no_grad():
        after = reloaded(wav)
    assert torch.allclose(before, after, atol=1e-6)
    assert reloaded.cfg == model.cfg


def test_from_backbone(backbone):
    model = VADModel.from_backbone(backbone)
    assert model.cfg.mixer.kind == backbone
    assert model.param_count < 5_000_000  # under budget


def test_speech_probability_range(make_model):
    model = make_model("mamba2").eval()
    with torch.no_grad():
        prob = model.speech_probability(model(torch.randn(2, 8000)))
    assert prob.shape == (2, 8000 // model.cfg.frontend.hop_length)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_streaming_vad_gate(make_model):
    vad = StreamingVAD(make_model("gru").eval())
    probs = vad.push(np.zeros(1600, dtype=np.float32))
    assert probs.shape[0] == 1600 // 160
    assert isinstance(vad.is_speaking, bool)
    vad.reset()
    assert not vad.is_speaking


def test_streaming_vad_resamples_8k(make_model):
    # 8 kHz telephony input must be upsampled to the model's 16 kHz.
    vad = StreamingVAD(make_model("gru").eval(), input_sample_rate=8000)
    probs = vad.push(np.zeros(800, dtype=np.float32))  # 100 ms @ 8 kHz -> 10 frames
    assert probs.shape[0] == 10

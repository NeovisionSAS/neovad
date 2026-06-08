import numpy as np
import pytest
import soundfile as sf

from neovad.config import ModelConfig
from neovad.models.vad import VADModel
from neovad.nn.mixer import StreamingMixer


@pytest.fixture(params=StreamingMixer.names())
def backbone(request) -> str:
    return request.param


@pytest.fixture
def make_model():
    def build(backbone: str, dim: int = 32, depth: int = 2, **kw) -> VADModel:
        return VADModel(ModelConfig(dim=dim, depth=depth, mixer={"kind": backbone}, **kw))

    return build


@pytest.fixture
def audio_pool(tmp_path):
    """Tiny on-disk speech/noise pools (gated tones + white noise) for synth tests."""
    speech, noise = [], []
    for i in range(6):
        t = np.linspace(0, 3, 48000, dtype=np.float32)
        gated = 0.3 * np.sin(2 * np.pi * (150 + 40 * i) * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
        sp = tmp_path / f"sp{i}.wav"
        ns = tmp_path / f"noise{i}.wav"
        sf.write(sp, gated, 16000)
        sf.write(
            ns, 0.1 * np.random.default_rng(i).standard_normal(48000).astype(np.float32), 16000
        )
        speech.append(sp)
        noise.append(ns)
    return speech, noise

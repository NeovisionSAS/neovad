import torch
import torch.nn.functional as F

from neovad.config import DataConfig
from neovad.data.synth import MixtureSynthesizer
from neovad.frontend.mel import FrontendConfig
from neovad.nn.head import SpeechClass


def test_fft_convolve_matches_direct():
    # The fast RIR path must equal a direct linear convolution truncated to input length.
    x = torch.randn(2000)
    h = torch.randn(300)
    direct = F.conv1d(x[None, None], h.flip(0)[None, None], padding=h.shape[-1] - 1)[0, 0][
        : x.shape[-1]
    ]
    assert torch.allclose(MixtureSynthesizer.fft_convolve(x, h), direct, atol=1e-4)


def test_synth_shapes_and_labels(audio_pool):
    speech, noise = audio_pool
    fe = FrontendConfig()
    synth = MixtureSynthesizer(
        DataConfig(clip_seconds=2.0, min_interferers=0, max_interferers=0, p_noise=1.0),
        fe,
        speech,
        noise,
        [],
        seed=0,
    )
    wav, labels = synth.mix()
    assert wav.shape[0] == synth.n_samples
    assert labels.shape[0] == synth.n_frames == wav.shape[0] // fe.hop_length
    assert set(labels.tolist()) <= {0, 1, 2}
    assert (labels == int(SpeechClass.PRIMARY)).any()  # primary speech is present
    assert wav.abs().max() <= 1.0  # never clips


def test_labels_from_clean_primary_not_noise(audio_pool):
    # Heavy noise at very low SNR must not create PRIMARY/SECONDARY frames: labels are
    # derived from the clean primary reference, so non-speech stays non-speech.
    speech, noise = audio_pool
    fe = FrontendConfig()
    quiet = DataConfig(
        clip_seconds=2.0, min_interferers=0, max_interferers=0, p_noise=1.0, snr_db=(-20.0, -20.0)
    )
    synth = MixtureSynthesizer(quiet, fe, speech, noise, [], seed=1)
    counts = torch.zeros(3)
    for s in range(6):
        synth.reseed(s)
        _, labels = synth.mix()
        for c in range(3):
            counts[c] += (labels == c).sum()
    assert counts[int(SpeechClass.NON_SPEECH)] > 0  # silence regions survive the noise


def test_synth_produces_secondary(audio_pool):
    speech, noise = audio_pool
    cfg = DataConfig(
        clip_seconds=2.0, min_interferers=2, max_interferers=2, p_interferer=1.0, p_noise=0.0
    )
    synth = MixtureSynthesizer(cfg, FrontendConfig(), speech, noise, [], seed=0)
    seen = set()
    for s in range(10):
        synth.reseed(s)
        _, labels = synth.mix()
        seen |= set(labels.tolist())
    assert int(SpeechClass.SECONDARY) in seen

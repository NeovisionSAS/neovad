import pytest
import torch

from neovad.frontend.mel import FrontendConfig, MelFrontend
from neovad.nn.attention import DiffAttnConfig
from neovad.nn.conv import CausalDepthwiseConv1d
from neovad.nn.norm import RMSNorm
from neovad.nn.rope import RotaryEmbedding


def test_rmsnorm_normalizes():
    norm = RMSNorm(16)
    x = torch.randn(4, 7, 16) * 5.0
    rms = norm(x).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_preserves_norm_and_is_relative():
    rope = RotaryEmbedding(8)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    pos = torch.arange(5)
    # rotation is orthogonal -> preserves vector norm
    assert torch.allclose(rope.apply_rotary(q, pos).norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    # relative: shifting both q and k positions by a constant leaves the score matrix unchanged
    s0 = rope.apply_rotary(q, pos) @ rope.apply_rotary(k, pos).transpose(-1, -2)
    s1 = rope.apply_rotary(q, pos + 3) @ rope.apply_rotary(k, pos + 3).transpose(-1, -2)
    assert torch.allclose(s0, s1, atol=1e-4)


def test_causal_conv_is_causal_and_streams():
    conv = CausalDepthwiseConv1d(4, kernel=3).eval()
    x = torch.randn(2, 10, 4)
    with torch.no_grad():
        y = conv(x)
        # changing a future frame must not change earlier outputs (causality)
        x2 = x.clone()
        x2[:, 7:] += 5.0
        y2 = conv(x2)
        assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)
        # step path equals forward
        state = conv.init_state(2, x.device, x.dtype)
        stepped = torch.cat([conv.step(x[:, i : i + 1], state) for i in range(10)], dim=1)
    assert torch.allclose(y, stepped, atol=1e-5)


def test_diffattn_lambda_init_depth_dependent():
    shallow = DiffAttnConfig().build(32, depth=1)
    deep = DiffAttnConfig().build(32, depth=8)
    # paper schedule lambda_init = 0.8 - 0.6*exp(-0.3*(l-1)): rises 0.2 -> ~0.8 with depth
    assert shallow.lambda_init < deep.lambda_init
    assert abs(shallow.lambda_init - 0.2) < 1e-6


def test_mel_filterbank_matches_torchaudio():
    ta = pytest.importorskip("torchaudio")
    cfg = FrontendConfig()
    fb = MelFrontend.mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.fmin, cfg.fmax)
    ref = ta.functional.melscale_fbanks(
        cfg.n_fft // 2 + 1,
        cfg.fmin,
        cfg.fmax,
        cfg.n_mels,
        cfg.sample_rate,
        norm=None,
        mel_scale="htk",
    ).t()
    assert torch.allclose(fb, ref, atol=1e-4)

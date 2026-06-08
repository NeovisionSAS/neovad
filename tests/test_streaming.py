import torch

from neovad.frontend.mel import FrontendConfig, MelFrontend
from neovad.nn.attention import DiffAttnConfig, GQAConfig, MLAConfig
from neovad.nn.gru import GRUConfig
from neovad.nn.mamba import Mamba2Config

# Small windows on purpose, so the streaming KV ring-buffer / state-truncation path is
# exercised (window < sequence length), not just the trivial full-context case.
MIXER_CFGS = {
    "gru": GRUConfig(hidden=32),
    "gqa": GQAConfig(n_heads=4, n_kv_heads=2, window=8),
    "mla": MLAConfig(n_heads=4, window=8),
    "diffattn": DiffAttnConfig(n_heads=2, window=8),
    "mamba2": Mamba2Config(headdim=16, d_state=16),
}


def test_mixer_forward_step_equivalence(backbone):
    """The library's central invariant: stepping equals the parallel forward."""
    torch.manual_seed(0)
    dim, t, b = 32, 20, 2
    mixer = MIXER_CFGS[backbone].build(dim, depth=2).eval()
    x = torch.randn(b, t, dim)
    with torch.no_grad():
        full = mixer(x)
        state = mixer.init_state(b, x.device, x.dtype)
        stepped = torch.cat([mixer.step(x[:, i : i + 1], state) for i in range(t)], dim=1)
    assert torch.allclose(full, stepped, atol=1e-4)


def test_model_forward_step_equivalence(backbone, make_model):
    torch.manual_seed(0)
    model = make_model(backbone).eval()
    wav = torch.randn(2, 8000)
    hop = model.cfg.frontend.hop_length
    with torch.no_grad():
        full = model(wav)
        state = model.init_state(2, wav.device, torch.float32)
        stepped = torch.cat(
            [model.step(wav[:, i : i + hop], state) for i in range(0, wav.shape[1], hop)], dim=1
        )
    n = min(full.shape[1], stepped.shape[1])
    assert torch.allclose(full[:, :n], stepped[:, :n], atol=1e-3)


def test_mamba2_forward_finite_long_sequence():
    # Regression: the quadratic SSD form must not overflow on long clips. With a large
    # dt the upper-triangle decay exponent is big; masking before exp keeps it finite
    # (an unmasked `exp(rel) * mask` produced inf*0 = nan and diverged training).
    torch.manual_seed(0)
    mixer = Mamba2Config(headdim=16, d_state=16).build(64, depth=2).eval()
    mixer.dt_bias.data.fill_(4.0)
    with torch.no_grad():
        out = mixer(torch.randn(1, 300, 64))
    assert torch.isfinite(out).all()


def test_frontend_forward_step_equivalence():
    torch.manual_seed(0)
    fe = MelFrontend(FrontendConfig()).eval()
    wav = torch.randn(2, 16000)
    hop = fe.cfg.hop_length
    with torch.no_grad():
        full = fe(wav)
        state = fe.init_state(2, wav.device, wav.dtype)
        stepped = torch.cat(
            [fe.step(wav[:, i : i + hop], state) for i in range(0, 16000, hop)], dim=1
        )
    n = min(full.shape[1], stepped.shape[1])
    assert torch.allclose(full[:, :n], stepped[:, :n], atol=1e-4)

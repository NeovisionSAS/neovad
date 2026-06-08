import pytest

from neovad.config import ModelConfig, NeoVADConfig
from neovad.nn.attention import DiffAttnConfig, GQAConfig
from neovad.nn.mamba import Mamba2Config
from neovad.nn.mixer import StreamingMixer


def test_config_roundtrip(tmp_path):
    cfg = NeoVADConfig(name="test", model=ModelConfig(dim=96, depth=3, mixer={"kind": "diffattn"}))
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    assert NeoVADConfig.load(path) == cfg


@pytest.mark.parametrize(
    "kind,expected",
    [("gqa", GQAConfig), ("diffattn", DiffAttnConfig), ("mamba2", Mamba2Config)],
)
def test_mixer_discriminated_union(kind, expected):
    cfg = ModelConfig(mixer={"kind": kind})
    assert isinstance(cfg.mixer, expected)
    assert cfg.mixer.kind == kind


def test_registry_unknown_kind():
    with pytest.raises(KeyError):
        StreamingMixer.by_name("does-not-exist")


def test_registry_lists_all_backbones():
    assert set(StreamingMixer.names()) == {"gru", "gqa", "mla", "diffattn", "mamba2"}

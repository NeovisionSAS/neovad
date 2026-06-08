# neovad

Small, streaming, CPU-friendly **Voice Activity Detection** with pluggable modern
backbones. Built at [Neovision](https://neovision.fr) to fire **only on the
foreground speaker** — background noise and secondary voices should *not* trigger
activation — and to beat [Silero VAD](https://github.com/snakers4/silero-vad) on
streaming decision latency while staying under a ~2 MB / sub-millisecond-per-chunk
CPU envelope.

Unlike a plain speech/non-speech VAD, neovad emits a per-frame 3-way decision —
`non-speech` / `primary` / `secondary` — so a real-time telephony agent can gate on
the locked primary speaker and ignore an interfering voice in the room or on the line.

## Why

In production telephony the denoiser lets non-stationary noise and background voices
leak through as false negatives, and speaker-isolation models are no longer enough.
A *stateful* model can instead lock onto the current dominant speaker and reject the
rest. neovad makes that the design centre and lets you A/B the architectures that
might deliver it.

## The one pluggable axis: the backbone

Every model is `frontend → N × ResidualBlock(RMSNorm → mixer → RMSNorm → SwiGLU) → head`.
The only thing that varies is the **sequence mixer**, and every mixer satisfies one
hard contract: a parallel causal `forward` (training) and a recurrent `step`
(streaming) that are provably equivalent. Swapping backbones is a one-line config
change.

| backbone   | what it is | streaming state | why it's here |
|------------|------------|-----------------|---------------|
| `gru`      | causal GRU | hidden `(L,B,H)` | proven-simple baseline (Silero-class control) |
| `gqa`      | grouped-query attention + RoPE, sliding window | windowed KV ring buffer | conventional efficient-attention reference |
| `mla`      | DeepSeek multi-head **latent** attention | compressed latent KV cache | smallest KV cache; the DeepSeek-V3 signature |
| `diffattn` | **Differential Attention** (two-softmax) | windowed KV ring buffer | common-mode cancellation kills background/secondary-voice leakage — the primary attention backbone |
| `mamba2`   | **Mamba-2** selective SSD, pure-PyTorch CPU step | `O(1)` SSM + conv state | constant per-step cost over a multi-minute call; its state implicitly tracks *who the dominant speaker is* |

All share the same modern building blocks (RMSNorm, SwiGLU, RoPE) and the same
streaming-state machinery. Add a backbone = subclass `StreamingMixer`, declare a
`MixerConfig`, done.

## Install

```bash
# inference only (torch + numpy + pydantic, CPU-friendly)
uv add git+https://github.com/NeovisionSAS/neovad.git
# or
pip install "git+https://github.com/NeovisionSAS/neovad.git"

# with the training engine and dataset tooling
pip install "neovad[train] @ git+https://github.com/NeovisionSAS/neovad.git"

# with the Silero / TEN-VAD comparison harness
pip install "neovad[bench] @ git+https://github.com/NeovisionSAS/neovad.git"
```

## Use as a library

### Streaming inference (the deployment path)

```python
import torch
from neovad import VADModel, StreamingVAD

model = VADModel.from_config("configs/mamba2.yaml")  # or a trained checkpoint
vad = StreamingVAD(model)                            # 16 kHz, 10 ms hops

# feed 10 ms chunks (160 samples @ 16 kHz) as they arrive
for chunk in audio_chunks(hop=160):
    probs = vad.push(chunk)        # foreground-speech probability per 10 ms frame
    if vad.is_speaking:            # hysteresis-smoothed gate
        ...                        # forward audio to STT
vad.reset()                        # at the call boundary
```

### Train a model

```python
from neovad import NeoVADConfig, train

cfg = NeoVADConfig.load("configs/mamba2.yaml")
train(cfg)                          # Lightning under the hood; multi-GPU aware
```

### From the CLI

```bash
neovad list-backbones                       # gru gqa mla diffattn mamba2
neovad download --root /disk/manual         # fetch training + eval datasets
neovad train  configs/mamba2.yaml
neovad bench  --checkpoint runs/mamba2/best.ckpt   # latency + accuracy vs Silero
neovad infer  audio.wav --checkpoint ...
```

## How it beats Silero

Silero's weakness is not raw compute (it is already sub-millisecond) but a
several-hundred-millisecond *decision* delay on speech→silence transitions, plus
speaker-agnostic firing. neovad targets both: a 10 ms hop with zero look-ahead
(`center=False`), smoothing kept *outside* the weights (tunable hysteresis), and a
foreground-only head no speaker-agnostic VAD can match. The benchmark reports RTF,
per-chunk wall-time, model size, **transition-decision latency**, per-condition
ROC-AUC (AVA-Speech: clean / +music / +noise), and false-activation rate on
interferer-only segments.

## Datasets

Training data is synthesized on the fly (no pre-rendered set): one primary speaker
(LibriSpeech / Common Voice) + 1–3 interferers + MUSAN/DNS5 noise + room impulse
response + telephony codec degradation, with labels derived from the **clean** primary
reference. `neovad download` fetches the sources into `/disk/manual`. WHAM! is
CC BY-NC and is eval-only, never in the shipped mix.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and the research
behind every choice.

## License

Apache-2.0 © Neovision

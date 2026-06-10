# neovad — architecture & methodology

This document records *why* neovad is built the way it is. Every choice is grounded in
a survey of recent (2024–2026) speech and sequence-modelling work; the goal was to
take the building blocks of very recent models (DeepSeek-V3, Mamba-2, Differential
Transformer, Llama) and assemble the smallest causal model that fires on the
foreground speaker alone.

## 1. The problem

A real-time telephony agent runs STT behind a denoiser. The denoiser lets
non-stationary noise and *background voices* through as false negatives, and
commercial speaker-isolation models are no longer enough. The team's own
conclusion: a *stateful* model that locks onto the current dominant speaker and rejects
everything else — "a Mamba that identifies the current speaker via its hidden state".

So neovad is not a plain speech/non-speech VAD. It emits a per-frame decision over
`{non-speech, primary, secondary}` and gates on `primary`. Background noise
(`non-speech`) and interfering voices (`secondary`) do not trigger activation.

## 2. What we must beat: Silero VAD

Silero v6 (~0.46–1.8M params depending on count method, ~2 MB, RTF ≈ 0.009 on one CPU
thread) is the incumbent. Its STFT → 4×Conv1d → 128-unit LSTM → sigmoid runs on fixed
512-sample (32 ms) chunks carrying a `(2,1,128)` LSTM state. Its documented weakness is
**not** per-chunk compute but a several-hundred-millisecond *decision* delay on
speech→silence transitions, plus speaker-agnostic firing (it degrades on noisy
overlapping-speaker audio such as MSDWild — to recompute in our own harness rather than
quote a single unverified number). TEN-VAD (2025, open, ~32% lower RTF, faster
transitions) and ai-coustics Quail (commercial, lifts noisy balanced accuracy 79→90%)
show the headroom. neovad targets the *decision-latency* and *foreground* axes, not a
raw-RTF race.

## 3. Design

```
waveform → MelFrontend (causal log-mel + streaming PCEN)
         → Linear(n_mels → dim)
         → N × ResidualBlock( RMSNorm → StreamingMixer → RMSNorm → SwiGLU )
         → RMSNorm
         → head → per-frame logits {non-speech, primary, secondary}
```

The single pluggable axis is the **StreamingMixer**. Everything else (frontend, block
shape, norms, FFN, head) is shared. A backbone is a `StreamingMixer` subclass plus a
typed `MixerConfig`; swapping it is one line of config.

### 3.1 The streaming contract (the central invariant)

Every `StreamingMixer` — and the whole model and frontend — exposes two paths over one
set of weights:

* `forward([B,T,D])` — parallel, causal, for training.
* `init_state` + `step([B,1,D], state)` — recurrent, for streaming inference.

They are **provably equivalent** within float tolerance, verified by a parametrized
test over all backbones (`tests/test_streaming.py`, ~1e-6 in practice). This is what
lets us train in parallel on GPUs and serve frame-by-frame on CPU with identical
behaviour. Fragile spots (causal masks, KV ring-buffer windowing, RoPE absolute
position, PCEN/STFT state carry, Mamba-2 recurrence vs its quadratic form) are exactly
what the test guards.

### 3.2 Frontend — fixed log-mel + streaming PCEN

`sample_rate=16000, n_fft=512, win=400 (25 ms), hop=160 (10 ms), n_mels=64, center=False`
(zero look-ahead). PCEN (per-channel energy normalization) replaces log compression:
its first-order IIR smoother suppresses stationary background and normalizes loudness —
the leaked-noise complaint — for a few multiplies per band, and streams losslessly via a
carried per-band state. Learnable front-ends (LEAF ~300×, EfficientLEAF ~10× slower than
mel and no consistent accuracy win) were rejected on the latency budget. The mel
filterbank uses the HTK formula, verified against `torchaudio` in tests; PCEN constants
are the librosa defaults, made trainable per band.

10 ms frames are 3.2× finer than Silero's 32 ms — the structural lever against Silero's
transition delay.

### 3.3 Backbones (the comparison)

| backbone | recent-model lineage | streaming state | role |
|---|---|---|---|
| `gru` | classic RNN (Silero/Personal-VAD) | hidden `(L,B,H)` | proven baseline / CPU-fastest |
| `gqa` | Llama-2 / Mistral GQA + RoPE, sliding window | windowed KV ring buffer | efficient-attention reference |
| `mla` | DeepSeek-V2/V3 Multi-head Latent Attention | compressed latent KV + decoupled RoPE key | smallest KV cache |
| `diffattn` | Differential Transformer (2024) | windowed KV ring buffer (two K groups) | **primary** — noise/secondary cancellation |
| `mamba2` | Mamba-2 SSD (2024) | `O(1)` SSM state + short-conv tail | **headline** — stateful speaker tracking |

Shared modern modules: **RMSNorm**, **SwiGLU**, **RoPE** (attention only), plus a
shared **CausalDepthwiseConv1d** (frontend stem / Mamba short conv). Tricks that only
pay off at billion scale — QK-norm, NSA, MoE, sandwich-norm — were deliberately skipped
for a <5M-param model.

Two design notes worth recording:

* **Differential Attention** is the primary attention bet because its
  `softmax(A1) − λ·softmax(A2)` cancels the common mode (diffuse attention to background
  noise / secondary voices that both maps share) like a differential amplifier — a
  direct match to the failure mode. `λ_init = 0.8 − 0.6·exp(−0.3·(l−1))` rises with depth
  (paper schedule).
* **Mamba-2** is implemented in pure PyTorch (no `mamba_ssm` CUDA/Triton, which does not
  run on the CPU target). The training `forward` uses the exact **dual quadratic SSD
  form** — structured masked attention, `scores[t,s] = (C_t·B_s)·exp(cumdecay_t −
  cumdecay_s)` masked causal — which is parallel and exactly equal to the `O(1)`-state
  recurrence used in `step` (verified).

### 3.4 Head — foreground gating

`AttractorHead` (default): per-frame cosine similarity to learnable primary/secondary
speaker attractors plus a learnable non-speech bias, a lightweight EEND-SAA-style stand-in
that gives the foreground/background decision its own geometry. `LinearHead` is the plain
alternative. Both are pointwise (no streaming state).

## 4. Data — on-the-fly synthesis

No pre-rendered training set. Each example (`MixtureSynthesizer`): one primary speaker
(LibriSpeech / MUSAN speech) placed with leading/trailing silence + 0–3 interferers at
0.1–0.8× gain + MUSAN/DNS noise & music at −5..20 dB SNR + optional room impulse response
+ optional 8 kHz µ-law telephony round-trip. **Labels are derived from the clean primary
reference (energy gate + median smoothing) before mixing** — only the primary is
`primary`, interferers are `secondary`, everything else `non-speech`. That is what
teaches robustness to noise and secondary voices (tested in `tests/test_synth.py`:
heavy noise never creates speech frames; interferers produce `secondary`).

Loss is cost-sensitive cross-entropy upweighting `primary` (be conservative about firing
on anyone but the locked speaker). The gate hysteresis lives *outside* the weights
(`HysteresisGate`) so latency/stability is tuned at deploy time, not baked into training.

### Datasets (in `/disk/manual`, fetched by `neovad download`)

| dataset | role | size | license | registration |
|---|---|---|---|---|
| LibriSpeech train-clean-100 (SLR12) | clean speech | 6.3 GB | CC BY 4.0 | no |
| MUSAN (SLR17) | noise / music / babble | 11 GB | CC BY 4.0 | no |
| RIRS_NOISES (SLR28) | room impulse responses | 2 GB | Apache-2.0 | no |

Eval/extension sets (documented, fetched separately): AVA-Speech (per-condition
clean/+music/+noise), VoxConverse (overlap), MSDWild (the noisy-overlap proving ground),
CHiME-6, DNS5 noise, Common Voice. **WHAM! is CC BY-NC → eval-only, never in the shipped
mix.**

## 5. Benchmark (honest, current state)

`neovad bench --all-backbones --silero` on 1 CPU thread, 20 s audio, **untrained eager
fp32** models at dim=128/depth=4:

| model | params (M) | size (MB) | RTF |
|---|---|---|---|
| diffattn | 0.68 | 2.75 | 0.204 |
| gqa | 0.58 | 2.35 | 0.172 |
| gru | 0.88 | 3.55 | 0.102 |
| mamba2 | 0.89 | 3.58 | 0.199 |
| mla | 0.69 | 2.78 | 0.209 |
| **silero-v6** | 0.46 | 2.19 | **0.009** |

**First trained result (pipeline validation).** A 6-epoch smoke run of `mamba2`
(0.89M params) on the on-the-fly noisy multi-speaker mix reaches **val/primary_f1
≈ 0.91** (train loss 0.58→0.30, val/acc 0.80→0.85) — i.e. the 0.89M-param stateful
model isolates the foreground speaker at ~91% F1 with interferers and noise present.
This is a short validation run, not a tuned model; a longer clean→noisy curriculum is
expected to improve calibration on out-of-distribution clean wideband audio.

Reading the latency honestly: neovad already matches Silero on **size** and is comfortably
real-time (RTF ≤ 0.21 = ≥5× faster than real time), but it is **not yet beating Silero on
raw RTF** — expected, because (a) these are eager fp32 PyTorch graphs vs Silero's
optimized JIT/ONNX, and (b) neovad runs 3.2× more frames/second (10 ms vs 32 ms hop). The
harness exists precisely to drive this number down and to measure the metric Silero is
actually weak on — transition-decision latency — which raw RTF hides.

## 6. Compression & export (`neovad export`, `neovad.export.ModelExporter`)

Correction to an earlier draft: **Silero ships fp32**, not int8 — it explicitly dropped
quantization (ARM/mobile instability); its speed is operator *fusion* (fp32 JIT/ONNX) plus
tiny tensors. So fused-fp32 is the main speed lever, int8 is an optional size win, and
`torch.fft.rfft` — not quantization — is the real ONNX blocker. neovad offers three paths:

| path | what | result | when |
|---|---|---|---|
| `quantize_dynamic` | torch int8 on Linear/GRU | gru **3.55 → 1.00 MB** (under Silero's 2 MB), keeps the streaming `step` API | Python CPU deployment; re-check AUC after |
| `jit_trace` | TorchScript fused graph | exact, faster offline (Silero's fp32 recipe) | offline / batch in Python |
| `onnx` | ONNX Runtime via the DFT-matmul frontend (`rfft` is not exportable) | faithful (1e-5) for mamba2/diffattn/gqa; GRU export is lossy; **int8-ONNX grows tiny RNNs — skip it** | cross-language (C++/Rust) CPU |

"Is ONNX best?" — not universally: for the GRU baseline, **torch int8 (1 MB)** wins; for
mamba/attention deployed cross-platform, **ONNX fp32** is faithful and portable; ONNX-int8
is counter-productive at this size.

## 7. Roadmap

1. **Fused-fp32 first** (JIT) then optional int8 — the realistic path to ≤ Silero RTF;
   int8 is gated on a post-quant AUC check, not assumed free.
2. **Decision-latency metric** in the bench (ms from true offset to detected offset) —
   the axis where the 10 ms hop + zero look-ahead structurally beats Silero.
3. **Accuracy eval harness** on AVA-Speech / VoxConverse / MSDWild: per-condition
   ROC-AUC and the foreground-specific `secondary_false_fire` rate (now logged each val).
4. **Sharper labels** — Montreal Forced Alignment on LibriSpeech transcripts (gold
   standard) over the energy gate, to tighten transition labels for the latency goal.
5. **More real overlap/noise** — AMI (CC-BY real crosstalk), FSD50K/DEMAND (real noise),
   VoxCeleb2 (diverse interferer pool); see §4.
6. **EEND-SAA causal-aware labelling** + optional **FiLM speaker-conditioning** on the
   same weights.

## 7. Risks (carried from the design review)

* Pure-torch Mamba-2 CPU `step` may be the latency bottleneck → `gru` is the proven
  fallback; profile early.
* Foreground labelling is the riskiest part: synthetic "dominant speaker" may not
  transfer to a faint/distant caller (the known hard case) → keep a
  plain-VAD ablation (drop the secondary head) as a safety net; eval on faint/far primary.
* Telephony domain gap: validate on codec-degraded audio and real flagged calls, not
  clean wideband.
* RMSNorm CPU kernels can regress vs LayerNorm on some builds → benchmark on the target
  runtime.

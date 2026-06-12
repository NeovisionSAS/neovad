---
license: apache-2.0
language: multilingual
pipeline_tag: voice-activity-detection
tags:
  - vad
  - voice-activity-detection
  - streaming
  - mamba
  - foreground-speaker
library_name: neovad
---

# neovad — mamba2 (foreground-speaker streaming VAD)

Small (0.89M params, 3.6 MB fp32 / 1.0 MB int8) streaming CPU voice activity detector
that fires **only on the foreground speaker**: per-frame `{non-speech, primary,
secondary}` so background noise and interfering voices do not trigger activation.
Mamba-2 (SSD) backbone with O(1) streaming state; 10 ms decision granularity with zero
look-ahead. Built by [Neovision](https://neovision.fr); code at
[NeovisionSAS/neovad](https://github.com/NeovisionSAS/neovad).

## Results (synthetic noisy multi-speaker validation)

| metric | value |
|---|---|
| primary-speaker frame F1 | **0.955** (P 0.93 / R 0.98) |
| false-fire on interferer-only frames | **0.19** (down from 0.55 at init) |
| frame accuracy | 0.93 |
| ONNX (1 CPU thread, offline 2 s windows) | RTF **0.005** |
| streaming step @30 ms cadence (1 thread) | RTF 0.089 |

Trained on on-the-fly mixtures: LibriSpeech primary + 0–3 interfering speakers
(0.1–0.8× gain) + MUSAN noise/music at −5..20 dB SNR + room impulse responses +
8 kHz µ-law telephony round-trip. Labels derived from the clean primary reference.

## Usage

```python
# pip install "git+ssh://git@github.com/NeovisionSAS/neovad.git"
from neovad import StreamingVAD

vad = StreamingVAD.from_pretrained("gru", input_sample_rate=8000)
for chunk in audio_stream():
    probs = vad.push(chunk)
    if vad.is_speaking:
        ...  # forward to STT
```

`VADModel.from_pretrained` resolves weights bundled in the wheel first, then this
repository.

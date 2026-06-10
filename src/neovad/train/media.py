import lightning as L
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402  (backend must be set before import)

from neovad.config import NeoVADConfig
from neovad.data.sources import Datasets
from neovad.data.synth import Mixture, MixtureSynthesizer
from neovad.nn.head import SpeechClass


class AnalysisLogger(L.Callback):
    """Logs everything needed to eyeball a run, to TensorBoard, with folded tags:

    * ``audio/`` — the clean primary (before augmentation) and the final mixture
      (after interferers + noise + reverb + telephony), so you can *hear* the aug.
    * ``media/`` — the PCEN mel the model sees, and a per-frame figure overlaying the
      ground-truth labels with the predicted foreground probability and argmax class.
    * ``hist/`` — weight and gradient histograms; ``train/grad_norm`` scalar.

    Examples use fixed seeds so the same scenes recur every epoch and are comparable.
    Degrades to a no-op when no dataset is present (e.g. CI).
    """

    def __init__(self, cfg: NeoVADConfig):
        super().__init__()
        self.log_cfg = cfg.train.log
        self.sr = cfg.model.frontend.sample_rate
        self.hop = cfg.model.frontend.hop_length
        speech = Datasets.files("speech", cfg.data.root, cfg.data.speech_sources)
        noise = Datasets.files("noise", cfg.data.root, cfg.data.noise_sources) + Datasets.files(
            "music", cfg.data.root, cfg.data.noise_sources
        )
        rir = Datasets.files("rir", cfg.data.root)
        self.synth = (
            MixtureSynthesizer(cfg.data, cfg.model.frontend, speech, noise, rir, seed=12345)
            if speech
            else None
        )
        self._grad_epoch = -1

    @staticmethod
    def writer(trainer):
        return getattr(trainer.logger, "experiment", None)

    @staticmethod
    def norm_audio(x: torch.Tensor) -> torch.Tensor:
        peak = x.abs().max()
        return x / peak if peak > 1 else x

    def mel_image(self, mel: torch.Tensor) -> torch.Tensor:
        m = mel.t()  # [n_mels, T]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m.flip(0).unsqueeze(0)  # [1, n_mels, T], low freq at bottom

    def figure(self, ex: Mixture, prob: torch.Tensor, pred: torch.Tensor):
        labels = ex.labels.cpu().numpy()
        prob, pred = prob.cpu().numpy(), pred.cpu().numpy()
        n = min(len(prob), len(labels))
        t_wav = np.arange(ex.mix.shape[0]) / self.sr
        t_fr = np.arange(n) * self.hop / self.sr
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
        ax[0].plot(t_wav, ex.mix.cpu().numpy(), lw=0.4)
        ax[0].set_title("augmented waveform (model input)")
        ax[1].plot(t_fr, prob[:n], label="P(foreground)", lw=1.2)
        ax[1].fill_between(
            t_fr, 0, labels[:n] == int(SpeechClass.PRIMARY), alpha=0.25, label="primary GT"
        )
        ax[1].fill_between(
            t_fr, 0, labels[:n] == int(SpeechClass.SECONDARY), alpha=0.25, label="secondary GT"
        )
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_title("foreground probability vs ground truth")
        ax[1].legend(loc="upper right", fontsize=7)
        ax[2].step(t_fr, pred[:n], where="mid")
        ax[2].set_yticks([0, 1, 2])
        ax[2].set_title("argmax class — 0 non-speech / 1 primary / 2 secondary")
        ax[2].set_xlabel("time (s)")
        return fig

    def on_validation_epoch_end(self, trainer, pl):
        w = self.writer(trainer)
        if w is None or trainer.sanity_checking:
            return
        step = trainer.global_step
        if self.log_cfg.histograms:
            for name, p in pl.model.named_parameters():
                w.add_histogram(f"hist/weight/{name}", p.detach().float().cpu(), step)
        if not self.log_cfg.media or self.synth is None:
            return
        if trainer.current_epoch % self.log_cfg.media_every_n_epochs != 0:
            return
        examples = []
        for i in range(self.log_cfg.n_media_samples):
            self.synth.reseed(1000 + i)
            examples.append(self.synth.mix_components())
        mix = torch.stack([e.mix for e in examples]).to(pl.device)
        with torch.no_grad():
            logits = pl.model(mix)
            prob = pl.model.speech_probability(logits)
            pred = logits.argmax(-1)
            mel = pl.model.frontend(mix).cpu()
        for i, ex in enumerate(examples):
            w.add_audio(
                f"audio/{i}_primary_clean", self.norm_audio(ex.primary), step, sample_rate=self.sr
            )
            w.add_audio(
                f"audio/{i}_augmented_mix", self.norm_audio(ex.mix), step, sample_rate=self.sr
            )
            w.add_image(f"media/{i}_mel", self.mel_image(mel[i]), step)
            w.add_figure(f"media/{i}_labels_vs_pred", self.figure(ex, prob[i], pred[i]), step)

    def on_before_optimizer_step(self, trainer, pl, optimizer):
        if not self.log_cfg.histograms or trainer.current_epoch == self._grad_epoch:
            return
        self._grad_epoch = trainer.current_epoch
        w = self.writer(trainer)
        if w is None:
            return
        total = 0.0
        for name, p in pl.model.named_parameters():
            if p.grad is not None:
                w.add_histogram(
                    f"hist/grad/{name}", p.grad.detach().float().cpu(), trainer.global_step
                )
                total += float(p.grad.detach().norm() ** 2)
        w.add_scalar("train/grad_norm", total**0.5, trainer.global_step)

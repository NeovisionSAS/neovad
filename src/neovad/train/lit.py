import math
from pathlib import Path

import lightning as L
import torch
from torch import Tensor

from neovad.config import NeoVADConfig
from neovad.data.dataset import SynthVADDataset
from neovad.models.vad import VADModel
from neovad.nn.head import SpeechClass
from neovad.train.loss import FrameVADLoss


class NeoVADLit(L.LightningModule):
    """LightningModule wrapping a :class:`VADModel`, its loss, optimizer schedule, and
    foreground metrics. Training uses the parallel ``forward`` path; the saved weights
    serve via ``step`` unchanged."""

    def __init__(self, cfg: NeoVADConfig):
        super().__init__()
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg
        self.model = VADModel(cfg.model)
        self.loss = FrameVADLoss(cfg.train.loss, self.model.head.n_classes)

    def forward(self, wav: Tensor) -> Tensor:
        return self.model(wav)

    @staticmethod
    def align(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        t = min(logits.shape[1], labels.shape[1])
        return logits[:, :t], labels[:, :t]

    def training_step(self, batch, _):
        wav, labels = batch
        logits, labels = self.align(self.model(wav), labels)
        loss = self.loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        wav, labels = batch
        logits, labels = self.align(self.model(wav), labels)
        prob = self.model.speech_probability(logits)
        pred, tgt = prob > 0.5, labels == int(SpeechClass.PRIMARY)
        tp = (pred & tgt).sum().float()
        fp = (pred & ~tgt).sum().float()
        fn = (~pred & tgt).sum().float()
        self.log_dict(
            {
                "val/loss": self.loss(logits, labels),
                "val/primary_f1": 2 * tp / (2 * tp + fp + fn + 1e-9),
                "val/acc": (logits.argmax(-1) == labels).float().mean(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
        )
        total = (
            self.cfg.train.max_epochs * self.cfg.data.steps_per_epoch // self.cfg.train.accumulate
        )
        warmup = self.cfg.train.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @classmethod
    def run(cls, cfg: NeoVADConfig) -> VADModel:
        L.seed_everything(cfg.train.seed, workers=True)
        lit = cls(cfg)
        out = Path(cfg.train.output_dir) / cfg.name
        out.mkdir(parents=True, exist_ok=True)
        ckpt = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(out), monitor="val/primary_f1", mode="max", save_top_k=1, filename="best"
        )
        trainer = L.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=cfg.train.devices,
            precision=cfg.train.precision,
            gradient_clip_val=cfg.train.grad_clip,
            accumulate_grad_batches=cfg.train.accumulate,
            limit_train_batches=cfg.data.steps_per_epoch,
            limit_val_batches=20,
            val_check_interval=cfg.train.val_interval,
            default_root_dir=str(out),
            callbacks=[ckpt],
        )
        trainer.fit(lit, SynthVADDataset.loader(cfg), SynthVADDataset.loader(cfg))
        lit.model.save(out / "model.pt")
        return lit.model

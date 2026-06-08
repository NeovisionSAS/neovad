import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neovad.config import LossConfig
from neovad.nn.head import SpeechClass


class FrameVADLoss(nn.Module):
    """Cost-sensitive per-frame loss.

    Class weights upweight PRIMARY frames so the model pays more to miss the foreground
    speaker than to mislabel an interferer — the conservative bias the project wants
    (fire on the locked speaker, tolerate dropping a secondary voice). Falls back to a
    binary objective when the head has a single logit.
    """

    def __init__(self, cfg: LossConfig, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = cfg.label_smoothing
        self.register_buffer(
            "weight", torch.tensor(cfg.class_weights[:n_classes], dtype=torch.float32)
        )

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.n_classes == 1:
            target = (labels == int(SpeechClass.PRIMARY)).float()
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target)
        return F.cross_entropy(
            logits.reshape(-1, self.n_classes),
            labels.reshape(-1),
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )

"""Early fusion: project both modalities, concatenate, classify (with modality dropout)."""
from __future__ import annotations

import torch
from torch import nn

from .image_model import ImageClassifier
from .text_model import TextClassifier


def drop_modalities(z_img: torch.Tensor, z_txt: torch.Tensor, p: float, training: bool = True):
    """With probability p per sample, zero ONE modality (chosen uniformly). Never both."""
    if not training or p <= 0:
        return z_img, z_txt
    n = z_img.shape[0]
    drop = torch.rand(n, device=z_img.device) < p
    pick_img = torch.rand(n, device=z_img.device) < 0.5
    keep_img = (~(drop & pick_img)).to(z_img.dtype).unsqueeze(1)
    keep_txt = (~(drop & ~pick_img)).to(z_txt.dtype).unsqueeze(1)
    return z_img * keep_img, z_txt * keep_txt


def _projection(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())


class MultimodalClassifier(nn.Module):
    def __init__(self, image_model: ImageClassifier, text_model: TextClassifier, num_classes: int,
                 hidden: int = 512, dropout: float = 0.3, modality_dropout: float = 0.1):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.img_proj = _projection(image_model.feat_dim, hidden)
        self.txt_proj = _projection(text_model.feat_dim, hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )
        self.modality_dropout = modality_dropout

    def forward(self, image, input_ids, attention_mask, return_features: bool = False):
        z_img = self.img_proj(self.image_model.forward_features(image))
        z_txt = self.txt_proj(self.text_model.forward_features(input_ids, attention_mask))
        z_img, z_txt = drop_modalities(z_img, z_txt, self.modality_dropout, self.training)
        fused = torch.cat([z_img, z_txt], dim=1)
        logits = self.classifier(fused)
        return (logits, fused) if return_features else logits

    def param_groups(self, lr_image: float, lr_text: float, lr_head: float, weight_decay: float) -> list[dict]:
        head = [*self.img_proj.parameters(), *self.txt_proj.parameters(), *self.classifier.parameters()]
        return [
            {"params": list(self.image_model.backbone.parameters()), "lr": lr_image, "weight_decay": weight_decay},
            {"params": list(self.text_model.encoder.parameters()), "lr": lr_text, "weight_decay": weight_decay},
            {"params": head, "lr": lr_head, "weight_decay": weight_decay},
        ]

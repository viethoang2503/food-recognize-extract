"""Image-only classifier on a timm backbone."""
from __future__ import annotations

import timm
from timm.data import resolve_model_data_config
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)  # pooled features
        self.feat_dim = int(self.backbone.num_features)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def data_mean_std(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        cfg = resolve_model_data_config(self.backbone)
        return tuple(cfg["mean"]), tuple(cfg["std"])

    def forward_features(self, image):
        return self.backbone(image)

    def forward(self, image, return_features: bool = False):
        feats = self.forward_features(image)
        logits = self.head(self.dropout(feats))
        return (logits, feats) if return_features else logits

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": list(self.head.parameters()), "lr": lr_head, "weight_decay": weight_decay},
        ]

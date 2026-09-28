"""Text-only classifier on a Hugging Face encoder with masked mean pooling."""
from __future__ import annotations

from torch import nn
from transformers import AutoConfig, AutoModel


class TextClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.1):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:  # random init from the config only (used by fast tests)
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.feat_dim = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward_features(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask, return_features: bool = False):
        feats = self.forward_features(input_ids, attention_mask)
        logits = self.head(self.dropout(feats))
        return (logits, feats) if return_features else logits

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
        return [
            {"params": list(self.encoder.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": list(self.head.parameters()), "lr": lr_head, "weight_decay": weight_decay},
        ]

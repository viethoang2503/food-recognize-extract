"""Model definitions and a config-driven factory."""
from __future__ import annotations

from torch import nn

from .fusion import MultimodalClassifier, drop_modalities
from .image_model import ImageClassifier
from .text_model import TextClassifier

MODALITIES = ("image", "text", "multimodal")
__all__ = ["MODALITIES", "ImageClassifier", "MultimodalClassifier", "TextClassifier", "build_model", "drop_modalities"]


def build_model(modality: str, cfg: dict, num_classes: int) -> nn.Module:
    if modality == "image":
        c = cfg["image"]
        return ImageClassifier(c["backbone"], num_classes, pretrained=bool(c["pretrained"]),
                               drop_rate=float(c["drop_rate"]))
    if modality == "text":
        c = cfg["text"]
        return TextClassifier(c["model_name"], num_classes, pretrained=bool(c["pretrained"]),
                              drop_rate=float(c["drop_rate"]))
    if modality == "multimodal":
        f = cfg["fusion"]
        return MultimodalClassifier(
            build_model("image", cfg, num_classes), build_model("text", cfg, num_classes), num_classes,
            hidden=int(f["hidden"]), dropout=float(f["dropout"]), modality_dropout=float(f["modality_dropout"]),
        )
    raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}")

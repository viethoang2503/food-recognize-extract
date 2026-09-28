"""CLIP zero-shot classification from cached image embeddings."""
from __future__ import annotations

import shutil
from collections.abc import Sequence

import numpy as np

from ..data.text_utils import class_to_phrase


def class_prompts(classes: Sequence[str], template: str) -> list[str]:
    return [template.format(class_to_phrase(c)) for c in classes]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def zero_shot_logits(image_pooled: np.ndarray, prompt_embeds: np.ndarray, scale: float = 100.0) -> np.ndarray:
    return scale * _normalize(image_pooled) @ _normalize(prompt_embeds).T

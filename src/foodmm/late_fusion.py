"""Late fusion: weighted average of image and text class probabilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_preds(run_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    path = Path(run_dir) / f"preds_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def check_aligned(a: dict, b: dict) -> None:
    if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
        raise ValueError("Prediction files are not aligned: ids differ in content or order")


def combine(p_img: np.ndarray, p_txt: np.ndarray, w: float) -> np.ndarray:
    return w * p_img + (1.0 - w) * p_txt


def search_weight(p_img: np.ndarray, p_txt: np.ndarray, labels: np.ndarray,
                  step: float = 0.05) -> tuple[float, list[dict[str, float]]]:
    """Grid-search w in [0, 1]; ties keep the smallest w."""
    if not 0 < step <= 1:
        raise ValueError(f"step must be in (0, 1], got {step}")
    n_steps = int(round(1.0 / step))
    curve = []
    for i in range(n_steps + 1):
        w = min(1.0, round(i * step, 6))
        acc = float((combine(p_img, p_txt, w).argmax(axis=1) == labels).mean())
        curve.append({"w": w, "acc": acc})
    best = max(curve, key=lambda r: r["acc"])
    return best["w"], curve

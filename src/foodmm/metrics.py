"""Classification metrics shared by every run."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    k = min(k, scores.shape[1])
    topk = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def compute_metrics(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    """`scores` may be logits, probabilities or log-probabilities (only the ranking matters)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    preds = scores.argmax(axis=1)
    return {
        "acc": float((preds == labels).mean()),
        "top5": topk_accuracy(scores, labels, k),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                   labels=np.arange(scores.shape[1]), zero_division=0)),
        "n": int(len(labels)),
    }


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(classes):
        mask = labels == i
        n = int(mask.sum())
        rows.append({"class": c, "n": n, "acc": float((preds[mask] == i).mean()) if n else float("nan")})
    return pd.DataFrame(rows, columns=["class", "n", "acc"])


def confusion(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(labels, preds, labels=np.arange(n_classes))


def top_confused_pairs(preds: np.ndarray, labels: np.ndarray, classes: Sequence[str], k: int = 10) -> pd.DataFrame:
    cm = confusion(preds, labels, len(classes))
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis=None)[::-1][:k]
    rows = [{"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
            for i, j in zip(*np.unravel_index(order, cm.shape)) if cm[i, j] > 0]
    return pd.DataFrame(rows, columns=["true", "pred", "count"])

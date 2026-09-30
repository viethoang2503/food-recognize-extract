"""Uncertainty (bootstrap CI) and paired significance tests (McNemar) on per-sample correctness."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .late_fusion import load_preds

_EXACT_MAX = 50  # discordant pairs up to this count use the exact binomial test


def correct_vector(preds: dict) -> np.ndarray:
    return np.asarray(preds["logits"]).argmax(axis=1) == np.asarray(preds["labels"])


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of `values` (e.g. a 0/1 correctness vector)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(int(n_boot))])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar(a, b) -> tuple[int, int, float]:
    """Two-sided McNemar test for paired correctness; returns (only_b_correct, only_a_correct, p)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_b, only_a = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = only_b + only_a
    if n == 0:
        return only_b, only_a, 1.0
    if n <= _EXACT_MAX:
        k = min(only_b, only_a)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return only_b, only_a, min(1.0, p)
    chi2 = (abs(only_b - only_a) - 1) ** 2 / n
    return only_b, only_a, math.erfc(math.sqrt(chi2 / 2))


def aligned_correct(dir_a: str | Path, dir_b: str | Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    ca, cb = ({str(i): bool(c) for i, c in zip(p["ids"], correct_vector(p))}
              for p in (load_preds(dir_a, split), load_preds(dir_b, split)))
    common = sorted(set(ca) & set(cb))
    if not common:
        raise ValueError(f"{dir_a} and {dir_b} have no test ids in common")
    return np.array([ca[i] for i in common]), np.array([cb[i] for i in common])


def compare_runs(dir_a: str | Path, dir_b: str | Path, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    a, b = aligned_correct(dir_a, dir_b)
    d = b.astype(np.float64) - a.astype(np.float64)
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    only_b, only_a, p = mcnemar(a, b)
    return {"n": int(len(a)), "acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff": float(d.mean()),
            "diff_lo": lo, "diff_hi": hi, "only_a_correct": only_a, "only_b_correct": only_b, "mcnemar_p": p}

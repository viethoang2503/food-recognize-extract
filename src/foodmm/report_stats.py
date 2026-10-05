"""Extra statistics quoted in the final report, computed from the stored run outputs.

Bonferroni-corrected p-values, bootstrap CIs for any list of runs, a residual-leakage audit of the masked text,
paired VLM comparisons and the manual-grading summary (bootstrap CIs and label-based accuracy on the same ids).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, mcnemar

GRADES = ("grade_dish", "grade_ingredients", "grade_method")


def add_bonferroni(df: pd.DataFrame, p_col: str = "mcnemar_p") -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    out["n_comparisons"] = n
    out["p_bonferroni"] = np.minimum(1.0, out[p_col].astype(float) * n)
    return out


def run_ci_table(runs_dir: str | Path, runs: Sequence[str], n_boot: int, alpha: float, seed: int) -> pd.DataFrame:
    """Accuracy and percentile bootstrap CI for each run directory; missing runs give NaN."""
    from .late_fusion import load_preds
    from .stats import correct_vector

    rows = []
    for run in runs:
        try:
            c = correct_vector(load_preds(Path(runs_dir) / run, "test"))
        except FileNotFoundError:
            rows.append({"run": run, "n": np.nan, "acc": np.nan, "acc_lo": np.nan, "acc_hi": np.nan})
            continue
        lo, hi = bootstrap_ci(c, n_boot, alpha, seed)
        rows.append({"run": run, "n": len(c), "acc": float(c.mean()), "acc_lo": lo, "acc_hi": hi})
    return pd.DataFrame(rows, columns=["run", "n", "acc", "acc_lo", "acc_hi"])


def fold_accents(text: str) -> str:
    """Strip diacritics (phở -> pho, crème -> creme); other characters are kept."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_NON_ASCII_LETTER = re.compile(r"[^\x00-\x7f]")


def residual_leakage(texts: Sequence[str], pattern: re.Pattern[str] | None, top_k: int = 20) -> dict:
    """How often already-masked texts still contain a class word once accents are removed.

    `pattern` is the masking pattern of the mode the texts were masked with (e.g. strict); any match after
    folding is a dish-name variant that the mask missed because of diacritics.
    """
    n = len(texts)
    hits, non_ascii, tokens = 0, 0, Counter()
    for t in texts:
        t = str(t)
        non_ascii += bool(_NON_ASCII_LETTER.search(t))
        if pattern is None:
            continue
        found = [m.group(0) for m in pattern.finditer(fold_accents(t).lower())]
        hits += bool(found)
        tokens.update(re.sub(r"\s+", " ", f) for f in found)
    denom = max(n, 1)
    return {"n": n, "non_ascii_rate": non_ascii / denom, "folded_hit_rate": hits / denom,
            "top_tokens": tokens.most_common(top_k)}


def _dish_correct(records: Sequence[dict], classes: Sequence[str], threshold: float) -> dict[str, bool]:
    from .vlm.evaluate import map_dish_to_class

    return {str(r["id"]): bool(r["valid"] and map_dish_to_class(r["output"]["dish_name"], classes, threshold)
                               == r["label"]) for r in records}


def vlm_paired(records_a: Sequence[dict], records_b: Sequence[dict], classes: Sequence[str], threshold: float,
               n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Paired comparison of two VLM modes on their common ids (dish accuracy after mapping)."""
    ca, cb = _dish_correct(records_a, classes, threshold), _dish_correct(records_b, classes, threshold)
    ids = sorted(set(ca) & set(cb))
    a, b = np.array([ca[i] for i in ids]), np.array([cb[i] for i in ids])
    lo, hi = bootstrap_ci(b.astype(float) - a.astype(float), n_boot, alpha, seed)
    only_b, only_a, p = mcnemar(a, b)
    return {"n": len(ids), "acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "diff_lo": lo, "diff_hi": hi, "only_a_correct": only_a, "only_b_correct": only_b, "mcnemar_p": p,
            "valid_a": sum(bool(r["valid"]) for r in records_a), "valid_b": sum(bool(r["valid"]) for r in records_b)}


def graded_summary(sheet: pd.DataFrame, key: pd.DataFrame, classes: Sequence[str], threshold: float,
                   n_boot: int, seed: int, alpha: float = 0.05, pair: tuple[str, str] = ("image", "image_text")
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manual grades per mode with bootstrap CIs, next to label-based dish accuracy on the same rows.

    Returns (summary per mode, paired McNemar of the 0/1 grades between the two modes in `pair`).
    """
    from .vlm.evaluate import map_dish_to_class

    d = sheet.merge(key, on="row")
    d["label_acc"] = [float(isinstance(x, str) and map_dish_to_class(x, classes, threshold) == lab)
                      for x, lab in zip(d["dish_name"], d["label"])]
    rows = []
    for mode, g in d.groupby("mode", sort=True):
        row = {"mode": mode, "n": len(g)}
        for col in (*GRADES, "label_acc"):
            lo, hi = bootstrap_ci(g[col].astype(float), n_boot, alpha, seed)
            row.update({col: float(g[col].mean()), f"{col}_lo": lo, f"{col}_hi": hi})
        rows.append(row)
    summary = pd.DataFrame(rows)
    a = d[d["mode"] == pair[0]].set_index("id")
    b = d[d["mode"] == pair[1]].set_index("id")
    ids = sorted(set(a.index) & set(b.index))
    paired = []
    for col in ("grade_dish", "grade_method", "label_acc"):
        only_b, only_a, p = mcnemar(a.loc[ids, col].astype(bool).to_numpy(), b.loc[ids, col].astype(bool).to_numpy())
        paired.append({"grade": col, "a": pair[0], "b": pair[1], "n": len(ids), "only_a_correct": only_a,
                       "only_b_correct": only_b, "mcnemar_p": p})
    return summary, pd.DataFrame(paired)


def vlm_vs_classifier(records: Sequence[dict], clf_preds: dict, classes: Sequence[str], threshold: float,
                      n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Paired comparison of a VLM mode (dish mapped to a class) with a classifier on the VLM sample ids."""
    from .stats import correct_vector

    cv = _dish_correct(records, classes, threshold)
    cc = dict(zip(np.asarray(clf_preds["ids"]).astype(str), correct_vector(clf_preds)))
    ids = sorted(set(cv) & set(cc))
    v, c = np.array([cv[i] for i in ids]), np.array([bool(cc[i]) for i in ids])
    lo, hi = bootstrap_ci(c.astype(float) - v.astype(float), n_boot, alpha, seed)
    only_c, only_v, p = mcnemar(v, c)
    return {"n": len(ids), "acc_vlm": float(v.mean()), "acc_classifier": float(c.mean()),
            "diff": float(c.mean() - v.mean()), "diff_lo": lo, "diff_hi": hi, "only_vlm_correct": only_v,
            "only_classifier_correct": only_c, "mcnemar_p": p}


def classifier_errors(image_preds: dict, fusion_preds: dict, classes: Sequence[str], k: int = 10
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Most confused class pairs of the fusion model, and the classes most helped / hurt by adding text.

    The two prediction dicts must cover the same test ids in the same order (same manifest).
    """
    from .analysis import per_class_gain
    from .metrics import top_confused_pairs

    confused = top_confused_pairs(np.asarray(fusion_preds["logits"]).argmax(axis=1),
                                  np.asarray(fusion_preds["labels"]).astype(int), classes, k=k)
    gain = per_class_gain(image_preds, fusion_preds, classes)
    gain = gain[gain["n"] > 0].reset_index(drop=True)
    helped = gain.head(k).assign(side="most helped")
    hurt = gain.drop(helped.index).tail(k).iloc[::-1].assign(side="most hurt")
    return confused, pd.concat([helped, hurt], ignore_index=True)

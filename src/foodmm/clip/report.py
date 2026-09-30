"""Milestone 2 result tables and t-SNE helpers."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import MASK_ORDER
from ..utils import load_json

HEAD_ORDER = {"zeroshot": 0, "image": 1, "text": 2, "late": 3, "concat": 4, "gated": 5, "xattn": 6}
CLIP_COLUMNS = ["run", "head", "text_mask", "modality_dropout", "train_frac", "acc", "top5", "macro_f1", "n"]


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame({
        "h": df["head"].map(HEAD_ORDER).fillna(99), "m": df["text_mask"].map(MASK_ORDER).fillna(9),
        "md": pd.to_numeric(df["modality_dropout"], errors="coerce").fillna(-1.0),
        "f": -pd.to_numeric(df["train_frac"], errors="coerce").fillna(1.0), "r": df["run"],
    })
    order = keys.sort_values(["h", "m", "md", "f", "r"]).index
    return df.loc[order].reset_index(drop=True)


def collect_clip_results(runs_dir: str | Path) -> pd.DataFrame:
    rows = [load_json(p) for p in sorted(Path(runs_dir).glob("*/metrics_test.json"))]
    if not rows:
        return pd.DataFrame(columns=CLIP_COLUMNS)
    return _sort(pd.DataFrame(rows).reindex(columns=CLIP_COLUMNS))


def _default_md(df: pd.DataFrame, default_md: float) -> pd.Series:
    md = pd.to_numeric(df["modality_dropout"], errors="coerce")
    return md.isna() | np.isclose(md.fillna(-1.0), default_md)


def main_table(df: pd.DataFrame, default_md: float) -> pd.DataFrame:
    keep = np.isclose(pd.to_numeric(df["train_frac"]), 1.0) & _default_md(df, default_md)
    return df[keep].reset_index(drop=True)


def collect_robust(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir).glob("*/metrics_robust.json")):
        meta_path = p.parent / "metrics_test.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        for r in load_json(p):
            rows.append({**{k: meta.get(k) for k in CLIP_COLUMNS[:5]}, "condition": r["condition"],
                         "acc": r["acc"], "top5": r["top5"], "macro_f1": r["macro_f1"]})
    cols = CLIP_COLUMNS[:5] + ["condition", "acc", "top5", "macro_f1"]
    return pd.DataFrame(rows, columns=cols)


def missing_table(robust: pd.DataFrame, mask: str) -> pd.DataFrame:
    sub = robust[robust["head"].isin(["late", "concat", "gated", "xattn"]) & (robust["text_mask"] == mask)
                 & np.isclose(pd.to_numeric(robust["train_frac"]), 1.0)
                 & robust["condition"].isin(["full", "no_image", "no_text"])]
    if sub.empty:
        return pd.DataFrame(columns=["run", "head", "modality_dropout", "full", "no_image", "no_text"])
    wide = sub.pivot_table(index="run", columns="condition", values="acc").reset_index()
    meta = sub.drop_duplicates("run").set_index("run")
    wide["head"] = wide["run"].map(meta["head"])
    wide["modality_dropout"] = wide["run"].map(meta["modality_dropout"])
    wide["text_mask"], wide["train_frac"] = mask, 1.0
    out = _sort(wide)
    return out.reindex(columns=["run", "head", "modality_dropout", "full", "no_image", "no_text"])


def table_to_markdown(df: pd.DataFrame, pct_cols: Sequence[str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif c in pct_cols:
                cells.append(f"{float(v) * 100:.2f}")
            else:
                cells.append(f"{v:g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sample_for_tsne(labels: np.ndarray, n_classes: int = 20, per_class: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen = np.sort(rng.choice(classes, size=min(n_classes, len(classes)), replace=False))
    idx = []
    for c in chosen:
        pool = np.flatnonzero(labels == c)
        idx.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def tsne_2d(x: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE

    x = np.asarray(x, dtype=np.float32)
    perplexity = float(min(perplexity, max(2.0, (len(x) - 1) / 3.0)))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(x)


def plot_tsne(emb: np.ndarray, labels: np.ndarray, classes: Sequence[str], title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(i), label=str(classes[int(c)]).replace("_", " "))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


SIG_COLUMNS = ["run_a", "run_b", "n", "acc_a", "acc_b", "diff", "diff_lo", "diff_hi", "only_a_correct",
               "only_b_correct", "mcnemar_p"]


def add_acc_ci(table: pd.DataFrame, runs_dir: str | Path, n_boot: int, alpha: float, seed: int) -> pd.DataFrame:
    from ..late_fusion import load_preds
    from ..stats import bootstrap_ci, correct_vector

    out = table.copy()
    lo, hi = [], []
    for run in out["run"]:
        try:
            c = correct_vector(load_preds(Path(runs_dir) / run, "test"))
        except FileNotFoundError:
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue
        a, b = bootstrap_ci(c, n_boot, alpha, seed)
        lo.append(a)
        hi.append(b)
    out.insert(out.columns.get_loc("acc") + 1, "acc_lo", lo)
    out.insert(out.columns.get_loc("acc_lo") + 1, "acc_hi", hi)
    return out


def significance_table(work_dir: str | Path, pairs: Sequence[Sequence[str]], n_boot: int, alpha: float,
                       seed: int) -> pd.DataFrame:
    from ..stats import compare_runs

    rows = []
    for run_a, run_b in pairs:
        try:
            res = compare_runs(Path(work_dir) / run_a, Path(work_dir) / run_b, n_boot, alpha, seed)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {run_a} vs {run_b}: {e}")
            continue
        rows.append({"run_a": run_a, "run_b": run_b, **res})
    return pd.DataFrame(rows, columns=SIG_COLUMNS)

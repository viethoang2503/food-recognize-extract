"""Result tables, leakage table, run comparison and plots used by the notebook and summarize.py."""
from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data.text_utils import MASK_MODES, leakage_stats
from .late_fusion import check_aligned
from .metrics import per_class_accuracy
from .utils import load_json

MODALITY_ORDER = {"tfidf": 0, "text": 1, "image": 2, "late": 3, "early": 4}
MASK_ORDER = {"-": 0, "none": 1, "exact": 2, "strict": 3}
RESULT_COLUMNS = ["run", "modality", "text_mask", "acc", "top5", "macro_f1", "n"]


def collect_results(runs_dir: str | Path) -> pd.DataFrame:
    rows = [load_json(p) for p in sorted(Path(runs_dir).glob("*/metrics_test.json"))]
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    key = df["modality"].map(MODALITY_ORDER).fillna(99) * 10 + df["text_mask"].map(MASK_ORDER).fillna(9)
    return df.assign(_key=key).sort_values(["_key", "run"]).drop(columns="_key").reset_index(drop=True)


def results_to_markdown(df: pd.DataFrame) -> str:
    lines = ["| run | modality | text_mask | acc (%) | top-5 (%) | macro-F1 (%) |", "|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.run} | {r.modality} | {r.text_mask} | {r.acc * 100:.2f} | "
                     f"{r.top5 * 100:.2f} | {r.macro_f1 * 100:.2f} |")
    return "\n".join(lines)


def leakage_table(df: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    texts, labels = df["text"].tolist(), df["label"].tolist()
    return pd.DataFrame([leakage_stats(texts, labels, classes, mode) for mode in MASK_MODES])


def compare_predictions(a: dict, b: dict) -> pd.DataFrame:
    check_aligned(a, b)
    pred_a, pred_b = a["logits"].argmax(axis=1), b["logits"].argmax(axis=1)
    labels = a["labels"].astype(int)
    return pd.DataFrame({"id": a["ids"], "label": labels, "pred_a": pred_a, "pred_b": pred_b,
                         "correct_a": pred_a == labels, "correct_b": pred_b == labels})


def per_class_gain(a: dict, b: dict, classes: Sequence[str]) -> pd.DataFrame:
    cmp = compare_predictions(a, b)
    labels = cmp["label"].to_numpy()
    acc_a = per_class_accuracy(cmp["pred_a"].to_numpy(), labels, classes).rename(columns={"acc": "acc_a"})
    acc_b = per_class_accuracy(cmp["pred_b"].to_numpy(), labels, classes)[["acc"]].rename(columns={"acc": "acc_b"})
    out = pd.concat([acc_a, acc_b], axis=1)
    out["gain"] = out["acc_b"] - out["acc_a"]
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def plot_results_bar(results: pd.DataFrame, metric: str = "acc"):
    import matplotlib.pyplot as plt

    colors = {"-": "#999999", "none": "#1f77b4", "exact": "#2ca02c", "strict": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(results["run"], results[metric] * 100, color=[colors.get(m, "#333333") for m in results["text_mask"]])
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Test {metric} theo run (xanh: text gốc, cam: text đã che, xám: không dùng text)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_weight_curve(curves: dict[str, list[dict]]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        ax.plot([r["w"] for r in curve], [r["acc"] * 100 for r in curve], marker="o", ms=3, label=name)
    ax.set_xlabel("w (trọng số của ảnh)")
    ax.set_ylabel("val acc (%)")
    ax.set_title("Late fusion: accuracy trên val theo w")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, title: str = ""):
    import matplotlib.pyplot as plt

    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("thật")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_examples(rows: pd.DataFrame, data_root: str | Path, *, text_col: str = "text",
                  title_cols: Sequence[str] = ("label",), n_cols: int = 4, max_text_chars: int = 160):
    import matplotlib.pyplot as plt
    from PIL import Image

    if len(rows) == 0:
        raise ValueError("no rows to plot")
    n_rows = math.ceil(len(rows) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        with Image.open(Path(data_root) / row["image_path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(" | ".join(str(row[c]) for c in title_cols), fontsize=9)
        snippet = textwrap.fill(str(row[text_col])[:max_text_chars], 45)
        ax.text(0.5, -0.02, snippet, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.tight_layout()
    return fig

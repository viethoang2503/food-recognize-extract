"""Figures for the final report. Numbers are copied from the result files on Google Drive
(results/summary.md, clip/results/main.md, clip/results/missing.md).

Run: .venv/bin/python docs/report/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", facecolor=SURFACE, bbox_inches="tight", dpi=200)
    plt.close(fig)


# 1. Main comparison (strict mask), horizontal bars.
rows = [  # bottom -> top
    ("Text only: CLIP + MLP", "Text only", 38.22),
    ("Text only: DistilBERT", "Text only", 41.42),
    ("Text only: TF-IDF + LR", "Text only", 43.69),
    ("Image only: ResNet-50", "Image only", 65.56),
    ("Late fusion: ResNet-50 + DistilBERT", "Image + text", 75.63),
    ("Early fusion: ResNet-50 + DistilBERT", "Image + text", 77.65),
    ("Image only: CLIP + MLP", "Image only", 79.06),
    ("Late fusion: CLIP", "Image + text", 83.73),
    ("Gated fusion: CLIP", "Image + text", 84.94),
    ("Concat fusion: CLIP", "Image + text", 85.08),
    ("Cross-attention fusion: CLIP", "Image + text", 85.45),
]
colors = {"Text only": ORANGE, "Image only": BLUE, "Image + text": AQUA}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
y = np.arange(len(rows))
ax.barh(y, [r[2] for r in rows], height=0.62, color=[colors[r[1]] for r in rows])
for i, (_, _, acc) in enumerate(rows):
    ax.text(acc + 0.8, i, f"{acc:.2f}", va="center", fontsize=9.5, color=INK,
            fontweight="bold" if i == len(rows) - 1 else "normal")
ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Test accuracy (%), strict text mask, n = 22,712")
ax.xaxis.grid(True, color=GRID)
style(ax)
ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=9, mfc=c, mec="none", label=k) for k, c in colors.items()],
          loc="lower right", frameon=False, fontsize=9.5)
save(fig, "fig_main_results")

# 2. Effect of label leakage: none vs strict.
pairs = [("Text: CLIP", 86.35, 38.22), ("Text: TF-IDF", 86.47, 43.69), ("Text: DistilBERT", 86.43, 41.42),
         ("Late fusion: CLIP", 94.81, 83.73), ("Concat: CLIP", 94.70, 85.08),
         ("Cross-attention: CLIP", 95.14, 85.45), ("Image only: CLIP", 79.06, 79.06)]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
yy = np.arange(len(pairs))[::-1]
for yi, (name, none, strict) in zip(yy, pairs):
    ax.plot([strict, none], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter([none], [yi], s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter([strict], [yi], s=60, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    if none != strict:
        ax.text(none + 1.2, yi, f"{none:.1f}", va="center", fontsize=9, color=INK2)
        ax.text(strict - 1.2, yi, f"{strict:.1f}", va="center", ha="right", fontsize=9, color=INK2)
    else:
        ax.scatter([none], [yi], s=60, color=MUTED, zorder=4, edgecolor=SURFACE, linewidth=1.5)
        ax.text(none + 1.2, yi, f"{none:.1f} (uses no text)", va="center", fontsize=9, color=INK2)
ax.set_yticks(yy, [p[0] for p in pairs], fontsize=9.5)
ax.set_xlim(25, 105)
ax.set_xlabel("Test accuracy (%)")
ax.xaxis.grid(True, color=GRID)
style(ax)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=8, mfc=BLUE, mec="none", label="none (dish names kept)"),
                   Line2D([0], [0], marker="o", ls="", ms=8, mfc=ORANGE, mec="none", label="strict (dish names masked)")],
          loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 1.0), ncol=2)
save(fig, "fig_leakage")

# 3. Missing modality at test time (strict).
heads = ["Late", "Concat", "Gated", "Cross-attn"]
full = [83.73, 85.08, 84.94, 85.45]
no_text = [79.06, 77.78, 78.17, 77.70]
no_image = [38.22, 33.82, 33.76, 34.96]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
x = np.arange(len(heads))
w = 0.26
for k, (vals, lab, col) in enumerate([(full, "Both modalities", AQUA), (no_text, "Text removed", BLUE),
                                      (no_image, "Image removed", ORANGE)]):
    bars = ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=col, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=INK)
ax.axhline(79.06, color=BLUE, lw=1, ls="--")
ax.axhline(38.22, color=ORANGE, lw=1, ls="--")
ax.text(3.45, 79.06, "image-only\nhead 79.1", fontsize=8, color=INK2, ha="left", va="center")
ax.text(3.45, 38.22, "text-only\nhead 38.2", fontsize=8, color=INK2, ha="left", va="center")
ax.set_xlim(-0.5, 4.05)
ax.set_xticks(x, heads)
ax.set_ylim(0, 100)
ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID)
style(ax)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0))
save(fig, "fig_missing")

print("wrote", sorted(p.name for p in OUT.iterdir()))

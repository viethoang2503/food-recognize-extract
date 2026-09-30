import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from foodmm.clip.report import (  # noqa: E402
    collect_clip_results, collect_robust, main_table, missing_table, plot_tsne, sample_for_tsne, table_to_markdown,
    tsne_2d,
)
from foodmm.utils import save_json  # noqa: E402
from helpers import run_script  # noqa: E402

RUNS = [  # run, head, mask, md, frac, acc
    ("zeroshot", "zeroshot", "-", None, 1.0, 0.5), ("image", "image", "-", None, 1.0, 0.6),
    ("text_strict", "text", "strict", None, 1.0, 0.4), ("late_strict", "late", "strict", None, 1.0, 0.65),
    ("xattn_strict", "xattn", "strict", 0.1, 1.0, 0.7), ("xattn_strict_md0.3", "xattn", "strict", 0.3, 1.0, 0.69),
    ("xattn_strict_frac0.5", "xattn", "strict", 0.1, 0.5, 0.6), ("image_frac0.5", "image", "-", None, 0.5, 0.5),
]


def _write(runs_dir):
    for run, head, mask, md, frac, acc in RUNS:
        save_json({"run": run, "modality": head, "head": head, "text_mask": mask, "modality_dropout": md,
                   "train_frac": frac, "acc": acc, "top5": 0.9, "macro_f1": acc, "n": 10}, runs_dir / run / "metrics_test.json")
        if head == "zeroshot":
            continue
        conds = [("full", acc)]
        if head in ("xattn", "late"):
            conds += [("no_image", 0.3), ("no_text", acc - 0.05)]
        save_json([{"condition": c, "acc": a, "top5": 0.9, "macro_f1": a, "n": 10} for c, a in conds],
                  runs_dir / run / "metrics_robust.json")


def test_tables(tmp_path):
    _write(tmp_path)
    df = collect_clip_results(tmp_path)
    assert df["run"].tolist()[:4] == ["zeroshot", "image", "image_frac0.5", "text_strict"]
    main = main_table(df, 0.1)
    assert main["run"].tolist() == ["zeroshot", "image", "text_strict", "late_strict", "xattn_strict"]
    miss = missing_table(collect_robust(tmp_path), "strict")
    assert miss["run"].tolist() == ["late_strict", "xattn_strict", "xattn_strict_md0.3"]
    assert miss.set_index("run").loc["xattn_strict", "no_image"] == 0.3
    md = table_to_markdown(main, ["acc", "top5", "macro_f1"])
    assert md.splitlines()[0].startswith("| run |") and "| 70.00 |" in md


def test_tsne():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(25), 8)
    idx = sample_for_tsne(labels, n_classes=20, per_class=5, seed=0)
    assert len(idx) == 100 and len(np.unique(labels[idx])) == 20
    emb = tsne_2d(rng.normal(size=(len(idx), 6)), seed=0)
    assert emb.shape == (100, 2)
    assert isinstance(plot_tsne(emb, labels[idx], [f"c{i}" for i in range(25)], "t"), Figure)
    plt.close("all")


def test_summarize_clip_script(tmp_path):
    work = tmp_path / "work"
    res = run_script("summarize_clip.py", "--set", f"paths.work_dir={work}", check=False)
    assert res.returncode != 0 and "No finished runs" in res.stderr
    _write(work / "clip" / "runs")
    run_script("summarize_clip.py", "--set", f"paths.work_dir={work}")
    out = work / "clip" / "results"
    for name in ("main.csv", "main.md", "missing.csv", "missing.md", "significance.csv", "significance.md"):
        assert (out / name).exists(), name


def test_acc_ci_and_significance(tmp_path, capsys):
    from foodmm.clip.report import SIG_COLUMNS, add_acc_ci, significance_table

    runs = tmp_path / "clip" / "runs"
    _write(runs)
    labels = np.arange(10) % 3
    for run, n_right in (("image", 6), ("xattn_strict", 9)):
        logits = np.zeros((10, 3), dtype=np.float32)
        pred = labels.copy()
        pred[n_right:] = (labels[n_right:] + 1) % 3
        logits[np.arange(10), pred] = 1.0
        np.savez(runs / run / "preds_test.npz", logits=logits, labels=labels, ids=np.array([f"s{i}" for i in range(10)]))
    main = add_acc_ci(main_table(collect_clip_results(runs), 0.1), runs, n_boot=100, alpha=0.05, seed=0)
    row = main.set_index("run").loc["image"]
    assert row["acc_lo"] <= 0.6 <= row["acc_hi"]
    assert np.isnan(main.set_index("run").loc["text_strict", "acc_lo"])  # no preds_test.npz
    sig = significance_table(tmp_path, [["clip/runs/image", "clip/runs/xattn_strict"],
                                        ["runs/image", "runs/early_strict"]], n_boot=100, alpha=0.05, seed=0)
    assert list(sig.columns) == SIG_COLUMNS and len(sig) == 1
    r = sig.iloc[0]
    assert r["run_a"] == "clip/runs/image" and r["diff"] == pytest.approx(0.3) and r["only_b_correct"] == 3
    assert "Skipping runs/image vs runs/early_strict" in capsys.readouterr().out

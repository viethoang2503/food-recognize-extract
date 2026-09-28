import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from foodmm.analysis import (  # noqa: E402
    collect_results, compare_predictions, leakage_table, per_class_gain, plot_confusion,
    plot_examples, plot_results_bar, plot_weight_curve, results_to_markdown,
)
from foodmm.data.prepare import build_manifest  # noqa: E402
from foodmm.utils import save_json  # noqa: E402
from helpers import CLASSES, make_fake_dataset  # noqa: E402


def _write_runs(runs):
    for run, modality, mask, acc in [("early_none", "early", "none", 0.9), ("image", "image", "-", 0.6),
                                     ("tfidf_strict", "tfidf", "strict", 0.3), ("text_none", "text", "none", 0.8),
                                     ("tfidf_none", "tfidf", "none", 0.85)]:
        save_json({"run": run, "modality": modality, "text_mask": mask, "acc": acc, "top5": 0.95,
                   "macro_f1": acc - 0.01, "n": 10}, runs / run / "metrics_test.json")


def test_collect_results_orders_runs(tmp_path):
    _write_runs(tmp_path)
    df = collect_results(tmp_path)
    assert df["run"].tolist() == ["tfidf_none", "tfidf_strict", "text_none", "image", "early_none"]


def test_collect_results_empty(tmp_path):
    assert collect_results(tmp_path).empty


def test_results_to_markdown(tmp_path):
    _write_runs(tmp_path)
    md = results_to_markdown(collect_results(tmp_path))
    lines = md.splitlines()
    assert lines[0] == "| run | modality | text_mask | acc (%) | top-5 (%) | macro-F1 (%) |"
    assert "| tfidf_none | tfidf | none | 85.00 | 95.00 | 84.00 |" in lines
    assert len(lines) == 2 + 5


def test_leakage_table(tmp_path):
    root = make_fake_dataset(tmp_path / "ds")
    df, _, _ = build_manifest(root, val_ratio=0.3, seed=0)
    table = leakage_table(df, CLASSES)
    assert table["mode"].tolist() == ["none", "exact", "strict"]
    assert table.loc[0, "own_label_rate"] == 1.0
    assert table.loc[2, "own_word_rate"] == 0.0


A = {"ids": np.array(["x", "y", "z"]), "labels": np.array([0, 1, 1]),
     "logits": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])}
B = {"ids": np.array(["x", "y", "z"]), "labels": np.array([0, 1, 1]),
     "logits": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])}


def test_compare_predictions_and_gain():
    cmp = compare_predictions(A, B)
    assert cmp["correct_a"].tolist() == [True, False, True]
    assert cmp["correct_b"].tolist() == [True, True, True]
    gain = per_class_gain(A, B, ["a", "b"])
    assert gain.iloc[0]["class"] == "b"
    assert gain.iloc[0]["gain"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compare_predictions(A, {**B, "ids": np.array(["z", "y", "x"])})


def test_plots_return_figures(tmp_path):
    _write_runs(tmp_path / "runs")
    root = make_fake_dataset(tmp_path / "ds")
    df, _, _ = build_manifest(root, val_ratio=0.3, seed=0)
    figs = [
        plot_results_bar(collect_results(tmp_path / "runs")),
        plot_weight_curve({"none": [{"w": 0.0, "acc": 0.5}, {"w": 1.0, "acc": 0.7}]}),
        plot_confusion(np.eye(3, dtype=int), title="x"),
        plot_examples(df.head(2), root, text_col="text", title_cols=["label"]),
    ]
    assert all(isinstance(f, Figure) for f in figs)
    with pytest.raises(ValueError):
        plot_examples(df.head(0), root)
    plt.close("all")

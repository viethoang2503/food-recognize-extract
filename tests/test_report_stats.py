import numpy as np
import pandas as pd
import pytest

from foodmm.data.text_utils import build_mask_pattern
from foodmm.report_stats import (
    add_bonferroni, classifier_errors, fold_accents, graded_summary, residual_leakage, run_ci_table, vlm_paired,
    vlm_vs_classifier,
)

CLASSES = ["pho", "fried_rice", "creme_brulee", "hamburger"]


def test_add_bonferroni():
    df = pd.DataFrame({"run_a": ["a", "b", "c"], "mcnemar_p": [0.01, 0.2, 1e-30]})
    out = add_bonferroni(df)
    assert out["n_comparisons"].tolist() == [3, 3, 3]
    assert out["p_bonferroni"].tolist() == pytest.approx([0.03, 0.6, 3e-30])
    assert add_bonferroni(pd.DataFrame({"mcnemar_p": [0.5, 0.6]}))["p_bonferroni"].tolist() == [1.0, 1.0]
    assert "p_bonferroni" not in df.columns  # input untouched


def test_run_ci_table(tmp_path):
    ids, labels = np.arange(100).astype(str), np.zeros(100, dtype=int)
    logits = np.zeros((100, 2), dtype=np.float32)
    logits[:70, 0] = 1.0  # 70 correct
    logits[70:, 1] = 1.0
    (tmp_path / "r").mkdir()
    np.savez(tmp_path / "r" / "preds_test.npz", logits=logits, labels=labels, ids=ids)
    t = run_ci_table(tmp_path, ["r", "missing"], n_boot=200, alpha=0.05, seed=0)
    row = t.set_index("run").loc["r"]
    assert row["acc"] == pytest.approx(0.7) and row["acc_lo"] < 0.7 < row["acc_hi"]
    assert t.set_index("run").loc["missing"].isna().all()


def test_fold_accents():
    assert fold_accents("phở bò") == "pho bo"
    assert fold_accents("crème brûlée") == "creme brulee"
    assert fold_accents("plain") == "plain"


def test_residual_leakage():
    pattern = build_mask_pattern(CLASSES, "strict")
    texts = ["best phở in town", "[MASK] with [MASK]", "crème brûlée recipe", "nothing here", "bonjour"]
    res = residual_leakage(texts, pattern, top_k=5)
    assert res["n"] == 5
    assert res["non_ascii_rate"] == pytest.approx(2 / 5)
    assert res["folded_hit_rate"] == pytest.approx(2 / 5)  # "pho" and "creme"/"brulee" appear after folding
    assert dict(res["top_tokens"])["pho"] == 1


def _rec(sid, label, dish, valid=True):
    return {"id": sid, "label": label, "valid": valid, "output": {"dish_name": dish} if valid else None}


def test_vlm_paired():
    a = [_rec("1", "pho", "Pho"), _rec("2", "fried_rice", "rice salad"), _rec("3", "pho", "x", valid=False)]
    b = [_rec("1", "pho", "beef pho"), _rec("2", "fried_rice", "Fried Rice"), _rec("3", "pho", "pho")]
    res = vlm_paired(a, b, CLASSES, threshold=0.6, n_boot=200, seed=0)
    assert res["n"] == 3 and res["acc_a"] == pytest.approx(1 / 3) and res["acc_b"] == pytest.approx(1.0)
    assert (res["only_a_correct"], res["only_b_correct"]) == (0, 2)
    assert res["valid_a"] == 2 and res["valid_b"] == 3


def _preds(ids, labels, preds, n_classes=4):
    logits = np.zeros((len(ids), n_classes), dtype=np.float32)
    logits[np.arange(len(ids)), preds] = 1.0
    return {"ids": np.asarray(ids), "labels": np.asarray(labels), "logits": logits}


def test_vlm_vs_classifier():
    recs = [_rec("1", "pho", "Pho"), _rec("2", "fried_rice", "rice salad"), _rec("3", "pho", "x", valid=False),
            _rec("9", "pho", "pho")]  # id 9 has no classifier prediction and is ignored
    clf = _preds(["3", "2", "1"], [0, 1, 0], [0, 1, 0])  # classifier right on all three, ids shuffled
    res = vlm_vs_classifier(recs, clf, CLASSES, threshold=0.6, n_boot=200, seed=0)
    assert res["n"] == 3
    assert res["acc_vlm"] == pytest.approx(1 / 3) and res["acc_classifier"] == pytest.approx(1.0)
    assert (res["only_vlm_correct"], res["only_classifier_correct"]) == (0, 2)
    assert res["diff"] == pytest.approx(2 / 3)


def test_classifier_errors():
    ids = [str(i) for i in range(8)]
    labels = [0, 0, 0, 0, 1, 1, 2, 2]
    img = _preds(ids, labels, [1, 1, 0, 0, 1, 1, 2, 0])   # class 0: 2/4, class 2: 1/2
    fus = _preds(ids, labels, [0, 1, 0, 0, 1, 1, 2, 2])   # class 0: 3/4, class 2: 2/2
    confused, gain = classifier_errors(img, fus, CLASSES, k=2)
    assert confused.iloc[0][["true", "pred", "count"]].tolist() == ["pho", "fried_rice", 1]
    g = gain.set_index("class")
    assert g.loc["pho", "gain"] == pytest.approx(0.25) and g.loc["creme_brulee", "gain"] == pytest.approx(0.5)
    assert set(gain["side"]) == {"most helped", "most hurt"}


def test_graded_summary():
    sheet = pd.DataFrame({
        "row": [0, 1, 2, 3], "id": ["1", "1", "2", "2"], "label": ["pho", "pho", "fried_rice", "fried_rice"],
        "dish_name": ["Pho", "noodle soup", "Fried Rice", "rice salad"],
        "grade_dish": [1, 1, 1, 0], "grade_ingredients": [2, 1, 2, 2], "grade_method": [1, 1, 1, 1]})
    key = pd.DataFrame({"row": [0, 1, 2, 3], "mode": ["image", "image_text", "image_text", "image"]})
    summary, paired = graded_summary(sheet, key, CLASSES, threshold=0.6, n_boot=200, seed=0)
    s = summary.set_index("mode")
    assert s.loc["image", "n"] == 2
    assert s.loc["image", "grade_dish"] == pytest.approx(0.5)
    assert s.loc["image", "label_acc"] == pytest.approx(0.5)  # "Pho" maps to pho; "rice salad" does not
    assert s.loc["image_text", "label_acc"] == pytest.approx(0.5)  # "noodle soup" unmapped, "Fried Rice" ok
    assert {"grade_dish_lo", "grade_dish_hi", "label_acc_lo"} <= set(summary.columns)
    p = paired.set_index("grade").loc["grade_dish"]
    assert (p["only_a_correct"], p["only_b_correct"]) == (0, 1)


def test_report_stats_script(tmp_path):
    """Smoke test: the script writes every section it has inputs for and skips the rest."""
    import importlib.util
    import json
    from pathlib import Path

    work = tmp_path / "work"
    (work / "clip" / "results").mkdir(parents=True)
    pd.DataFrame({"run_a": ["a"], "run_b": ["b"], "mcnemar_p": [0.02]}).to_csv(
        work / "clip" / "results" / "significance.csv", index=False)
    (work / "data").mkdir()
    (work / "data" / "classes.json").write_text(json.dumps(CLASSES))
    pd.DataFrame({"id": ["1", "2"], "split": ["test", "train"], "text_strict": ["best phở", "[MASK]"],
                  "label_idx": [0, 1]}).to_csv(work / "data" / "manifest.csv", index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "report_stats.py"
    spec = importlib.util.spec_from_file_location("report_stats_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--set", f"paths.work_dir={work}", f"paths.data_root={tmp_path}"]) == 0
    out = work / "results" / "report_stats"
    sig = pd.read_csv(out / "significance_bonferroni.csv")
    assert sig["p_bonferroni"].tolist() == [0.02]
    leak = pd.read_csv(out / "leakage_audit.csv")
    assert leak.set_index("split").loc["test", "folded_hit_rate"] == 1.0
    assert (out / "m1_ci.md").exists() and not (out / "vlm_paired.csv").exists()

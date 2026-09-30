import math

import numpy as np
import pytest

from foodmm.stats import aligned_correct, bootstrap_ci, compare_runs, correct_vector, mcnemar


def _save(run_dir, ids, labels, preds):
    run_dir.mkdir(parents=True)
    logits = np.zeros((len(ids), 3), dtype=np.float32)
    logits[np.arange(len(ids)), preds] = 1.0
    np.savez(run_dir / "preds_test.npz", logits=logits, labels=np.asarray(labels), ids=np.asarray(ids))


def test_correct_vector():
    preds = {"logits": np.array([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]]), "labels": np.array([0, 0, 0])}
    assert correct_vector(preds).tolist() == [True, False, True]


def test_bootstrap_ci():
    assert bootstrap_ci(np.ones(50), n_boot=100) == (1.0, 1.0)
    lo, hi = bootstrap_ci(np.arange(1000) % 2, n_boot=500, seed=0)
    assert lo < 0.5 < hi and hi - lo < 0.08
    assert bootstrap_ci(np.arange(1000) % 2, n_boot=500, seed=0) == (lo, hi)  # deterministic
    assert all(math.isnan(x) for x in bootstrap_ci(np.array([])))


def test_mcnemar():
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 1, 1], dtype=bool)
    assert mcnemar(a, b) == (2, 1, 1.0)
    only_b = np.zeros(10, dtype=bool), np.ones(10, dtype=bool)
    assert mcnemar(*only_b) == (10, 0, pytest.approx(2 / 1024))
    a = np.array([False] * 60 + [True] * 30 + [True] * 100)
    b = np.array([True] * 60 + [False] * 30 + [True] * 100)
    n01, n10, p = mcnemar(a, b)
    assert (n01, n10) == (60, 30) and p == pytest.approx(math.erfc(math.sqrt((29 ** 2 / 90) / 2)))
    assert mcnemar(a, a)[2] == 1.0


def test_compare_runs_aligns_on_ids(tmp_path):
    _save(tmp_path / "a", ["x", "y", "z", "w"], [0, 1, 2, 0], [0, 0, 0, 0])      # correct: x, w
    _save(tmp_path / "b", ["w", "z", "y", "q"], [0, 2, 1, 1], [0, 2, 1, 0])      # correct: w, z, y (q not in a)
    a, b = aligned_correct(tmp_path / "a", tmp_path / "b")
    assert len(a) == 3 and a.sum() == 1 and b.sum() == 3  # common ids: w, y, z
    res = compare_runs(tmp_path / "a", tmp_path / "b", n_boot=200, seed=0)
    assert res["n"] == 3 and res["diff"] == pytest.approx(2 / 3)
    assert res["only_b_correct"] == 2 and res["only_a_correct"] == 0
    assert res["diff_lo"] <= res["diff"] <= res["diff_hi"] and 0 < res["mcnemar_p"] <= 1
    _save(tmp_path / "c", ["other"], [0], [0])
    with pytest.raises(ValueError, match="no test ids in common"):
        aligned_correct(tmp_path / "a", tmp_path / "c")

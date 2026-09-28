import numpy as np
import pytest

from foodmm.metrics import compute_metrics, per_class_accuracy, softmax, top_confused_pairs


def test_compute_metrics_basic():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.1, 0.6, 0.3]])
    labels = np.array([0, 1, 2, 2])
    m = compute_metrics(scores, labels, k=2)
    assert m["acc"] == pytest.approx(0.75)
    assert m["top5"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    assert m["n"] == 4


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[1, 0] == pytest.approx(1.0)


def test_per_class_accuracy_and_confused_pairs():
    preds, labels = np.array([1, 1, 0, 2]), np.array([0, 0, 0, 2])
    pc = per_class_accuracy(preds, labels, ["a", "b", "c"])
    assert pc["acc"].tolist()[0] == pytest.approx(1 / 3)
    assert pc["n"].tolist() == [3, 0, 1]
    pairs = top_confused_pairs(preds, labels, ["a", "b", "c"], k=3)
    assert pairs.iloc[0].to_dict() == {"true": "a", "pred": "b", "count": 2}
    assert len(pairs) == 1

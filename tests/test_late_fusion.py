import numpy as np
import pytest

from foodmm.late_fusion import check_aligned, combine, load_preds, search_weight


def test_combine():
    a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(combine(a, b, 0.3), [[0.3, 0.7]])


def test_search_weight_prefers_better_modality():
    labels = np.array([0, 1, 0, 1])
    p_img = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])  # all correct
    p_txt = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])  # all wrong
    w, curve = search_weight(p_img, p_txt, labels, step=0.25)
    assert w == 1.0
    assert [r["w"] for r in curve] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert curve[0]["acc"] == 0.0 and curve[-1]["acc"] == 1.0


def test_search_weight_rejects_bad_step():
    with pytest.raises(ValueError):
        search_weight(np.ones((1, 2)), np.ones((1, 2)), np.array([0]), step=0.0)


def test_check_aligned():
    check_aligned({"ids": np.array(["a", "b"])}, {"ids": np.array(["a", "b"])})
    with pytest.raises(ValueError, match="not aligned"):
        check_aligned({"ids": np.array(["a", "b"])}, {"ids": np.array(["b", "a"])})


def test_load_preds(tmp_path):
    np.savez(tmp_path / "preds_val.npz", logits=np.zeros((2, 3)), labels=np.array([0, 1]), ids=np.array(["x", "y"]))
    p = load_preds(tmp_path, "val")
    assert p["logits"].shape == (2, 3) and p["ids"].tolist() == ["x", "y"]
    with pytest.raises(FileNotFoundError):
        load_preds(tmp_path, "test")

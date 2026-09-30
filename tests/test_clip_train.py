import json

import numpy as np
import pytest

from foodmm.clip.features import FeatureWriter, MissingFeatures, clip_paths, image_set_name, text_set_name
from foodmm.clip.train_heads import run_late, run_name_for, stratified_fraction, train_head_run
from foodmm.config import load_config, set_by_path, work_paths
from foodmm.data.prepare import load_manifest
from helpers import clip_smoke_overrides, make_fake_dataset, run_script

E, DI, DT, N_TOK = 8, 6, 5, 16


def _write(cfg, name, ids, arrays):
    w = FeatureWriter(clip_paths(cfg)["features"] / name, ids, shard_size=64)
    for i, idx in w.pending_shards():
        w.write_shard(i, {k: v[idx] for k, v in arrays.items()})
    w.finalize()


def _fake_features(cfg, df, rng):
    """Class-dependent pooled features so heads can learn; tokens are noise."""
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        ids, y = sub["id"].tolist(), sub["label_idx"].to_numpy()
        n = len(ids)
        signal = np.zeros((n, E), dtype=np.float32)
        signal[np.arange(n), y] = 3.0
        img = {"pooled": (signal + rng.normal(0, 0.5, (n, E))).astype(np.float16),
               "tokens": rng.normal(0, 1, (n, N_TOK, DI)).astype(np.float16)}
        mask = np.zeros((n, N_TOK), dtype=bool)
        mask[:, :4] = True
        txt = {"pooled": (signal + rng.normal(0, 0.5, (n, E))).astype(np.float16),
               "tokens": rng.normal(0, 1, (n, N_TOK, DT)).astype(np.float16), "token_mask": mask}
        _write(cfg, image_set_name(split), ids, img)
        for m in ("none", "strict"):
            _write(cfg, text_set_name(m, split), ids, txt)
        if split == "test":
            _write(cfg, image_set_name("test", "blur2"), ids, img)
            _write(cfg, image_set_name("test", "noise0.1"), ids, img)
            _write(cfg, text_set_name("strict", "test", "drop0.5"), ids, txt)


@pytest.fixture()
def env(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work) + ["head.epochs=20", "head.lr=0.01", "head.patience=20"]
    run_script("prepare_data.py", "--set", *sets)
    cfg = load_config(overrides=sets)
    _fake_features(cfg, load_manifest(work_paths(cfg)["manifest"]), np.random.default_rng(0))
    return cfg


def _with(cfg, **kv):
    import copy

    c = copy.deepcopy(cfg)
    for k, v in kv.items():
        set_by_path(c, k.replace("__", "."), v)
    return c


def test_run_name_for():
    cfg = load_config()
    assert run_name_for(_with(cfg, head__name="image", data__text_mask="strict")) == "image"
    assert run_name_for(_with(cfg, head__name="text", data__text_mask="none")) == "text_none"
    assert run_name_for(_with(cfg, head__name="xattn", data__text_mask="strict")) == "xattn_strict"
    assert run_name_for(_with(cfg, head__name="gated", head__modality_dropout=0.0)) == "gated_none_md0"
    assert run_name_for(_with(cfg, head__name="concat", data__text_mask="strict", head__train_frac=0.25)) == \
        "concat_strict_frac0.25"
    assert run_name_for(_with(cfg, head__name="text", head__modality_dropout=0.3)) == "text_none"


def test_stratified_fraction():
    labels = np.repeat(np.arange(4), [10, 10, 10, 1])
    idx = stratified_fraction(labels, 0.3, seed=0)
    counts = np.bincount(labels[idx], minlength=4)
    assert counts.tolist() == [3, 3, 3, 1]
    assert np.array_equal(idx, stratified_fraction(labels, 0.3, seed=0)) and np.all(np.diff(idx) > 0)


def test_train_heads_robustness_and_late(env):
    cfg = env
    runs = clip_paths(cfg)["runs"]
    m_img = train_head_run(_with(cfg, head__name="image"))
    assert m_img["run"] == "image" and m_img["text_mask"] == "-" and m_img["acc"] >= 0.5
    rob = json.loads((runs / "image" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob] == ["full", "blur2", "noise0.1"]

    for mask in ("none", "strict"):
        train_head_run(_with(cfg, head__name="text", data__text_mask=mask))
    rob_none = json.loads((runs / "text_none" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob_none] == ["full"]  # word drop only exists for the main mask

    m_x = train_head_run(_with(cfg, head__name="xattn", data__text_mask="strict"))
    assert m_x["modality_dropout"] == 0.1 and m_x["train_frac"] == 1.0
    rob_x = json.loads((runs / "xattn_strict" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob_x] == ["full", "no_image", "no_text", "blur2", "noise0.1", "drop0.5"]
    with np.load(runs / "xattn_strict" / "preds_test.npz") as z:
        assert z["logits"].shape == (12, 3) and z["features"].dtype == np.float16
    with np.load(runs / "xattn_strict" / "preds_robust.npz") as z:
        assert {"full", "no_image", "drop0.5", "ids", "labels"} <= set(z.files)

    m_g = train_head_run(_with(cfg, head__name="gated", data__text_mask="strict", head__train_frac=0.5))
    assert m_g["run"] == "gated_strict_frac0.5" and 0.0 <= m_g["mean_gate"] <= 1.0

    late = run_late(cfg, "strict")
    assert late["run"] == "late_strict" and 0.0 <= late["w"] <= 1.0
    rob_late = json.loads((runs / "late_strict" / "metrics_robust.json").read_text())
    assert {r["condition"] for r in rob_late} == {"full", "no_image", "no_text", "blur2", "noise0.1", "drop0.5"}
    assert train_head_run(_with(cfg, head__name="image"))["acc"] == m_img["acc"]  # already finished


def test_missing_features_and_runs(env):
    with pytest.raises(MissingFeatures):
        train_head_run(_with(env, head__name="text", data__text_mask="exact"))
    with pytest.raises(SystemExit, match="image"):
        run_late(env, "none")

import numpy as np
import pytest

from foodmm.clip.features import (
    FeatureWriter, MissingFeatures, check_ids, clip_paths, image_set_name, is_done, load_feature_set,
    text_set_name,
)
from foodmm.config import load_config


def test_names_and_paths(tmp_path):
    cfg = load_config(overrides=[f"paths.work_dir={tmp_path}"])
    p = clip_paths(cfg)
    assert p["features"] == tmp_path / "clip" / "features" / "vitb16"
    assert p["runs"] == tmp_path / "clip" / "runs" and p["results"] == tmp_path / "clip" / "results"
    assert image_set_name("train") == "image_train"
    assert image_set_name("test", "blur2") == "image_test_blur2"
    assert text_set_name("strict", "val") == "text_strict_val"
    assert text_set_name("strict", "test", "drop0.5") == "text_strict_test_drop0.5"


def _arrays(idx):
    return {"pooled": np.full((len(idx), 4), idx[:, None], dtype=np.float16),
            "tokens": np.zeros((len(idx), 2, 3), dtype=np.float16)}


def test_writer_resume_and_finalize(tmp_path):
    ids = [f"id{i}" for i in range(10)]
    out = tmp_path / "fs"
    w = FeatureWriter(out, ids, shard_size=4)
    pending = w.pending_shards()
    assert [i for i, _ in pending] == [0, 1, 2] and pending[2][1].tolist() == [8, 9]
    w.write_shard(0, _arrays(pending[0][1]))
    w2 = FeatureWriter(out, ids, shard_size=4)  # "restart"
    assert [i for i, _ in w2.pending_shards()] == [1, 2]
    for i, idx in w2.pending_shards():
        w2.write_shard(i, _arrays(idx))
    assert w2.finalize() == 10 and is_done(out) and w2.is_done()
    assert not (out / "shards").exists()
    fs = load_feature_set(out)
    assert fs["ids"].tolist() == ids
    assert fs["pooled"][:, 0].tolist() == list(range(10))
    only = load_feature_set(out, keys=["pooled"])
    assert set(only) == {"ids", "pooled"}


def test_finalize_requires_all_shards(tmp_path):
    w = FeatureWriter(tmp_path / "fs", ["a", "b", "c"], shard_size=2)
    w.write_shard(0, _arrays(np.array([0, 1])))
    with pytest.raises(RuntimeError, match="missing shards"):
        w.finalize()


def test_load_missing_raises(tmp_path):
    with pytest.raises(MissingFeatures, match="extract_clip.py"):
        load_feature_set(tmp_path / "nope")


def test_check_ids():
    check_ids(np.array(["a", "b"]), np.array(["a", "b"]), "x")
    with pytest.raises(ValueError, match="x"):
        check_ids(np.array(["b", "a"]), np.array(["a", "b"]), "x")

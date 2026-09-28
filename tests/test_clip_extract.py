import json

import numpy as np
import pytest

from foodmm.clip.extract import build_jobs
from foodmm.config import load_config
from helpers import clip_smoke_overrides, make_fake_dataset, run_script


def test_build_jobs():
    cfg = load_config()
    jobs = build_jobs(cfg, ["image", "text_strict", "corrupt"], ["train", "test"])
    names = [j.name for j in jobs]
    assert names[:4] == ["image_train", "image_test", "text_strict_train", "text_strict_test"]
    assert "image_test_blur2" in names and "text_strict_test_drop0.5" in names
    blur = next(j for j in jobs if j.name == "image_test_blur2")
    assert blur.kind == "image" and blur.split == "test" and blur.corruption == ("blur", 2.0)
    drop = next(j for j in jobs if j.name == "text_strict_test_drop0.5")
    assert drop.column == "text_strict" and drop.corruption == ("drop", 0.5)
    with pytest.raises(ValueError):
        build_jobs(cfg, ["bogus"], ["train"])


@pytest.mark.network
def test_extract_and_zero_shot_scripts(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work)
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_strict,corrupt", "--set", *sets)
    feats = work / "clip" / "features" / "vitb16"
    for name in ("image_train", "image_val", "image_test", "text_strict_test", "image_test_blur2",
                 "image_test_noise0.1", "text_strict_test_drop0.5"):
        assert (feats / name / "done.json").exists(), name
    tokens = np.load(feats / "text_strict_train" / "tokens.npy")
    assert tokens.shape[1] == 16 and np.load(feats / "text_strict_train" / "token_mask.npy").shape == tokens.shape[:2]
    assert "skip image_train" in run_script("extract_clip.py", "--parts", "image", "--set", *sets).stdout

    run_script("zero_shot_clip.py", "--set", *sets)
    m = json.loads((work / "clip" / "runs" / "zeroshot" / "metrics_test.json").read_text())
    assert m["run"] == "zeroshot" and m["n"] == 12 and 0 <= m["acc"] <= 1
    assert "already finished" in run_script("zero_shot_clip.py", "--set", *sets).stdout

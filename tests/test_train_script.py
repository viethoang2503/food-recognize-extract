import json

import numpy as np
import pytest

from foodmm.data.prepare import build_manifest
from foodmm.utils import save_json
from helpers import make_fake_dataset, run_script, smoke_overrides


@pytest.mark.network
def test_train_script_all_modalities(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work)
    df, classes, _ = build_manifest(data_root, val_ratio=0.3, seed=0)
    (work / "data").mkdir(parents=True)
    df.to_csv(work / "data" / "manifest.csv", index=False)
    save_json(classes, work / "data" / "classes.json")

    res = run_script("train.py", "--modality", "multimodal", "--set", *sets, "data.text_mask=strict", check=False)
    assert res.returncode != 0 and "image-only" in res.stderr

    run_script("train.py", "--modality", "image", "--set", *sets)
    run_script("train.py", "--modality", "text", "--set", *sets, "data.text_mask=strict")
    run_script("train.py", "--modality", "multimodal", "--set", *sets, "data.text_mask=strict")

    runs = work / "runs"
    expected = [("image", "image", "-", 512), ("text_strict", "text", "strict", None), ("early_strict", "early", "strict", 32)]
    for name, modality, mask, feat_dim in expected:
        m = json.loads((runs / name / "metrics_test.json").read_text())
        assert (m["run"], m["modality"], m["text_mask"], m["n"]) == (name, modality, mask, 12)
        assert 0.0 <= m["acc"] <= 1.0 and "best_val_acc" in m
        assert (runs / name / "config.yaml").exists() and (runs / name / "history.json").exists()
        for split, n in (("val", 9), ("test", 12)):
            with np.load(runs / name / f"preds_{split}.npz") as z:
                assert z["logits"].shape == (n, 3) and len(z["ids"]) == n
                assert z["features"].shape[0] == n and z["features"].dtype == np.float16
                if feat_dim is not None:
                    assert z["features"].shape[1] == feat_dim

    res = run_script("train.py", "--modality", "image", "--set", *sets)
    assert "already finished" in res.stdout

    res = run_script("train.py", "--modality", "multimodal", "--run_name", "early_bad", "--set", *sets,
                     "data.text_mask=none", "fusion.text_init=text_strict", check=False)
    assert res.returncode != 0 and "text_mask" in res.stderr

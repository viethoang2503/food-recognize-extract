import numpy as np
import pandas as pd

from foodmm.data.prepare import load_manifest
from helpers import make_fake_dataset, run_script, smoke_overrides


def test_vlm_scripts_with_fake_backend(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work) + ["vlm.backend=fake", "vlm.n_samples=6", "vlm.n_grading=2"]
    run_script("prepare_data.py", "--set", *sets)

    test = load_manifest(work / "data" / "manifest.csv").query("split == 'test'")
    run_dir = work / "clip" / "runs" / "xattn_strict"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "preds_test.npz", logits=np.random.default_rng(0).normal(size=(len(test), 3)).astype(np.float32),
             labels=test["label_idx"].to_numpy(), ids=test["id"].to_numpy().astype(str))

    out = run_script("vlm_extract.py", "--mode", "all", "--set", *sets).stdout
    assert out.count("n_new=6") == 3
    assert run_script("vlm_extract.py", "--mode", "image", "--set", *sets).stdout.count("n_new=0") == 1

    run_script("vlm_evaluate.py", "--set", *sets)
    out_dir = work / "vlm" / "qwen3vl4b"
    summary = pd.read_csv(out_dir / "eval_summary.csv")
    assert summary["mode"].tolist() == ["image", "text", "image_text"]
    assert (summary["valid_rate"] == 1.0).all() and summary["uses_text"].tolist() == [False, True, True]
    assert len(pd.read_csv(out_dir / "manual_grading.csv")) == 6  # 2 ids x 3 modes
    assert (out_dir / "manual_grading_key.csv").exists()
    assert (out_dir / "eval_summary.md").read_text().startswith("| mode |")
    clf = pd.read_csv(out_dir / "classifier_on_sample.csv")
    assert clf["name"].tolist() == ["image_text"] and clf["n"].iloc[0] == 6  # only xattn_strict exists here

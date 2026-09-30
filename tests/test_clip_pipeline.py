import json

import pandas as pd
import pytest

from helpers import clip_smoke_overrides, make_fake_dataset, run_script


@pytest.mark.network
def test_clip_pipeline_end_to_end(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work)
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_none,text_strict", "--set", *sets)
    out = run_script("run_clip_suite.py", "--set", *sets).stdout
    assert "zeroshot" in out and "late_strict" in out
    run_script("summarize_clip.py", "--set", *sets)
    main = pd.read_csv(work / "clip" / "results" / "main.csv")
    assert main["run"].tolist()[0] == "zeroshot" and "xattn_strict" in set(main["run"])
    missing = pd.read_csv(work / "clip" / "results" / "missing.csv")
    assert {"late_strict", "concat_strict", "gated_strict", "xattn_strict"} <= set(missing["run"])
    assert {"acc_lo", "acc_hi"} <= set(main.columns)
    sig = pd.read_csv(work / "clip" / "results" / "significance.csv")
    assert ("clip/runs/image", "clip/runs/xattn_strict") in set(zip(sig["run_a"], sig["run_b"]))
    m = json.loads((work / "clip" / "runs" / "gated_strict" / "metrics_test.json").read_text())
    assert m["head"] == "gated" and "mean_gate" in m

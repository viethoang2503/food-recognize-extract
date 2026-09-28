import json

import pandas as pd
import pytest

from helpers import make_fake_dataset, run_script, smoke_overrides

ALL_RUNS = ["tfidf_none", "tfidf_strict", "text_none", "text_strict", "image",
            "late_none", "late_strict", "early_none", "early_strict"]


@pytest.mark.network
def test_full_pipeline(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work)

    run_script("prepare_data.py", "--set", *sets)
    assert (work / "data" / "manifest.csv").exists() and (work / "data" / "stats.json").exists()
    assert "already exists" in run_script("prepare_data.py", "--set", *sets).stdout

    res = run_script("summarize.py", "--set", *sets, check=False)
    assert res.returncode != 0 and "No finished runs" in res.stderr

    for mask in ("none", "strict"):
        run_script("train_tfidf.py", "--set", *sets, f"data.text_mask={mask}")
    run_script("train.py", "--modality", "image", "--set", *sets)
    for mask in ("none", "strict"):
        run_script("train.py", "--modality", "text", "--set", *sets, f"data.text_mask={mask}")
        run_script("late_fusion.py", "--text_run", f"text_{mask}", "--set", *sets)
        run_script("train.py", "--modality", "multimodal", "--set", *sets, f"data.text_mask={mask}")
    assert "already finished" in run_script("train_tfidf.py", "--set", *sets).stdout
    assert "already finished" in run_script("late_fusion.py", "--text_run", "text_none", "--set", *sets).stdout

    run_script("summarize.py", "--set", *sets)
    summary = pd.read_csv(work / "results" / "summary.csv")
    assert summary["run"].tolist() == ALL_RUNS
    assert (work / "results" / "summary.md").read_text().startswith("| run |")

    tfidf = json.loads((work / "runs" / "tfidf_none" / "metrics_test.json").read_text())
    assert tfidf["modality"] == "tfidf" and tfidf["C"] == 1.0 and tfidf["n"] == 12
    late = json.loads((work / "runs" / "late_strict" / "metrics_test.json").read_text())
    assert late["modality"] == "late" and late["text_mask"] == "strict" and 0.0 <= late["w"] <= 1.0
    assert len(json.loads((work / "runs" / "late_strict" / "weight_curve.json").read_text())) == 21

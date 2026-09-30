import json

import pandas as pd
import pytest
from PIL import Image

from foodmm.config import load_config
from foodmm.data.prepare import load_manifest
from foodmm.utils import load_json
from foodmm.vlm.backend import FAKE_OUTPUT, FakeBackend
from foodmm.vlm.extract import extract_one, read_records, run_extraction, select_sample, vlm_dir
from helpers import CLASSES, make_fake_dataset, run_script, smoke_overrides

IMG = Image.new("RGB", (8, 8))


def test_extract_one_retry_then_valid():
    fb = FakeBackend(["not json at all", FAKE_OUTPUT])
    rec = extract_one(fb, IMG, "prompt")
    assert rec["valid"] and rec["attempts"] == 2 and rec["output"]["dish_name"] == "apple pie"
    assert "not valid" in fb.calls[1][0] and rec["error"] is None


def test_extract_one_gives_up_after_retry():
    rec = extract_one(FakeBackend(["nope"]), IMG, "prompt")
    assert not rec["valid"] and rec["attempts"] == 2 and rec["output"] is None
    assert rec["raw"] == "nope" and "no JSON" in rec["error"] and rec["latency_s"] >= 0


def test_select_sample_is_stratified_and_cached(tmp_path):
    df = pd.DataFrame({"id": [f"{c}_{i}" for c in "abc" for i in range(10)], "label": [c for c in "abc" for _ in range(10)],
                       "split": "test"})
    path = tmp_path / "sample_ids.json"
    ids = select_sample(df, 6, seed=0, path=path)
    assert len(ids) == 6 and sorted(i[0] for i in ids) == ["a", "a", "b", "b", "c", "c"]
    assert load_json(path) == ids
    assert select_sample(df, 3, seed=1, path=path) == ids  # cached file wins
    assert len(select_sample(df, 7, seed=0, path=tmp_path / "x.json")) == 7


@pytest.fixture()
def env(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work) + ["vlm.backend=fake", "vlm.n_samples=6", "vlm.log_every=2"]
    run_script("prepare_data.py", "--set", *sets)
    cfg = load_config(overrides=sets)
    return cfg, load_manifest(work / "data" / "manifest.csv")


def test_run_extraction_resumes(env):
    cfg, df = env
    fb = FakeBackend()
    res = run_extraction(cfg, "image", fb, df, CLASSES)
    assert res["n_new"] == 6 and res["n_total"] == 6 and res["n_valid"] == 6
    recs = read_records(vlm_dir(cfg) / "extract_image.jsonl")
    assert [r["mode"] for r in recs] == ["image"] * 6 and all("label" in r for r in recs)
    assert "photo" in fb.calls[0][0] and all(im is not None for im in fb.images)
    again = run_extraction(cfg, "image", FakeBackend(), df, CLASSES)
    assert again["n_new"] == 0 and again["n_total"] == 6
    txt = run_extraction(cfg, "image_text", FakeBackend(), df, CLASSES, n=2)
    assert txt["n_new"] == 2
    line = (vlm_dir(cfg) / "extract_image_text.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["output"]["cuisine"] == "American"
    tb = FakeBackend()
    only_text = run_extraction(cfg, "text", tb, df, CLASSES, n=3)
    assert only_text["n_new"] == 3 and tb.images == [None] * 3  # the text-only mode never sends the image
    assert "Identify the dish described in this text" in tb.calls[0][0]

import json
import os

import pytest
from PIL import Image

from foodmm.config import load_config
from helpers import REPO_ROOT, clip_smoke_overrides, make_fake_dataset, run_script


def test_demo_refuses_public_link_without_auth():
    env = {k: v for k, v in os.environ.items() if k not in ("DEMO_USERNAME", "DEMO_PASSWORD")}
    import subprocess
    import sys

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "demo.py"), "--share", "--dry_run"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert res.returncode != 0 and "DEMO_USERNAME" in res.stderr
    ok = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                        env={**env, "DEMO_USERNAME": "u", "DEMO_PASSWORD": "p"})
    assert ok.returncode == 0 and "dry run ok" in ok.stdout and "auth=on" in ok.stdout
    no_auth = subprocess.run(cmd + ["--no_auth"], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert no_auth.returncode == 0 and "auth=off" in no_auth.stdout


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    data_root = make_fake_dataset(tmp / "ds")
    sets = clip_smoke_overrides(data_root, tmp / "work") + ["head.epochs=1"]
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_strict", "--set", *sets)
    for head in ("image", "text", "xattn"):
        run_script("train_head.py", "--head", head, "--set", *sets, "data.text_mask=strict")
    return sets


@pytest.mark.network
def test_predictor(trained):
    from foodmm.demo.predictor import DemoPredictor

    pred = DemoPredictor(load_config(overrides=trained))
    assert pred.run_names == {"image": "image", "text": "text_strict", "fusion": "xattn_strict"}
    img = Image.new("RGB", (40, 30), (200, 150, 60))
    both = pred.predict(img, "Best apple pie recipe from grandma")
    assert set(both) == {"image", "text", "fusion", "masked_text"}
    assert len(both["fusion"]) == 3 and abs(sum(both["fusion"].values()) - 1.0) < 1e-4
    assert "apple" not in both["masked_text"] and "[MASK]" in both["masked_text"]
    assert all(" " in k or k.isalpha() for k in both["image"])  # display names, no underscores
    image_only = pred.predict(img, "   ")
    assert image_only["text"] is None and image_only["fusion"] is not None and image_only["masked_text"] == ""
    text_only = pred.predict(None, "crispy fries")
    assert text_only["image"] is None and text_only["fusion"] is not None
    with pytest.raises(ValueError):
        pred.predict(None, "")


@pytest.mark.network
def test_build_demo_examples_script(trained):
    run_script("build_demo_examples.py", "--n", "2", "--set", *trained)
    cfg = load_config(overrides=trained)
    from foodmm.demo.examples import demo_dir, load_examples

    items = load_examples(cfg)
    assert len(items) == 2 and items[0]["name"].startswith("1. ") and items[0]["vlm"] is None
    assert os.path.exists(items[0]["image_path"])
    raw = json.loads((demo_dir(cfg) / "examples.json").read_text())
    assert raw[0]["image"].startswith("examples/")  # stored relative, resolved on load


def test_demo_missing_runs(tmp_path):
    res = run_script("demo.py", "--no_auth", "--set", f"paths.work_dir={tmp_path}", check=False)
    assert res.returncode != 0 and "Missing Milestone 2 runs" in res.stderr

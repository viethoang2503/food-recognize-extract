import importlib.util
import json
from pathlib import Path

import pytest

from foodmm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("image", "text_strict", "xattn_strict")


def _script():
    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _work(tmp_path, runs=RUNS, examples=False):
    work = tmp_path / "work"
    for r in runs:
        d = work / "clip" / "runs" / r
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"weights")
        (d / "config.yaml").write_text("head: {}\n")
        (d / "preds_test.npz").write_bytes(b"not needed")
    (work / "data").mkdir(parents=True)
    (work / "data" / "classes.json").write_text(json.dumps(["pho", "pizza"]))
    if examples:
        (work / "demo" / "examples").mkdir(parents=True)
        (work / "demo" / "examples.json").write_text("[]")
        (work / "demo" / "examples" / "01.jpg").write_bytes(b"jpg")
    return load_config(overrides=[f"paths.work_dir={work}", f"paths.data_root={tmp_path}"])


def test_build_bundle(tmp_path):
    cfg = _work(tmp_path)
    out = _script().build_bundle(cfg, tmp_path / "bundle", ROOT)
    for f in ("app.py", "requirements.txt", "README.md", "configs/default.yaml", "src/foodmm/demo/predictor.py",
              "work/data/classes.json"):
        assert (out / f).exists(), f
    for r in RUNS:
        assert (out / "work/clip/runs" / r / "best.pt").read_bytes() == b"weights"
        assert (out / "work/clip/runs" / r / "config.yaml").exists()
        assert not (out / "work/clip/runs" / r / "preds_test.npz").exists()  # only what the demo loads
    assert not list(out.rglob("__pycache__"))
    assert not (out / "work/demo").exists()  # dataset images are not published unless asked


def test_build_bundle_with_examples_and_rebuild(tmp_path):
    cfg = _work(tmp_path, examples=True)
    mod = _script()
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "stale.txt").write_text("old")
    out = mod.build_bundle(cfg, tmp_path / "bundle", ROOT, with_examples=True)
    assert (out / "work/demo/examples.json").exists() and (out / "work/demo/examples/01.jpg").exists()
    assert not (out / "stale.txt").exists()


def test_build_bundle_missing_run(tmp_path):
    cfg = _work(tmp_path, runs=("image", "text_strict"))
    with pytest.raises(SystemExit, match="xattn_strict"):
        _script().build_bundle(cfg, tmp_path / "bundle", ROOT)


def test_main_dry_run(tmp_path):
    work = tmp_path / "work"
    _work(tmp_path)
    out = tmp_path / "bundle"
    rc = _script().main(["--space", "user/food", "--out", str(out), "--dry_run",
                         "--set", f"paths.work_dir={work}", f"paths.data_root={tmp_path}"])
    assert rc == 0 and (out / "app.py").exists()

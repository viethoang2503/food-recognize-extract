import json

from foodmm.clip.features import clip_paths
from foodmm.clip.suite import job_config, run_suite, suite_jobs
from foodmm.config import load_config
from helpers import run_script
from test_clip_train import env  # noqa: F401  (fixture with a fake feature store)


def test_suite_jobs_default():
    cfg = load_config()
    jobs = suite_jobs(cfg)
    assert len(jobs) == 12 and jobs[0] == ("zeroshot", [])
    assert [k for k, _ in jobs].count("late") == 2
    kind, ov = jobs[4]
    c = job_config(cfg, ov)
    assert kind == "head" and c["head"]["name"] == "concat" and c["data"]["text_mask"] == "none"
    c = job_config(cfg, ["head.modality_dropout=0.3"])
    assert c["head"]["modality_dropout"] == 0.3 and cfg["head"]["modality_dropout"] == 0.1  # base untouched


def test_run_suite_on_fake_features(env):  # noqa: F811
    cfg = job_config(env, ["head.epochs=2"])
    table = run_suite(cfg, skip_zeroshot=True)
    assert set(table["run"]) == {"image", "text_none", "text_strict", "concat_none", "concat_strict", "gated_none",
                                 "gated_strict", "xattn_none", "xattn_strict", "late_none", "late_strict"}
    rob = json.loads((clip_paths(cfg)["runs"] / "gated_strict" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob][:3] == ["full", "no_image", "no_text"]


def test_train_head_script(env, tmp_path):  # noqa: F811
    work = env["paths"]["work_dir"]
    sets = [f"paths.data_root={env['paths']['data_root']}", f"paths.work_dir={work}", "head.epochs=1",
            "head.hidden=16", "head.batch_size=16", "data.text_mask=strict",
            "clip.corruptions.blur=[2]", "clip.corruptions.noise=[0.1]", "clip.corruptions.word_drop=[0.5]"]
    run_script("train_head.py", "--head", "image", "--set", *sets)
    run_script("train_head.py", "--head", "text", "--set", *sets)
    out = run_script("train_head.py", "--head", "late", "--set", *sets).stdout
    assert '"run": "late_strict"' in out
    assert "already finished" in run_script("train_head.py", "--head", "image", "--set", *sets).stdout

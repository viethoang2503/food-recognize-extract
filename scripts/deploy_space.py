#!/usr/bin/env python
"""Build the Hugging Face Space bundle for the demo and upload it (CPU Space, no VLM).

The bundle holds space/{app.py,requirements.txt,README.md}, src/foodmm, configs/default.yaml and, under work/, the
three Milestone 2 heads the demo loads (best.pt + config.yaml) and classes.json. Dataset images (the precomputed
examples) are only added with --with_examples. Uploading needs a write token in the HF_TOKEN environment variable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from foodmm.clip.features import clip_paths  # noqa: E402
from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402

SPACE_FILES = ("app.py", "requirements.txt", "README.md")
RUN_FILES = ("best.pt", "config.yaml")


def demo_runs(cfg: dict) -> list[str]:
    d = cfg["demo"]
    return [d["image_run"], d["text_run"] or f"text_{d['text_mask']}", d["fusion_run"]]


def build_bundle(cfg: dict, out: str | Path, repo_root: str | Path = REPO, with_examples: bool = False) -> Path:
    out, repo_root = Path(out), Path(repo_root)
    runs_dir, wp = clip_paths(cfg)["runs"], work_paths(cfg)
    missing = [r for r in demo_runs(cfg) if not (runs_dir / r / "best.pt").exists()]
    if missing:
        raise SystemExit(f"Missing demo runs {missing} in {runs_dir}: run notebooks/02_milestone2.ipynb first")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in SPACE_FILES:
        shutil.copy2(repo_root / "space" / f, out / f)
    shutil.copytree(repo_root / "src" / "foodmm", out / "src" / "foodmm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "configs").mkdir()
    shutil.copy2(repo_root / "configs" / "default.yaml", out / "configs" / "default.yaml")
    for r in demo_runs(cfg):
        dst = out / "work" / "clip" / "runs" / r
        dst.mkdir(parents=True)
        for f in RUN_FILES:
            shutil.copy2(runs_dir / r / f, dst / f)
    (out / "work" / "data").mkdir(parents=True)
    shutil.copy2(wp["classes"], out / "work" / "data" / "classes.json")
    demo = wp["work_dir"] / "demo"
    if with_examples and (demo / "examples.json").exists():
        shutil.copytree(demo, out / "work" / "demo")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. <hf-username>/food-recognition")
    parser.add_argument("--out", default="space_bundle", help="local folder for the bundle (recreated)")
    parser.add_argument("--with_examples", action="store_true", help="also publish the precomputed example images")
    parser.add_argument("--dry_run", action="store_true", help="build the bundle without uploading")
    add_config_args(parser)
    args = parser.parse_args(argv)
    out = build_bundle(config_from_args(args), args.out, with_examples=args.with_examples)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Built {out} ({size:.1f} MB)")
    if args.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face write token (Colab Secrets) to upload")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.space, repo_type="space", commit_message="Deploy demo")
    print(f"Uploaded. The Space builds in a few minutes: https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

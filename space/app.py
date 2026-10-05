"""Hugging Face Space entry point: image / text / fusion food recognition on CPU (the VLM is disabled).

The bundle is built by scripts/deploy_space.py: it contains this file, src/foodmm, configs/default.yaml and
work/ with the three Milestone 2 head checkpoints and classes.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from foodmm.config import load_config  # noqa: E402
from foodmm.demo.app import build_app  # noqa: E402
from foodmm.demo.examples import load_examples  # noqa: E402
from foodmm.demo.predictor import DemoPredictor  # noqa: E402

WORK = ROOT / "work"
cfg = load_config(ROOT / "configs" / "default.yaml",
                  [f"paths.work_dir={WORK}", f"paths.data_root={WORK}", "demo.vlm=false"])
demo = build_app(DemoPredictor(cfg), None, load_examples(cfg))
demo.queue(default_concurrency_limit=1).launch()

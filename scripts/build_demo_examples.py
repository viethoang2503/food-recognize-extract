#!/usr/bin/env python
"""Precompute demo examples (predictions and, optionally, VLM output) into work_dir/demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args  # noqa: E402
from foodmm.demo.examples import build_examples, demo_dir  # noqa: E402
from foodmm.demo.predictor import DemoPredictor  # noqa: E402
from foodmm.demo.vlm_service import LazyVLM  # noqa: E402
from foodmm.vlm.backend import make_backend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--with_vlm", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    vlm = LazyVLM(lambda: make_backend(cfg), cfg["vlm"]["max_image_side"]) if args.with_vlm else None
    items = build_examples(cfg, DemoPredictor(cfg), vlm=vlm, n=args.n)
    print(f"Wrote {len(items)} examples to {demo_dir(cfg) / 'examples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

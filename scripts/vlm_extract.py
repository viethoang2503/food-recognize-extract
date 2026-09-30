#!/usr/bin/env python
"""Extract structured food information with a local VLM (resumable JSONL per prompt mode)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.utils import load_json  # noqa: E402
from foodmm.vlm.backend import make_backend  # noqa: E402
from foodmm.vlm.extract import run_extraction  # noqa: E402
from foodmm.vlm.prompts import MODES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=[*MODES, "all"])
    parser.add_argument("--n", type=int, default=None, help="only the first N ids of the sample")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    if not paths["manifest"].exists():
        raise SystemExit(f"Missing {paths['manifest']}: run scripts/prepare_data.py first")
    df, classes = load_manifest(paths["manifest"]), load_json(paths["classes"])
    modes = list(MODES) if args.mode == "all" else [args.mode]
    backend = make_backend(cfg)
    for mode in modes:
        res = run_extraction(cfg, mode, backend, df, classes, n=args.n)
        print(f"{mode}: n_new={res['n_new']} n_total={res['n_total']} n_valid={res['n_valid']} -> {res['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

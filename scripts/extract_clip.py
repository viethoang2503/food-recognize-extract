#!/usr/bin/env python
"""Extract frozen CLIP features (pooled + tokens) for clean and corrupted inputs; resumable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.extract import PARTS, run_extraction  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    results = run_extraction(cfg, args.parts.split(","), args.splits.split(","), force=args.force)
    print(f"{sum(r['status'] == 'done' for r in results)} extracted, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

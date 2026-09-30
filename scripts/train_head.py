#!/usr/bin/env python
"""Train one head on cached CLIP features (or compute late fusion with --head late)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.heads import HEADS  # noqa: E402
from foodmm.clip.train_heads import run_late, train_head_run  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, choices=[*HEADS, "late"])
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if args.head == "late":
        metrics = run_late(cfg, cfg["data"]["text_mask"], force=args.force)
    else:
        cfg["head"]["name"] = args.head
        metrics = train_head_run(cfg, force=args.force)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

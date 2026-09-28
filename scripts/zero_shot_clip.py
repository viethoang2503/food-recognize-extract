#!/usr/bin/env python
"""CLIP zero-shot baseline on the test split (prompt from clip.prompt)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.zero_shot import run_zero_shot  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    print(json.dumps(run_zero_shot(config_from_args(args), force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

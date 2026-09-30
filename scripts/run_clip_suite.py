#!/usr/bin/env python
"""Run every Milestone 2 main run; finished runs are skipped."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.suite import run_suite  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip_zeroshot", action="store_true")
    parser.add_argument("--force", action="store_true", help="retrain every run")
    add_config_args(parser)
    args = parser.parse_args(argv)
    table = run_suite(config_from_args(args), force=args.force, skip_zeroshot=args.skip_zeroshot)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

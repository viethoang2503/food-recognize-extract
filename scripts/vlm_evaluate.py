#!/usr/bin/env python
"""Summarise VLM extractions and create / score the manual grading sheet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.utils import load_json  # noqa: E402
from foodmm.vlm.evaluate import evaluate_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    summary = evaluate_all(cfg, load_manifest(paths["manifest"]), load_json(paths["classes"]))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

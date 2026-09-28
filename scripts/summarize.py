#!/usr/bin/env python
"""Collect runs/*/metrics_test.json into results/summary.csv and results/summary.md."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.analysis import collect_results, results_to_markdown  # noqa: E402
from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.utils import ensure_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    paths = work_paths(config_from_args(args))
    df = collect_results(paths["runs"])
    if df.empty:
        raise SystemExit(f"No finished runs found in {paths['runs']}")
    out = ensure_dir(paths["results"])
    df.to_csv(out / "summary.csv", index=False)
    md = results_to_markdown(df)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

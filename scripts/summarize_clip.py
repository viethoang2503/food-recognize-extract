#!/usr/bin/env python
"""Write the Milestone 2 tables and significance tests into work_dir/clip/results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.features import clip_paths  # noqa: E402
from foodmm.clip.report import (  # noqa: E402
    add_acc_ci, collect_clip_results, collect_robust, main_table, missing_table, significance_table,
    table_to_markdown,
)
from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.utils import ensure_dir  # noqa: E402

PCT = ["acc", "acc_lo", "acc_hi", "top5", "macro_f1", "full", "no_image", "no_text",
       "acc_a", "acc_b", "diff", "diff_lo", "diff_hi"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cp = clip_paths(cfg)
    df = collect_clip_results(cp["runs"])
    if df.empty:
        raise SystemExit(f"No finished runs found in {cp['runs']}")
    out = ensure_dir(cp["results"])
    mask, md = cfg["clip"]["main_mask"], float(cfg["head"]["default_modality_dropout"])
    st, seed = cfg["stats"], int(cfg["seed"])
    n_boot, alpha = int(st["n_boot"]), float(st["alpha"])
    tables = {"main": add_acc_ci(main_table(df, md), cp["runs"], n_boot, alpha, seed),
              "missing": missing_table(collect_robust(cp["runs"]), mask),
              "significance": significance_table(work_paths(cfg)["work_dir"], st["pairs"], n_boot, alpha, seed)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        md_text = table_to_markdown(table, PCT)
        (out / f"{name}.md").write_text(md_text + "\n", encoding="utf-8")
        print(f"\n## {name}\n{md_text}")
    print(f"\nWrote results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build manifest.csv, classes.json and stats.json from the extracted dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.prepare import build_manifest  # noqa: E402
from foodmm.utils import ensure_dir, save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the manifest exists")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    if paths["manifest"].exists() and not args.force:
        print(f"Manifest already exists at {paths['manifest']} (use --force to rebuild)")
        return 0
    d = cfg["data"]
    df, classes, stats = build_manifest(
        cfg["paths"]["data_root"], val_ratio=float(d["val_ratio"]), seed=int(cfg["seed"]),
        subset_classes=d["subset_classes"], max_per_class=d["max_per_class"],
        max_chars=int(d["max_chars"]), max_drop_ratio=float(d["max_drop_ratio"]),
    )
    ensure_dir(paths["data_dir"])
    df.to_csv(paths["manifest"], index=False)
    save_json(classes, paths["classes"])
    save_json(stats, paths["stats"])
    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(df)} rows to {paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

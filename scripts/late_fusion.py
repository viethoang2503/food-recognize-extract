#!/usr/bin/env python
"""Late fusion of an image run and a text run; the image weight w is tuned on val."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from foodmm.config import add_config_args, config_from_args, load_config, save_config, work_paths  # noqa: E402
from foodmm.late_fusion import check_aligned, combine, load_preds, search_weight  # noqa: E402
from foodmm.metrics import compute_metrics, softmax  # noqa: E402
from foodmm.utils import save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_run", default="image")
    parser.add_argument("--text_run", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    runs = work_paths(cfg)["runs"]
    img_dir, txt_dir = runs / args.image_run, runs / args.text_run
    for d in (img_dir, txt_dir):
        if not (d / "preds_val.npz").exists():
            raise SystemExit(f"Missing predictions in {d}: finish run '{d.name}' first.")
    text_mask = load_config(txt_dir / "config.yaml")["data"]["text_mask"]
    run_name = args.run_name or f"late_{text_mask}"
    run_dir = runs / run_name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "metrics_test.json").exists():
        print(f"{run_name} is already finished (use --force to rerun)")
        return 0

    preds = {(m, s): load_preds(d, s) for m, d in (("img", img_dir), ("txt", txt_dir)) for s in ("val", "test")}
    for s in ("val", "test"):
        check_aligned(preds[("img", s)], preds[("txt", s)])
    probs = {k: softmax(v["logits"]) for k, v in preds.items()}
    w, curve = search_weight(probs[("img", "val")], probs[("txt", "val")], preds[("img", "val")]["labels"],
                             step=float(cfg["late_fusion"]["w_step"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    fused = {}
    for s in ("val", "test"):
        fused[s] = combine(probs[("img", s)], probs[("txt", s)], w)
        np.savez(run_dir / f"preds_{s}.npz", logits=np.log(fused[s] + 1e-12).astype(np.float32),
                 labels=preds[("img", s)]["labels"], ids=preds[("img", s)]["ids"])
    metrics = compute_metrics(fused["test"], preds[("img", "test")]["labels"])
    metrics.update({"run": run_name, "modality": "late", "text_mask": text_mask, "w": w,
                    "image_run": args.image_run, "text_run": args.text_run,
                    "best_val_acc": max(r["acc"] for r in curve)})
    cfg["data"]["text_mask"] = text_mask  # record the mask of the fused text run, not the CLI default
    cfg["run"] = {"name": run_name, "modality": "late"}
    save_config(cfg, run_dir / "config.yaml")
    save_json(curve, run_dir / "weight_curve.json")
    save_json(metrics, run_dir / "metrics_test.json")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

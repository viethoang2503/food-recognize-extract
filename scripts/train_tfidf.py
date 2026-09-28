#!/usr/bin/env python
"""TF-IDF + logistic regression text baseline (C chosen on the val split)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from foodmm.config import add_config_args, config_from_args, save_config, work_paths  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.data.text_utils import text_column  # noqa: E402
from foodmm.metrics import compute_metrics  # noqa: E402
from foodmm.utils import load_json, save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    mask = cfg["data"]["text_mask"]
    col = text_column(mask)
    run_name = args.run_name or f"tfidf_{mask}"
    run_dir = paths["runs"] / run_name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "metrics_test.json").exists():
        print(f"{run_name} is already finished (use --force to retrain)")
        return 0

    df = load_manifest(paths["manifest"])
    classes = load_json(paths["classes"])
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    tc = cfg["tfidf"]
    vec = TfidfVectorizer(ngram_range=(1, int(tc["ngram_max"])), max_features=int(tc["max_features"]),
                          sublinear_tf=True, min_df=int(tc["min_df"]))
    X = {"train": vec.fit_transform(parts["train"][col])}
    X.update({s: vec.transform(parts[s][col]) for s in ("val", "test")})
    y = {s: parts[s]["label_idx"].to_numpy() for s in parts}


    best = None
    for C in tc["C_grid"]:
        clf = LogisticRegression(C=float(C), max_iter=1000)
        clf.fit(X["train"], y["train"])
        acc = float((clf.predict(X["val"]) == y["val"]).mean())
        print(f"C={C}: val_acc={acc:.4f}", flush=True)
        if best is None or acc > best[1]:
            best = (clf, acc, float(C))
    clf, val_acc, best_c = best
    if len(clf.classes_) != len(classes):
        raise SystemExit(f"Only {len(clf.classes_)}/{len(classes)} classes appear in the train split")

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"name": run_name, "modality": "tfidf"}
    save_config(cfg, run_dir / "config.yaml")
    for s in ("val", "test"):
        np.savez(run_dir / f"preds_{s}.npz", logits=clf.predict_log_proba(X[s]).astype(np.float32),
                 labels=y[s], ids=parts[s]["id"].to_numpy().astype(str))
    metrics = compute_metrics(clf.predict_log_proba(X["test"]), y["test"])
    metrics.update({"run": run_name, "modality": "tfidf", "text_mask": mask, "best_val_acc": val_acc, "C": best_c})
    save_json(metrics, run_dir / "metrics_test.json")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

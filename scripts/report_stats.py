#!/usr/bin/env python
"""Extra statistics for the final report, written to work_dir/results/report_stats.

Sections (each is skipped with a note when its inputs are missing):
  significance_bonferroni  clip/results/significance.csv + Bonferroni over its rows (run summarize_clip.py first)
  m1_ci                    bootstrap CI of the Milestone 1 runs (same method as the Milestone 2 table)
  leakage_audit            masked texts that still contain a class word once diacritics are removed
  classifier_errors        top confused class pairs of xattn_strict; classes most helped / hurt by adding text
  vlm_paired               image vs image_text dish accuracy on the VLM sample (paired, McNemar)
  vlm_vs_classifier        each VLM mode vs the CLIP classifier of the same input on the same images (McNemar)
  manual_grades            manual grades per mode with CIs and label-based accuracy on the same rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.report import table_to_markdown  # noqa: E402
from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.text_utils import build_mask_pattern, text_column  # noqa: E402
from foodmm.report_stats import (  # noqa: E402
    add_bonferroni, classifier_errors, graded_summary, residual_leakage, run_ci_table, vlm_paired, vlm_vs_classifier,
)
from foodmm.utils import ensure_dir, load_json  # noqa: E402

M1_RUNS = ["tfidf_none", "tfidf_strict", "text_none", "text_strict", "image", "late_none", "late_strict",
           "early_none", "early_strict"]


def _write(out: Path, name: str, table: pd.DataFrame) -> None:
    table.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(table_to_markdown(table, []) + "\n", encoding="utf-8")
    print(f"\n## {name}\n{table.to_string(index=False)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    wp = work_paths(cfg)
    work = wp["work_dir"]
    out = ensure_dir(work / "results" / "report_stats")
    st, seed = cfg["stats"], int(cfg["seed"])
    n_boot, alpha = int(st["n_boot"]), float(st["alpha"])

    sig = work / "clip" / "results" / "significance.csv"
    if sig.exists():
        _write(out, "significance_bonferroni", add_bonferroni(pd.read_csv(sig)))
    else:
        print(f"skip significance_bonferroni: {sig} not found (run summarize_clip.py)")

    _write(out, "m1_ci", run_ci_table(work / "runs", M1_RUNS, n_boot, alpha, seed))

    classes = load_json(wp["classes"]) if wp["classes"].exists() else None
    if wp["manifest"].exists() and classes:
        from foodmm.data.prepare import load_manifest

        df = load_manifest(wp["manifest"])
        rows = []
        for mask in ("strict", "exact"):
            col = text_column(mask)
            if col not in df.columns:
                continue
            for split in ("train", "val", "test"):
                res = residual_leakage(df.loc[df["split"] == split, col].tolist(),
                                       build_mask_pattern(classes, mask), top_k=15)
                rows.append({"mask": mask, "split": split, "n": res["n"], "non_ascii_rate": res["non_ascii_rate"],
                             "folded_hit_rate": res["folded_hit_rate"],
                             "top_tokens": ", ".join(f"{t} ({c})" for t, c in res["top_tokens"])})
        _write(out, "leakage_audit", pd.DataFrame(rows))
    else:
        print("skip leakage_audit: manifest or classes.json not found (run prepare_data.py)")

    from foodmm.late_fusion import load_preds

    clip_runs = work / "clip" / "runs"
    try:
        img_p, fus_p = load_preds(clip_runs / "image", "test"), load_preds(clip_runs / "xattn_strict", "test")
    except FileNotFoundError:
        img_p = fus_p = None
    if classes and img_p is not None:
        confused, gain = classifier_errors(img_p, fus_p, classes, k=10)
        _write(out, "classifier_confused_xattn", confused)
        _write(out, "classifier_text_gain", gain)
    else:
        print("skip classifier_errors: clip/runs/image or clip/runs/xattn_strict predictions not found")

    from foodmm.vlm.extract import read_records, vlm_dir

    vdir, v = vlm_dir(cfg), cfg["vlm"]
    thr = float(v["match_threshold"])
    recs = {m: read_records(vdir / f"extract_{m}.jsonl") for m in ("image", "text", "image_text")}
    if classes and recs["image"] and recs["image_text"]:
        res = vlm_paired(recs["image"], recs["image_text"], classes, thr, n_boot, seed, alpha)
        valid = {f"valid_{m}": sum(bool(r["valid"]) for r in recs[m]) for m in recs}
        _write(out, "vlm_paired", pd.DataFrame([{"a": "image", "b": "image_text", **res, **valid}]))
        rows = []
        for mode in ("image", "image_text"):
            rel = v["compare_runs"].get(mode)
            try:
                clf = load_preds(work / rel, "test")
            except (FileNotFoundError, TypeError):
                print(f"skip vlm_vs_classifier for {mode}: no predictions in {rel}")
                continue
            rows.append({"vlm_mode": mode, "classifier": rel,
                         **vlm_vs_classifier(recs[mode], clf, classes, thr, n_boot, seed, alpha)})
        if rows:
            _write(out, "vlm_vs_classifier", pd.DataFrame(rows))
    else:
        print(f"skip vlm_paired: extraction records not found in {vdir}")

    sheet, key = vdir / "manual_grading.csv", vdir / "manual_grading_key.csv"
    if classes and sheet.exists() and key.exists():
        summary, paired = graded_summary(pd.read_csv(sheet), pd.read_csv(key), classes, thr, n_boot, seed, alpha)
        _write(out, "manual_grades", summary)
        _write(out, "manual_grades_paired", paired)
    else:
        print(f"skip manual_grades: {sheet.name} or {key.name} not found in {vdir}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

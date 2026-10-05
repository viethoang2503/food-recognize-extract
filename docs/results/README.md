# Result tables

Copies of the tables that the scripts write to Google Drive (`MyDrive/foodmm/`), committed so that the numbers in
the report can be checked without access to the Drive. Regenerate them with the commands below; do not edit by hand.

| File | Written by | Drive location |
|---|---|---|
| `data_stats.json` | `scripts/prepare_data.py` | `data/stats.json` |
| `milestone1_summary.md` | `scripts/summarize.py` | `results/summary.md` |
| `clip_main.md` | `scripts/summarize_clip.py` | `clip/results/main.md` |
| `clip_significance.md` | `scripts/summarize_clip.py` | `clip/results/significance.md` |
| `clip_missing.md` | `scripts/summarize_clip.py` | `clip/results/missing.md` |
| `vlm_eval_summary.md` | `scripts/vlm_evaluate.py` | `vlm/qwen3vl4b/eval_summary.md` |
| `vlm_classifier_on_sample.md` | `scripts/vlm_evaluate.py` | `vlm/qwen3vl4b/classifier_on_sample.md` |
| `vlm_manual_scores.csv` | `scripts/vlm_evaluate.py` (from the graded sheet) | `vlm/qwen3vl4b/manual_scores.csv` |
| `report_stats/` | `scripts/report_stats.py` | `results/report_stats/` |
| `vlm_grading/manual_grading.csv` | `scripts/vlm_evaluate.py` (sheet), graded by the authors | `vlm/qwen3vl4b/manual_grading.csv` |
| `vlm_grading/manual_grading_key.csv` | `scripts/vlm_evaluate.py` (row to prompt mode, used only for scoring) | `vlm/qwen3vl4b/manual_grading_key.csv` |

Accuracies are in percent; `acc_lo` / `acc_hi` and `diff_lo` / `diff_hi` are 95% percentile bootstrap intervals
(1,000 resamples); `mcnemar_p` is the uncorrected two-sided McNemar p-value (0 means it underflowed double precision).

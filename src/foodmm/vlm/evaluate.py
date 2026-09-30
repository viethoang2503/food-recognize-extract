"""Evaluate VLM extractions: validity, latency, dish accuracy, ingredient grounding, manual grades."""
from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.text_utils import STOPWORDS, class_to_phrase, clean_text, phrase_variants, word_variants
from ..late_fusion import load_preds
from ..stats import bootstrap_ci, correct_vector
from .extract import read_records, vlm_dir
from .prompts import MODES, TEXT_MODES

SUMMARY_COLUMNS = ["mode", "uses_text", "n", "valid_rate", "retry_rate", "latency_mean", "latency_p90", "dish_acc",
                   "dish_acc_lo", "dish_acc_hi", "unmapped_rate", "ingredient_grounding"]
CLASSIFIER_COLUMNS = ["name", "run", "n", "acc", "acc_lo", "acc_hi"]
GRADING_COLUMNS = ["row", "id", "label", "image_path", "dish_name", "cuisine", "main_ingredients",
                   "cooking_method", "confidence", "grade_dish", "grade_ingredients", "grade_method", "notes"]
GRADES = ["grade_dish", "grade_ingredients", "grade_method"]
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(s: str) -> str:
    return _NON_ALNUM.sub(" ", str(s).lower()).strip()


def _stem_words(s: str) -> set[str]:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split() if w not in STOPWORDS}


def map_dish_to_class(dish_name: str, classes: Sequence[str], threshold: float = 0.6) -> str | None:
    name = normalize_name(dish_name)
    if not name:
        return None
    hits = []
    for c in classes:
        for v in phrase_variants(class_to_phrase(c)):
            if re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", name):
                hits.append((len(v), c))
    if hits:
        return max(hits)[1]
    best, best_score = None, 0.0
    words = _stem_words(name)
    for c in classes:
        phrase = class_to_phrase(c)
        ratio = SequenceMatcher(None, name, phrase).ratio()
        cw = _stem_words(phrase)
        overlap = len(words & cw) / min(len(words), len(cw)) if words and cw else 0.0  # overlap coefficient
        score = max(ratio, overlap)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


def ingredient_grounding(ingredients: Sequence[str], text: str) -> float | None:
    if not ingredients:
        return None
    tokens = set(re.findall(r"[a-z]+", clean_text(text)))
    matched = 0
    for ing in ingredients:
        words = [w for w in re.findall(r"[a-z]+", str(ing).lower()) if len(w) >= 3 and w not in STOPWORDS]
        if any(word_variants(w) & tokens for w in words):
            matched += 1
    return matched / len(ingredients)


def summarize_mode(records: Sequence[dict], classes: Sequence[str], texts: dict[str, str],
                   threshold: float = 0.6, n_boot: int = 1000, seed: int = 0) -> dict:
    n = len(records)
    mode = records[0]["mode"] if records else ""
    if n == 0:
        return {"mode": mode, "uses_text": mode in TEXT_MODES, "n": 0,
                **{k: float("nan") for k in SUMMARY_COLUMNS[3:]}}
    valid = [r for r in records if r["valid"]]
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    mapped = [map_dish_to_class(r["output"]["dish_name"], classes, threshold) if r["valid"] else None
              for r in records]
    correct = np.array([m is not None and m == r["label"] for m, r in zip(mapped, records)])
    acc_lo, acc_hi = bootstrap_ci(correct, n_boot, seed=seed)
    grounding = [g for r in valid
                 if (g := ingredient_grounding(r["output"]["main_ingredients"], texts.get(r["id"], ""))) is not None]
    return {
        "mode": mode, "uses_text": mode in TEXT_MODES, "n": n,
        "valid_rate": len(valid) / n,
        "retry_rate": sum(r["attempts"] > 1 for r in records) / n,
        "latency_mean": float(np.mean(lat)) if lat else float("nan"),
        "latency_p90": float(np.percentile(lat, 90)) if lat else float("nan"),
        "dish_acc": float(correct.mean()), "dish_acc_lo": acc_lo, "dish_acc_hi": acc_hi,
        "unmapped_rate": (sum(m is None for m, r in zip(mapped, records) if r["valid"]) / len(valid)
                          if valid else float("nan")),
        "ingredient_grounding": float(np.mean(grounding)) if grounding else float("nan"),
    }


def classifier_on_sample(work_dir: str | Path, runs: dict[str, str], ids: Sequence[str], n_boot: int = 1000,
                         seed: int = 0) -> pd.DataFrame:
    """Accuracy of trained classifiers on exactly the VLM sample ids; runs without predictions are skipped."""
    rows = []
    for name, rel in runs.items():
        try:
            preds = load_preds(Path(work_dir) / rel, "test")
        except FileNotFoundError:
            print(f"Skipping classifier '{name}': no predictions in {rel}")
            continue
        by_id = dict(zip(preds["ids"].astype(str), correct_vector(preds)))
        c = np.array([by_id[i] for i in ids if i in by_id], dtype=bool)
        lo, hi = bootstrap_ci(c, n_boot, seed=seed)
        rows.append({"name": name, "run": rel, "n": len(c), "acc": float(c.mean()) if len(c) else float("nan"),
                     "acc_lo": lo, "acc_hi": hi})
    return pd.DataFrame(rows, columns=CLASSIFIER_COLUMNS)


def grading_key_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_key.csv")


def make_grading_sheet(records_by_mode: dict[str, list[dict]], ids: Sequence[str], df: pd.DataFrame,
                       path: str | Path, seed: int = 0) -> bool:
    """Write the shuffled, mode-blind grading CSV plus its key file once (False if the sheet already exists)."""
    path = Path(path)
    if path.exists():
        return False
    meta = df.set_index(df["id"].astype(str))
    by_mode = {m: {r["id"]: r for r in records_by_mode.get(m, [])} for m in MODES}
    rows = []
    for sid in ids:
        for mode in MODES:
            rec = by_mode[mode].get(sid)
            if rec is None:
                continue
            out = rec["output"] or {}
            rows.append({"mode": mode, "id": sid, "label": rec["label"], "image_path": meta.loc[sid, "image_path"],
                         "dish_name": out.get("dish_name", ""), "cuisine": out.get("cuisine", ""),
                         "main_ingredients": "; ".join(out.get("main_ingredients", [])),
                         "cooking_method": out.get("cooking_method", ""), "confidence": out.get("confidence"),
                         "grade_dish": None, "grade_ingredients": None, "grade_method": None,
                         "notes": "" if rec["valid"] else f"invalid: {rec['error']}"})
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = [{**rows[i], "row": k} for k, i in enumerate(order)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=GRADING_COLUMNS).to_csv(path, index=False)
    pd.DataFrame(rows, columns=["row", "mode"]).to_csv(grading_key_path(path), index=False)
    return True


def score_grading_sheet(path: str | Path) -> pd.DataFrame | None:
    key_path = grading_key_path(path)
    if not key_path.exists():
        raise SystemExit(f"Missing {key_path}: it maps each graded row back to its prompt mode")
    sheet = pd.read_csv(path)
    graded = sheet[sheet[GRADES].notna().any(axis=1)].merge(pd.read_csv(key_path), on="row")
    if graded.empty:
        return None
    scores = graded.groupby("mode")[GRADES].mean()
    scores["n_graded"] = graded.groupby("mode").size()
    return scores.reset_index()


_PCT = {"valid_rate", "retry_rate", "dish_acc", "dish_acc_lo", "dish_acc_hi", "unmapped_rate",
        "ingredient_grounding", "acc", "acc_lo", "acc_hi"}


def _markdown(table: pd.DataFrame) -> str:
    def fmt(col, v):
        if col in ("latency_mean", "latency_p90"):
            return f"{v:.2f}s"
        return f"{v * 100:.1f}%" if col in _PCT else str(v)

    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, r[c]) for c in cols) + " |" for r in table.to_dict("records")]
    return "\n".join(lines)


def evaluate_all(cfg: dict, df: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    from ..config import work_paths
    from ..utils import load_json

    out_dir = vlm_dir(cfg)
    v, seed = cfg["vlm"], int(cfg["seed"])
    texts = dict(zip(df["id"].astype(str), df["text"].astype(str)))
    records = {m: read_records(out_dir / f"extract_{m}.jsonl") for m in MODES}
    records = {m: r for m, r in records.items() if r}
    if not records:
        raise SystemExit(f"No extraction results in {out_dir}: run scripts/vlm_extract.py first")
    summary = pd.DataFrame([summarize_mode(r, classes, texts, float(v["match_threshold"]), int(v["n_boot"]), seed)
                            for r in records.values()], columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    note = ("\n_ingredient_grounding is a reference only: modes with uses_text=True read the text it is measured "
            "against. Compare modes with the manual grades._\n")
    (out_dir / "eval_summary.md").write_text(_markdown(summary) + "\n" + note, encoding="utf-8")
    sample_ids = load_json(out_dir / "sample_ids.json")
    clf = classifier_on_sample(work_paths(cfg)["work_dir"], dict(v["compare_runs"]), sample_ids, int(v["n_boot"]), seed)
    clf.to_csv(out_dir / "classifier_on_sample.csv", index=False)
    (out_dir / "classifier_on_sample.md").write_text(_markdown(clf) + "\n", encoding="utf-8")
    sheet = out_dir / "manual_grading.csv"
    if make_grading_sheet(records, sample_ids[:int(v["n_grading"])], df, sheet, seed=seed):
        print(f"Wrote {sheet} (mode-blind, shuffled): fill the grade_* columns, then run vlm_evaluate.py again")
    scores = score_grading_sheet(sheet)
    if scores is not None:
        scores.to_csv(out_dir / "manual_scores.csv", index=False)
        print(scores.to_string(index=False))
    return summary

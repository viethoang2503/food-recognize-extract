"""Run the VLM over a fixed test sample; one JSONL line per image, resumable."""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import work_paths
from ..data.text_utils import text_column
from ..utils import ensure_dir, load_json, save_json
from .backend import resize_for_vlm
from .parse import ExtractionError, parse_output
from .prompts import MODES, RETRY_PROMPT, SYSTEM_PROMPT, build_prompt


def vlm_dir(cfg: dict) -> Path:
    return work_paths(cfg)["work_dir"] / "vlm" / str(cfg["vlm"]["tag"])


def select_sample(df: pd.DataFrame, n: int, seed: int, path: str | Path) -> list[str]:
    """Stratified sample of test ids (about n / n_classes per class), cached in `path`."""
    path = Path(path)
    if path.exists():
        return load_json(path)
    rng = np.random.default_rng(seed)
    test = df[df["split"] == "test"]
    groups = {c: g["id"].astype(str).tolist() for c, g in test.groupby("label", sort=True)}
    per_class = max(1, n // max(1, len(groups)))
    chosen = []
    for ids in groups.values():
        chosen += list(rng.choice(ids, size=min(per_class, len(ids)), replace=False))
    rest = sorted(set(test["id"].astype(str)) - set(chosen))
    if len(chosen) < n and rest:
        chosen += list(rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False))
    chosen = [str(i) for i in rng.permutation(chosen)[:n]]
    save_json(chosen, path)
    return chosen


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_one(backend, image: Image.Image | None, prompt: str) -> dict:
    t0 = time.time()
    raw, error, output, attempts = "", None, None, 0
    for attempt in range(2):
        attempts = attempt + 1
        p = prompt if attempt == 0 else f"{prompt}\n\n{RETRY_PROMPT.format(error=error)}"
        raw = backend.generate(image, p, SYSTEM_PROMPT)
        try:
            output, error = parse_output(raw), None
            break
        except ExtractionError as e:
            error = str(e)
    return {"valid": output is not None, "attempts": attempts, "latency_s": round(time.time() - t0, 3),
            "output": output, "raw": raw, "error": error}


def run_extraction(cfg: dict, mode: str, backend, df: pd.DataFrame, classes: Sequence[str],
                   n: int | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    v = cfg["vlm"]
    out_dir = ensure_dir(vlm_dir(cfg))
    ids = select_sample(df, int(v["n_samples"]), int(cfg["seed"]), out_dir / "sample_ids.json")
    if n is not None:
        ids = ids[:int(n)]
    out_path = out_dir / f"extract_{mode}.jsonl"
    done = {r["id"] for r in read_records(out_path)}
    todo = [i for i in ids if i not in done]
    rows = df.set_index(df["id"].astype(str))
    col = text_column(v["text_mask"])
    data_root = Path(cfg["paths"]["data_root"])
    n_valid, t_start = 0, time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for k, sid in enumerate(todo, 1):
            row = rows.loc[sid]
            prompt = build_prompt(mode, text=row[col], max_chars=int(v["max_context_chars"]))
            if mode == "text":
                res = extract_one(backend, None, prompt)
            else:
                try:
                    with Image.open(data_root / row["image_path"]) as im:
                        image = resize_for_vlm(im, int(v["max_image_side"]))
                except Exception:  # noqa: BLE001
                    res = {"valid": False, "attempts": 0, "latency_s": None, "output": None, "raw": "",
                           "error": "bad image"}
                else:
                    res = extract_one(backend, image, prompt)
            n_valid += bool(res["valid"])
            rec = {"id": sid, "label": row["label"], "mode": mode, **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % int(v["log_every"]) == 0 or k == len(todo):
                print(f"[{mode}] {k}/{len(todo)} new, valid {n_valid}/{k}, "
                      f"{(time.time() - t_start) / k:.2f}s per image", flush=True)
    all_recs = read_records(out_path)
    return {"mode": mode, "path": str(out_path), "n_total": len(all_recs), "n_new": len(todo),
            "n_valid": sum(bool(r["valid"]) for r in all_recs)}

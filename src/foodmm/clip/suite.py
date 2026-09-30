"""The Milestone 2 experiment list: zero-shot, image, text, fusion heads and late fusion per text mask."""
from __future__ import annotations

import copy

import pandas as pd

from ..config import parse_value, set_by_path
from .train_heads import run_late, train_head_run
from .zero_shot import run_zero_shot


def suite_jobs(cfg: dict) -> list[tuple[str, list[str]]]:
    s = cfg["clip_suite"]
    jobs: list[tuple[str, list[str]]] = [("zeroshot", []), ("head", ["head.name=image"])]
    jobs += [("head", ["head.name=text", f"data.text_mask={m}"]) for m in s["masks"]]
    jobs += [("head", [f"head.name={h}", f"data.text_mask={m}"]) for m in s["masks"] for h in s["fusion_heads"]]
    jobs += [("late", [f"data.text_mask={m}"]) for m in s["masks"]]
    return jobs


def job_config(cfg: dict, overrides: list[str]) -> dict:
    c = copy.deepcopy(cfg)
    for item in overrides:
        key, raw = item.split("=", 1)
        set_by_path(c, key, parse_value(raw))
    return c


def run_suite(cfg: dict, force: bool = False, skip_zeroshot: bool = False) -> pd.DataFrame:
    rows = []
    jobs = suite_jobs(cfg)
    for i, (kind, overrides) in enumerate(jobs, 1):
        c = job_config(cfg, overrides)
        print(f"[{i}/{len(jobs)}] {kind} {' '.join(overrides)}", flush=True)
        if kind == "zeroshot":
            if skip_zeroshot:
                continue
            m = run_zero_shot(c, force=force)
        elif kind == "head":
            m = train_head_run(c, force=force)
        else:
            m = run_late(c, c["data"]["text_mask"], force=force)
        rows.append({"kind": kind, "run": m["run"], "acc": m["acc"]})
    return pd.DataFrame(rows, columns=["kind", "run", "acc"])

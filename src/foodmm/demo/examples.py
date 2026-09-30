"""Precomputed demo examples (work without a GPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import work_paths
from ..data.prepare import load_manifest
from ..data.text_utils import class_to_phrase
from ..utils import ensure_dir, load_json, save_json


def demo_dir(cfg: dict) -> Path:
    return work_paths(cfg)["work_dir"] / "demo"


def pick_examples(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """One random test image from each of n random classes."""
    rng = np.random.default_rng(seed)
    test = df[df["split"] == "test"]
    labels = sorted(test["label"].unique())
    chosen = rng.choice(labels, size=min(n, len(labels)), replace=False)
    return [str(rng.choice(test.loc[test["label"] == c, "id"].to_numpy())) for c in chosen]


def build_examples(cfg: dict, predictor, vlm=None, n: int | None = None) -> list[dict]:
    df = load_manifest(work_paths(cfg)["manifest"]).set_index("id", drop=False)
    ids = pick_examples(df, int(n or cfg["demo"]["n_examples"]), int(cfg["seed"]))
    out = demo_dir(cfg)
    ensure_dir(out / "examples")
    items = []
    for k, sid in enumerate(ids, 1):
        row = df.loc[sid]
        with Image.open(Path(cfg["paths"]["data_root"]) / row["image_path"]) as im:
            img = im.convert("RGB")
        img.thumbnail((512, 512))
        rel = f"examples/{k:02d}.jpg"
        img.save(out / rel, quality=90)
        text = str(row["text"])[:500]
        pred = predictor.predict(img, text)
        items.append({"name": f"{k}. {class_to_phrase(row['label'])}", "id": sid, "label": row["label"],
                      "image": rel, "text": text,
                      "pred": {key: pred[key] for key in ("image", "text", "fusion", "masked_text")},
                      "vlm": vlm.extract(img, pred["masked_text"]) if vlm is not None else None})
        print(f"example {k}/{len(ids)}: {row['label']}", flush=True)
    save_json(items, out / "examples.json")
    return items


def load_examples(cfg: dict) -> list[dict] | None:
    path = demo_dir(cfg) / "examples.json"
    if not path.exists():
        return None
    items = load_json(path)
    for it in items:
        it["image_path"] = str(path.parent / it["image"])
    return items

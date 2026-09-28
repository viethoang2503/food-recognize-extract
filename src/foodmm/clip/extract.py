"""Feature-extraction jobs (clean and corrupted), resumable per shard."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.text_utils import text_column
from .corrupt import IMAGE_KINDS, apply_image_corruption, corruption_specs, sample_seed, word_drop
from .features import FeatureWriter, clip_paths, image_set_name, text_set_name

PARTS = ("image", "text_none", "text_exact", "text_strict", "corrupt")


@dataclass
class Job:
    name: str
    kind: str  # "image" | "text"
    split: str
    column: str | None = None
    corruption: tuple[str, float] | None = None


def build_jobs(cfg: dict, parts: list[str], splits: list[str]) -> list[Job]:
    jobs: list[Job] = []
    main = cfg["clip"]["main_mask"]
    for part in parts:
        if part == "image":
            jobs += [Job(image_set_name(s), "image", s) for s in splits]
        elif part in ("text_none", "text_exact", "text_strict"):
            mask = part[len("text_"):]
            jobs += [Job(text_set_name(mask, s), "text", s, text_column(mask)) for s in splits]
        elif part == "corrupt":
            for name, kind, level in corruption_specs(cfg):
                if kind in IMAGE_KINDS:
                    jobs.append(Job(image_set_name("test", name), "image", "test", None, (kind, level)))
                else:
                    jobs.append(Job(text_set_name(main, "test", name), "text", "test", text_column(main), (kind, level)))
        else:
            raise ValueError(f"unknown part {part!r}; expected one of {PARTS}")
    return jobs


class _ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: str | Path, image_processor, corruption, seed: int):
        self.paths = rows["image_path"].tolist()
        self.ids = rows["id"].astype(str).tolist()
        self.root = Path(data_root)
        self.image_processor = image_processor
        self.corruption = corruption
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        bad = False
        try:
            with Image.open(self.root / self.paths[i]) as im:
                img = im.convert("RGB")
        except Exception:  # noqa: BLE001 - any unreadable image becomes a black image
            img, bad = Image.new("RGB", (224, 224)), True
        if self.corruption is not None:
            kind, level = self.corruption
            img = apply_image_corruption(img, kind, level, sample_seed(self.seed, self.ids[i]))
        return self.image_processor(images=img, return_tensors="pt")["pixel_values"][0], bad


def _collate(batch):
    return torch.stack([b[0] for b in batch]), int(sum(b[1] for b in batch))


def run_job(job: Job, df: pd.DataFrame, encoder, cfg: dict, out_dir: Path) -> dict:
    sub = df[df["split"] == job.split].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"split {job.split!r} is empty in the manifest")
    cc = cfg["clip"]
    writer = FeatureWriter(out_dir, sub["id"].astype(str).tolist(), int(cc["shard_size"]))
    if writer.is_done():
        return {"name": job.name, "status": "skipped"}
    bs, seed = int(cc["batch_size"]), int(cfg["seed"])
    bad_total, max_bad = 0, int(cfg["data"]["max_bad_images"])
    for shard_idx, idx in writer.pending_shards():
        rows = sub.iloc[idx]
        if job.kind == "image":
            ds = _ImageDataset(rows, cfg["paths"]["data_root"], encoder.image_processor, job.corruption, seed)
            loader = DataLoader(ds, batch_size=bs, num_workers=int(cfg["data"]["num_workers"]), collate_fn=_collate)
            pooled, tokens = [], []
            for pixels, bad in loader:
                bad_total += bad
                if bad_total > max_bad:
                    raise SystemExit(f"{job.name}: more than {max_bad} unreadable images")
                p, t = encoder.encode_pixels(pixels)
                pooled.append(p)
                tokens.append(t)
            arrays = {"pooled": np.concatenate(pooled), "tokens": np.concatenate(tokens)}
        else:
            texts = rows[job.column].fillna("").astype(str).tolist()
            if job.corruption is not None:
                texts = [word_drop(t, job.corruption[1], sample_seed(seed, i)) for t, i in zip(texts, rows["id"].astype(str))]
            parts = [encoder.encode_texts(texts[i:i + bs]) for i in range(0, len(texts), bs)]
            arrays = {"pooled": np.concatenate([p[0] for p in parts]), "tokens": np.concatenate([p[1] for p in parts]),
                      "token_mask": np.concatenate([p[2] for p in parts])}
        writer.write_shard(shard_idx, arrays)
        print(f"{job.name}: shard {shard_idx + 1}/{writer.n_shards}", flush=True)
    n = writer.finalize()
    return {"name": job.name, "status": "done", "n": n, "bad_images": bad_total}


def run_extraction(cfg: dict, parts: list[str], splits: list[str], force: bool = False, encoder=None) -> list[dict]:
    from ..config import work_paths
    from ..data.prepare import load_manifest
    from .encoder import ClipEncoder

    manifest = work_paths(cfg)["manifest"]
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}: run scripts/prepare_data.py first")
    df = load_manifest(manifest)
    root = clip_paths(cfg)["features"]
    results = []
    for job in build_jobs(cfg, parts, splits):
        out_dir = root / job.name
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        if (out_dir / "done.json").exists():
            print(f"skip {job.name} (already extracted)", flush=True)
            results.append({"name": job.name, "status": "skipped"})
            continue
        if encoder is None:
            cc = cfg["clip"]
            encoder = ClipEncoder(cc["model_name"], fp16=bool(cc["fp16"]), n_tokens=int(cc["n_tokens"]))
        res = run_job(job, df, encoder, cfg, out_dir)
        print(res, flush=True)
        results.append(res)
    return results

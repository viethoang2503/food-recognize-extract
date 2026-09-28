"""On-disk feature sets: one directory of .npy arrays per (modality, mask, split, corruption)."""
from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import work_paths
from ..utils import ensure_dir, load_json, save_json


class MissingFeatures(FileNotFoundError):
    pass


def clip_paths(cfg: dict) -> dict[str, Path]:
    root = work_paths(cfg)["work_dir"] / "clip"
    return {"root": root, "features": root / "features" / str(cfg["clip"]["tag"]),
            "runs": root / "runs", "results": root / "results"}


def image_set_name(split: str, corruption: str | None = None) -> str:
    return f"image_{split}" + (f"_{corruption}" if corruption else "")


def text_set_name(mask: str, split: str, corruption: str | None = None) -> str:
    return f"text_{mask}_{split}" + (f"_{corruption}" if corruption else "")


def is_done(out_dir: str | Path) -> bool:
    return (Path(out_dir) / "done.json").exists()


def check_ids(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual.astype(str), expected.astype(str)):
        raise ValueError(f"ids of {what} do not match the manifest (re-extract with --force)")


class FeatureWriter:
    """Writes shards of `shard_size` rows (resumable), then concatenates them into .npy files."""

    def __init__(self, out_dir: str | Path, ids: Sequence[str], shard_size: int):
        self.out_dir = Path(out_dir)
        self.ids = np.asarray(ids).astype(str)
        self.shard_size = int(shard_size)
        self.shard_dir = self.out_dir / "shards"
        self.n_shards = max(1, -(-len(self.ids) // self.shard_size))

    def is_done(self) -> bool:
        return is_done(self.out_dir)

    def _shard_path(self, index: int) -> Path:
        return self.shard_dir / f"{index:05d}.npz"

    def pending_shards(self) -> list[tuple[int, np.ndarray]]:
        out = []
        for i in range(self.n_shards):
            idx = np.arange(i * self.shard_size, min((i + 1) * self.shard_size, len(self.ids)))
            if not self._shard_path(i).exists():
                out.append((i, idx))
        return out

    def write_shard(self, index: int, arrays: dict[str, np.ndarray]) -> None:
        ensure_dir(self.shard_dir)
        tmp = self.shard_dir / f"{index:05d}.tmp.npz"
        start = index * self.shard_size
        ids = self.ids[start:start + self.shard_size]
        np.savez(tmp, ids=ids, **arrays)
        os.replace(tmp, self._shard_path(index))

    def finalize(self) -> int:
        missing = [i for i in range(self.n_shards) if not self._shard_path(i).exists()]
        if missing:
            raise RuntimeError(f"{self.out_dir.name}: missing shards {missing[:5]}")
        parts: dict[str, list[np.ndarray]] = {}
        for i in range(self.n_shards):
            with np.load(self._shard_path(i)) as z:
                for k in z.files:
                    parts.setdefault(k, []).append(z[k])
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        check_ids(arrays["ids"], self.ids, self.out_dir.name)
        for k, v in arrays.items():
            np.save(self.out_dir / f"{k}.npy", v)
        save_json({"n": int(len(self.ids)), "keys": sorted(arrays)}, self.out_dir / "done.json")
        shutil.rmtree(self.shard_dir)
        return int(len(self.ids))


def load_feature_set(out_dir: str | Path, keys: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    out_dir = Path(out_dir)
    if not is_done(out_dir):
        raise MissingFeatures(f"Feature set {out_dir} is missing: run scripts/extract_clip.py first")
    meta = load_json(out_dir / "done.json")
    wanted = [k for k in (keys or meta["keys"]) if k != "ids"]
    out = {"ids": np.load(out_dir / "ids.npy")}
    for k in wanted:
        out[k] = np.load(out_dir / f"{k}.npy")
    return out

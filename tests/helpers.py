"""Shared test helpers: a tiny fake UPMC-style dataset and script runner."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["apple_pie", "caesar_salad", "french_fries"]
COLORS = {"apple_pie": (200, 150, 60), "caesar_salad": (60, 180, 60), "french_fries": (230, 210, 40)}
TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


def _write_image(path: Path, color: tuple[int, int, int], seed: int, size: int = 32) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-20, 21, (size, size, 3))
    arr = np.clip(np.array(color)[None, None, :] + noise, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_fake_dataset(root: Path, fmt: str = "csv", n_train: int = 10, n_test: int = 4,
                      csv_header: bool = False) -> Path:
    """Create images/<split>/<class>/*.jpg plus texts as CSV (texts/<split>_titles.csv) or .txt files."""
    root = Path(root)
    rows: dict[str, list[tuple[str, str, str]]] = {"train": [], "test": []}
    for split, n in (("train", n_train), ("test", n_test)):
        for ci, cls in enumerate(CLASSES):
            phrase = cls.replace("_", " ")
            for i in range(n):
                fname = f"{cls}_{split}_{i}.jpg"
                seed = ci * 1000 + i + (0 if split == "train" else 500)
                _write_image(root / "images" / split / cls / fname, COLORS[cls], seed)
                text = (f"<p>Best {phrase} recipe number {i}</p> easy homemade {phrase} "
                        f"with tasty ingredients http://example.com/{i}")
                rows[split].append((fname, text, cls))
    if fmt == "csv":
        for split, items in rows.items():
            path = root / "texts" / f"{split}_titles.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if csv_header:
                    writer.writerow(["image_path", "text", "food"])
                writer.writerows(items)
    elif fmt == "txt":
        for split, items in rows.items():
            for fname, text, cls in items:
                path = root / "texts" / split / cls / (Path(fname).stem + ".txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
    return root


def smoke_overrides(data_root: Path, work_dir: Path) -> list[str]:
    """--set values that make every script run in seconds on the fake dataset."""
    return [
        f"paths.data_root={data_root}", f"paths.work_dir={work_dir}",
        "data.val_ratio=0.3", "data.num_workers=0",
        "image.backbone=resnet18", "image.pretrained=false", "image.img_size=32",
        f"text.model_name={TINY_TEXT_MODEL}", "text.pretrained=false", "text.max_len=16",
        "fusion.hidden=16",
        "train.image.epochs=1", "train.image.batch_size=8",
        "train.text.epochs=1", "train.text.batch_size=8",
        "train.multimodal.epochs=1", "train.multimodal.batch_size=8",
        "train.amp=false", "tfidf.min_df=1", "tfidf.C_grid=[1]",
    ]


def run_script(script: str, *args: object, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, args)]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(f"{script} exited with {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


TINY_CLIP_MODEL = "hf-internal-testing/tiny-random-CLIPModel"


def clip_smoke_overrides(data_root: Path, work_dir: Path) -> list[str]:
    """Milestone 1 smoke settings plus a tiny CLIP and tiny heads / suite."""
    return smoke_overrides(data_root, work_dir) + [
        f"clip.model_name={TINY_CLIP_MODEL}", "clip.batch_size=8", "clip.shard_size=16", "clip.fp16=false",
        "clip.corruptions.blur=[2]", "clip.corruptions.noise=[0.1]", "clip.corruptions.word_drop=[0.5]",
        "head.epochs=2", "head.batch_size=16", "head.hidden=16", "head.xattn_dim=16", "head.xattn_heads=2",
        "clip_suite.masks=[none,strict]", "clip_suite.fusion_heads=[concat,gated,xattn]",
        "clip_suite.missing_md=[0.3]", "clip_suite.frac_heads=[image,xattn]", "clip_suite.fracs=[0.5]",
    ]

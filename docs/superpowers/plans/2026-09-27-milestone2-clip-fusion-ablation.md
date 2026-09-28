# Mốc 2 (CLIP features, fusion heads, ablations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract frozen CLIP ViT-B/16 features once (pooled + 16 tokens per modality, clean and corrupted), train light fusion heads (image, text, late, concat, gated, cross-attention) on them, and produce the main / missing-modality / robustness / data-fraction tables, plots and t-SNE.

**Architecture:** New sub-package `src/foodmm/clip/` (corrupt, features, encoder, zero_shot, heads, train_heads, report) plus thin scripts. Features are stored as `.npy` directories under `work_dir/clip/features/<tag>/`, runs under `work_dir/clip/runs/`, results under `work_dir/clip/results/`. Reuses Milestone 1 config, manifest, text masking, metrics, late-fusion math and the LR scheduler. Notebooks 02–04 share `tools/nb_utils.py`.

**Tech Stack:** as Milestone 1 (`transformers==5.17.0` `CLIPModel`), scikit-learn t-SNE. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-27-milestone2-clip-fusion-ablation-design.md`
**Depends on:** Milestone 1 plan fully implemented (config, prepare, text_utils, metrics, late_fusion, engine, analysis, tests/helpers).

## Global Constraints

- Same conventions as Milestone 1: `--config` + `--set KEY=VALUE` (last option), code in English, notebook text in Vietnamese, tests with `.venv/bin/pytest`, commit after every task.
- Feature set names: `image_{split}`, `text_{mask}_{split}`, `image_test_{corr}` (`blur1`, `noise0.05`, ...), `text_{main_mask}_test_{corr}` (`drop0.25`, ...). Level formatting uses `f"{float(x):g}"`.
- A feature set directory holds `ids.npy`, `pooled.npy` (float16, L2-normalised, `[N, E]`), `tokens.npy` (float16, `[N, 16, D]`), `token_mask.npy` (text only, bool `[N, 16]`) and `done.json`.
- Every head returns a dict with `logits` and `features` (and `gate` `[B]` for `gated`). Batch dict keys: `img`, `txt`, `img_tok`, `txt_tok`, `txt_mask`, `label`.
- Run names: `image`, `text_<mask>`, `<head>_<mask>`, suffix `_md<p:g>` when `head.modality_dropout != head.default_modality_dropout` (fusion heads only), suffix `_frac<f:g>` when `head.train_frac < 1`; `late_<mask>`; `zeroshot`.
- `metrics_test.json` keys: `run, modality, head, text_mask, modality_dropout, train_frac, acc, top5, macro_f1, n, best_val_acc` (+ `mean_gate` for gated, `w` for late). `modality` equals `head`. `text_mask` is `"-"` for `image` and `zeroshot`.
- `metrics_robust.json`: list of `{condition, acc, top5, macro_f1, n}`; `preds_robust.npz`: one float16 logits array per condition plus `ids`, `labels`.
- Tests marked `network` download `hf-internal-testing/tiny-random-CLIPModel` (~0.6 MB weights).

## File Map

| File | Responsibility |
|---|---|
| `configs/default.yaml` (append) | `clip`, `head`, `clip_suite` sections |
| `src/foodmm/clip/corrupt.py` | per-sample seeds, blur / noise / word drop, corruption specs |
| `src/foodmm/clip/features.py` | paths, feature-set names, sharded writer, loader |
| `src/foodmm/clip/encoder.py` | `chunk_pool`, `ClipEncoder` |
| `src/foodmm/clip/zero_shot.py` | prompts, zero-shot logits, `run_zero_shot` |
| `src/foodmm/clip/heads.py` | modality dropping, 5 heads, `build_head` |
| `src/foodmm/clip/train_heads.py` | `FeatureBank`, loading, run names, fit / predict, robustness, late fusion, `train_head_run` |
| `src/foodmm/clip/report.py` | result collection, tables, plots, t-SNE |
| `scripts/extract_clip.py`, `zero_shot_clip.py`, `train_head.py`, `run_clip_suite.py`, `summarize_clip.py` | CLI |
| `tools/nb_utils.py`, `tools/build_notebook_m2.py`, `notebooks/02_milestone2.ipynb` | notebook |
| `tests/helpers.py` (append), `tests/test_clip_*.py`, `tests/test_notebooks.py` | tests |

---

### Task 1: Config sections and corruptions

**Files:**
- Modify: `configs/default.yaml` (append)
- Create: `src/foodmm/clip/__init__.py`, `src/foodmm/clip/corrupt.py`
- Test: `tests/test_clip_corrupt.py`

**Interfaces:**
- Produces: `fmt_level(x) -> str`, `sample_seed(seed, sample_id) -> int`, `blur_image(img, radius)`, `noise_image(img, std, seed)`, `word_drop(text, p, seed) -> str`, `corruption_specs(cfg) -> list[tuple[name, kind, level]]` (`kind` in `blur|noise|drop`), `IMAGE_KINDS = ("blur", "noise")`, `apply_image_corruption(img, kind, level, seed)`.

- [ ] **Step 1: Append to `configs/default.yaml`**

```yaml

clip:
  model_name: openai/clip-vit-base-patch16
  tag: vitb16
  batch_size: 256
  n_tokens: 16             # image: 4x4 pooled patch grid; text: 16 contiguous chunks
  fp16: true               # only used on CUDA
  main_mask: strict        # text mask used by the missing/robustness/fraction ablations
  prompt: "a photo of {}, a type of food"
  shard_size: 2048
  corruptions:
    blur: [1, 2, 4]        # Gaussian blur radius (pixels)
    noise: [0.05, 0.1, 0.2]  # Gaussian noise std on [0, 1] pixels
    word_drop: [0.25, 0.5, 0.75]

head:
  name: concat             # image | text | concat | gated | xattn
  hidden: 512
  dropout: 0.3
  modality_dropout: 0.1
  default_modality_dropout: 0.1
  train_frac: 1.0
  epochs: 30
  batch_size: 256
  lr: 1.0e-3
  weight_decay: 0.01
  warmup_ratio: 0.05
  label_smoothing: 0.1
  patience: 3
  xattn_dim: 256
  xattn_heads: 4
  xattn_layers: 1

clip_suite:
  masks: [none, exact, strict]
  fusion_heads: [concat, gated, xattn]
  missing_md: [0.0, 0.3]
  frac_heads: [image, text, concat, xattn]
  fracs: [0.1, 0.25, 0.5]
```

- [ ] **Step 2: Write the failing test `tests/test_clip_corrupt.py`**

```python
import numpy as np
from PIL import Image

from foodmm.clip.corrupt import (
    apply_image_corruption, blur_image, corruption_specs, fmt_level, noise_image, sample_seed, word_drop,
)
from foodmm.config import load_config


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8))


def test_fmt_level_and_seed():
    assert fmt_level(1) == "1" and fmt_level(0.05) == "0.05" and fmt_level(2.0) == "2"
    assert sample_seed(42, "a") == sample_seed(42, "a") != sample_seed(42, "b")


def test_blur_and_noise_are_deterministic():
    img = _img()
    b = np.asarray(blur_image(img, 2), dtype=float)
    assert b.shape == (20, 20, 3) and b.std() < np.asarray(img, dtype=float).std()
    n1, n2 = noise_image(img, 0.1, seed=3), noise_image(img, 0.1, seed=3)
    assert np.array_equal(np.asarray(n1), np.asarray(n2))
    assert not np.array_equal(np.asarray(n1), np.asarray(img))
    same = apply_image_corruption(img, "noise", 0.1, 3)
    assert np.array_equal(np.asarray(same), np.asarray(n1))


def test_word_drop():
    text = " ".join(f"w{i}" for i in range(200))
    out = word_drop(text, 0.5, seed=1)
    assert out == word_drop(text, 0.5, seed=1)
    assert 60 < len(out.split()) < 140
    assert len(word_drop("one two", 0.999, seed=0).split()) == 1  # never empty
    assert word_drop("", 0.5, seed=0) == ""


def test_corruption_specs_from_default_config():
    specs = corruption_specs(load_config())
    names = [s[0] for s in specs]
    assert names == ["blur1", "blur2", "blur4", "noise0.05", "noise0.1", "noise0.2",
                     "drop0.25", "drop0.5", "drop0.75"]
    assert specs[0][1:] == ("blur", 1.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_corrupt.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip'`.

- [ ] **Step 4: Implement `src/foodmm/clip/__init__.py`**

```python
"""Frozen CLIP features, fusion heads and ablations (Milestone 2)."""
```

- [ ] **Step 5: Implement `src/foodmm/clip/corrupt.py`**

```python
"""Deterministic image / text corruptions used by the robustness ablation."""
from __future__ import annotations

import zlib

import numpy as np
from PIL import Image, ImageFilter

IMAGE_KINDS = ("blur", "noise")


def fmt_level(x: float) -> str:
    return f"{float(x):g}"


def sample_seed(seed: int, sample_id: str) -> int:
    return (int(seed) + zlib.crc32(str(sample_id).encode("utf-8"))) % (2 ** 32)


def blur_image(img: Image.Image, radius: float) -> Image.Image:
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))


def noise_image(img: Image.Image, std: float, seed: int) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    arr = np.clip(arr + rng.normal(0.0, float(std), arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def apply_image_corruption(img: Image.Image, kind: str, level: float, seed: int) -> Image.Image:
    if kind == "blur":
        return blur_image(img, level)
    if kind == "noise":
        return noise_image(img, level, seed)
    raise ValueError(f"unknown image corruption {kind!r}")


def word_drop(text: str, p: float, seed: int) -> str:
    """Drop each word with probability p; always keep at least one word of a non-empty text."""
    words = str(text).split()
    if not words:
        return ""
    rng = np.random.default_rng(seed)
    keep = rng.random(len(words)) >= float(p)
    if not keep.any():
        keep[rng.integers(len(words))] = True
    return " ".join(w for w, k in zip(words, keep) if k)


def corruption_specs(cfg: dict) -> list[tuple[str, str, float]]:
    """[(name, kind, level)] in config order: blur*, noise*, drop*."""
    c = cfg["clip"]["corruptions"]
    specs = [(f"blur{fmt_level(v)}", "blur", float(v)) for v in c.get("blur", [])]
    specs += [(f"noise{fmt_level(v)}", "noise", float(v)) for v in c.get("noise", [])]
    specs += [(f"drop{fmt_level(v)}", "drop", float(v)) for v in c.get("word_drop", [])]
    return specs
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_corrupt.py tests/test_config.py -v`
Expected: 10 passed.

- [ ] **Step 7: Commit**

```bash
git add configs/default.yaml src/foodmm/clip/__init__.py src/foodmm/clip/corrupt.py tests/test_clip_corrupt.py
git commit -m "feat(clip): config sections and deterministic corruptions"
```

---

### Task 2: Feature store (sharded writer and loader)

**Files:**
- Create: `src/foodmm/clip/features.py`
- Test: `tests/test_clip_features.py`

**Interfaces:**
- Consumes: `work_paths` (M1), `fmt_level`, `save_json`, `load_json`.
- Produces: `clip_paths(cfg) -> dict[root, features, runs, results]` (`features` = `work_dir/clip/features/<clip.tag>`), `image_set_name(split, corruption=None)`, `text_set_name(mask, split, corruption=None)`, `class FeatureWriter(out_dir, ids, shard_size)` with `.is_done()`, `.pending_shards() -> list[tuple[int, np.ndarray]]`, `.write_shard(index, arrays: dict)`, `.finalize() -> int`, `is_done(out_dir) -> bool`, `load_feature_set(out_dir, keys=None) -> dict[str, ndarray]` (always includes `ids`), `check_ids(actual, expected, what)` (raises `ValueError`), `class MissingFeatures(FileNotFoundError)`.

- [ ] **Step 1: Write the failing test `tests/test_clip_features.py`**

```python
import numpy as np
import pytest

from foodmm.clip.features import (
    FeatureWriter, MissingFeatures, check_ids, clip_paths, image_set_name, is_done, load_feature_set,
    text_set_name,
)
from foodmm.config import load_config


def test_names_and_paths(tmp_path):
    cfg = load_config(overrides=[f"paths.work_dir={tmp_path}"])
    p = clip_paths(cfg)
    assert p["features"] == tmp_path / "clip" / "features" / "vitb16"
    assert p["runs"] == tmp_path / "clip" / "runs" and p["results"] == tmp_path / "clip" / "results"
    assert image_set_name("train") == "image_train"
    assert image_set_name("test", "blur2") == "image_test_blur2"
    assert text_set_name("strict", "val") == "text_strict_val"
    assert text_set_name("strict", "test", "drop0.5") == "text_strict_test_drop0.5"


def _arrays(idx):
    return {"pooled": np.full((len(idx), 4), idx[:, None], dtype=np.float16),
            "tokens": np.zeros((len(idx), 2, 3), dtype=np.float16)}


def test_writer_resume_and_finalize(tmp_path):
    ids = [f"id{i}" for i in range(10)]
    out = tmp_path / "fs"
    w = FeatureWriter(out, ids, shard_size=4)
    pending = w.pending_shards()
    assert [i for i, _ in pending] == [0, 1, 2] and pending[2][1].tolist() == [8, 9]
    w.write_shard(0, _arrays(pending[0][1]))
    w2 = FeatureWriter(out, ids, shard_size=4)  # "restart"
    assert [i for i, _ in w2.pending_shards()] == [1, 2]
    for i, idx in w2.pending_shards():
        w2.write_shard(i, _arrays(idx))
    assert w2.finalize() == 10 and is_done(out) and w2.is_done()
    assert not (out / "shards").exists()
    fs = load_feature_set(out)
    assert fs["ids"].tolist() == ids
    assert fs["pooled"][:, 0].tolist() == list(range(10))
    only = load_feature_set(out, keys=["pooled"])
    assert set(only) == {"ids", "pooled"}


def test_finalize_requires_all_shards(tmp_path):
    w = FeatureWriter(tmp_path / "fs", ["a", "b", "c"], shard_size=2)
    w.write_shard(0, _arrays(np.array([0, 1])))
    with pytest.raises(RuntimeError, match="missing shards"):
        w.finalize()


def test_load_missing_raises(tmp_path):
    with pytest.raises(MissingFeatures, match="extract_clip.py"):
        load_feature_set(tmp_path / "nope")


def test_check_ids():
    check_ids(np.array(["a", "b"]), np.array(["a", "b"]), "x")
    with pytest.raises(ValueError, match="x"):
        check_ids(np.array(["b", "a"]), np.array(["a", "b"]), "x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_features.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.features'`.

- [ ] **Step 3: Implement `src/foodmm/clip/features.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_features.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/foodmm/clip/features.py tests/test_clip_features.py
git commit -m "feat(clip): resumable sharded feature store"
```

---

### Task 3: CLIP encoder and zero-shot math

**Files:**
- Create: `src/foodmm/clip/encoder.py`, `src/foodmm/clip/zero_shot.py`
- Modify: `tests/helpers.py` (append `TINY_CLIP_MODEL`, `clip_smoke_overrides`)
- Test: `tests/test_clip_encoder.py`

**Interfaces:**
- Produces (`encoder`): `chunk_pool(hidden [B,L,D], attention_mask [B,L], n) -> (tokens [B,n,D], mask [B,n] bool)`, `grid_pool(patches [B,P,D], n) -> [B,n,D]`, `class ClipEncoder(model_name, device=None, fp16=True, n_tokens=16)` with attributes `processor`, `image_processor`, `embed_dim`, `image_dim`, `text_dim`, `n_tokens`, `max_len`, and methods `preprocess(img) -> Tensor[3,H,W]`, `encode_pixels(pixel_values) -> (pooled f16 [N,E], tokens f16 [N,n,Di])`, `encode_images(list[PIL]) -> same`, `encode_texts(list[str]) -> (pooled f16 [N,E], tokens f16 [N,n,Dt], mask bool [N,n])`.
- Produces (`zero_shot`, math part): `class_prompts(classes, template) -> list[str]`, `zero_shot_logits(image_pooled, prompt_embeds, scale=100.0) -> ndarray`.
- Produces (`tests/helpers.py`): `TINY_CLIP_MODEL = "hf-internal-testing/tiny-random-CLIPModel"`, `clip_smoke_overrides(data_root, work_dir) -> list[str]` (M1 smoke overrides + tiny CLIP/head/suite settings).

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
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
```

- [ ] **Step 2: Write the failing test `tests/test_clip_encoder.py`**

```python
import numpy as np
import pytest
import torch
from PIL import Image

from foodmm.clip.encoder import chunk_pool, grid_pool
from foodmm.clip.zero_shot import class_prompts, zero_shot_logits
from helpers import TINY_CLIP_MODEL


def test_chunk_pool_splits_valid_tokens():
    hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1).repeat(2, 1, 1)
    attn = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]])
    tokens, mask = chunk_pool(hidden, attn, 3)
    assert tokens.shape == (2, 3, 1)
    assert tokens[0, :, 0].tolist() == [0.5, 2.5, 4.5]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, True, False]  # 2 valid tokens -> chunks 0 and 1
    assert tokens[1, 2, 0].item() == 0.0


def test_grid_pool():
    patches = torch.ones(2, 9, 5)
    assert grid_pool(patches, 4).shape == (2, 4, 5)
    with pytest.raises(ValueError):
        grid_pool(patches, 5)


def test_zero_shot_math():
    prompts = class_prompts(["apple_pie", "french_fries"], "a photo of {}, a type of food")
    assert prompts == ["a photo of apple pie, a type of food", "a photo of french fries, a type of food"]
    img = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float16)
    txt = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = zero_shot_logits(img, txt)
    assert logits.shape == (2, 2) and logits.argmax(1).tolist() == [0, 1]
    assert logits[1, 1] == pytest.approx(100.0)


@pytest.mark.network
def test_clip_encoder_shapes():
    from foodmm.clip.encoder import ClipEncoder

    enc = ClipEncoder(TINY_CLIP_MODEL, device=torch.device("cpu"), n_tokens=16)
    imgs = [Image.new("RGB", (40, 32), (i * 50, 100, 30)) for i in range(3)]
    pooled, tokens = enc.encode_images(imgs)
    assert pooled.shape == (3, enc.embed_dim) and tokens.shape == (3, 16, enc.image_dim)
    assert pooled.dtype == np.float16
    assert np.allclose(np.linalg.norm(pooled.astype(np.float32), axis=1), 1.0, atol=1e-2)
    tp, tt, tm = enc.encode_texts(["apple pie with cream and a long list of words " * 3, "", "fries"])
    assert tp.shape == (3, enc.embed_dim) and tt.shape == (3, 16, enc.text_dim) and tm.shape == (3, 16)
    assert tm[0].all() and tm[1].sum() < 16
    assert enc.preprocess(imgs[0]).shape[0] == 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_encoder.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.encoder'`.

- [ ] **Step 4: Implement `src/foodmm/clip/encoder.py`**

```python
"""Frozen CLIP encoder returning pooled embeddings and a fixed number of tokens per modality."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def chunk_pool(hidden: torch.Tensor, attention_mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each sequence's valid tokens into n contiguous chunks and mean-pool each chunk."""
    b, length, _ = hidden.shape
    valid = attention_mask.bool()
    lengths = valid.sum(1).clamp(min=1)
    pos = torch.arange(length, device=hidden.device).unsqueeze(0).expand(b, length)
    chunk = (pos * n) // lengths.unsqueeze(1)
    chunk = torch.where(valid, chunk, torch.full_like(chunk, n))
    onehot = F.one_hot(chunk, n + 1)[..., :n].to(hidden.dtype)  # [B, L, n]
    sums = torch.einsum("bln,bld->bnd", onehot, hidden)
    counts = onehot.sum(1)  # [B, n]
    return sums / counts.clamp(min=1).unsqueeze(-1), counts > 0


def grid_pool(patches: torch.Tensor, n: int) -> torch.Tensor:
    """[B, P, D] square patch grid -> [B, n, D] by adaptive average pooling (n must be a square)."""
    side_out = int(round(math.sqrt(n)))
    if side_out * side_out != n:
        raise ValueError(f"n_tokens must be a perfect square, got {n}")
    b, p, d = patches.shape
    side = int(round(math.sqrt(p)))
    grid = patches.transpose(1, 2).reshape(b, d, side, side)
    return F.adaptive_avg_pool2d(grid, side_out).flatten(2).transpose(1, 2)


class ClipEncoder:
    def __init__(self, model_name: str, device: torch.device | None = None, fp16: bool = True, n_tokens: int = 16):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device, self.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_processor = self.processor.image_processor
        cfg = self.model.config
        self.embed_dim = int(cfg.projection_dim)
        self.image_dim = int(cfg.vision_config.hidden_size)
        self.text_dim = int(cfg.text_config.hidden_size)
        self.max_len = int(min(77, cfg.text_config.max_position_embeddings))
        self.n_tokens = int(n_tokens)
        grid_pool(torch.zeros(1, 4, 1), self.n_tokens)  # validate n_tokens early

    def preprocess(self, img) -> torch.Tensor:
        return self.image_processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def encode_pixels(self, pixel_values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        out = self.model.vision_model(pixel_values=pixel_values.to(self.device, self.dtype))
        pooled = F.normalize(self.model.visual_projection(out.pooler_output).float(), dim=-1)
        tokens = grid_pool(out.last_hidden_state[:, 1:].float(), self.n_tokens)
        return pooled.cpu().numpy().astype(np.float16), tokens.cpu().numpy().astype(np.float16)

    def encode_images(self, images: Sequence) -> tuple[np.ndarray, np.ndarray]:
        return self.encode_pixels(torch.stack([self.preprocess(im) for im in images]))

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enc = self.processor.tokenizer([str(t) for t in texts], padding=True, truncation=True,
                                       max_length=self.max_len, return_tensors="pt")
        ids, attn = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        out = self.model.text_model(input_ids=ids, attention_mask=attn)
        pooled = F.normalize(self.model.text_projection(out.pooler_output).float(), dim=-1)
        tokens, mask = chunk_pool(out.last_hidden_state.float(), attn, self.n_tokens)
        return (pooled.cpu().numpy().astype(np.float16), tokens.cpu().numpy().astype(np.float16),
                mask.cpu().numpy())
```

- [ ] **Step 5: Implement `src/foodmm/clip/zero_shot.py` (math part; `run_zero_shot` is appended in Task 4)**

```python
"""CLIP zero-shot classification from cached image embeddings."""
from __future__ import annotations

import shutil
from collections.abc import Sequence

import numpy as np

from ..data.text_utils import class_to_phrase


def class_prompts(classes: Sequence[str], template: str) -> list[str]:
    return [template.format(class_to_phrase(c)) for c in classes]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def zero_shot_logits(image_pooled: np.ndarray, prompt_embeds: np.ndarray, scale: float = 100.0) -> np.ndarray:
    return scale * _normalize(image_pooled) @ _normalize(prompt_embeds).T
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_encoder.py -v`
Expected: 4 passed (the `network` test downloads the tiny CLIP).

- [ ] **Step 7: Commit**

```bash
git add src/foodmm/clip/encoder.py src/foodmm/clip/zero_shot.py tests/helpers.py tests/test_clip_encoder.py
git commit -m "feat(clip): frozen CLIP encoder with token pooling and zero-shot math"
```

---

### Task 4: Feature extraction and zero-shot scripts

**Files:**
- Create: `src/foodmm/clip/extract.py`, `scripts/extract_clip.py`, `scripts/zero_shot_clip.py`
- Modify: `src/foodmm/clip/zero_shot.py` (append `run_zero_shot`)
- Test: `tests/test_clip_extract.py`

**Interfaces:**
- Consumes: `load_manifest` (M1), `text_column` (M1), `corruption_specs`, `apply_image_corruption`, `word_drop`, `sample_seed`, `IMAGE_KINDS` (Task 1), `FeatureWriter`, `clip_paths`, names (Task 2), `ClipEncoder` (Task 3), `compute_metrics` (M1).
- Produces: `PARTS = ("image", "text_none", "text_exact", "text_strict", "corrupt")`, `@dataclass Job(name, kind, split, column, corruption)`, `build_jobs(cfg, parts, splits) -> list[Job]`, `run_job(job, df, encoder, cfg, out_dir) -> dict[name, status, n?, bad_images?]`, `run_extraction(cfg, parts, splits, force=False, encoder=None) -> list[dict]`, `run_zero_shot(cfg, force=False, encoder=None) -> dict`.
- CLI: `extract_clip.py [--parts a,b] [--splits train,val,test] [--force] --set ...` (prints `skip <name>` for finished sets); `zero_shot_clip.py [--force] --set ...` → run `zeroshot` (prints `already finished` when done).

- [ ] **Step 1: Write the failing test `tests/test_clip_extract.py`**

```python
import json

import numpy as np
import pytest

from foodmm.clip.extract import build_jobs
from foodmm.config import load_config
from helpers import clip_smoke_overrides, make_fake_dataset, run_script


def test_build_jobs():
    cfg = load_config()
    jobs = build_jobs(cfg, ["image", "text_strict", "corrupt"], ["train", "test"])
    names = [j.name for j in jobs]
    assert names[:4] == ["image_train", "image_test", "text_strict_train", "text_strict_test"]
    assert "image_test_blur2" in names and "text_strict_test_drop0.5" in names
    blur = next(j for j in jobs if j.name == "image_test_blur2")
    assert blur.kind == "image" and blur.split == "test" and blur.corruption == ("blur", 2.0)
    drop = next(j for j in jobs if j.name == "text_strict_test_drop0.5")
    assert drop.column == "text_strict" and drop.corruption == ("drop", 0.5)
    with pytest.raises(ValueError):
        build_jobs(cfg, ["bogus"], ["train"])


@pytest.mark.network
def test_extract_and_zero_shot_scripts(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work)
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_strict,corrupt", "--set", *sets)
    feats = work / "clip" / "features" / "vitb16"
    for name in ("image_train", "image_val", "image_test", "text_strict_test", "image_test_blur2",
                 "image_test_noise0.1", "text_strict_test_drop0.5"):
        assert (feats / name / "done.json").exists(), name
    tokens = np.load(feats / "text_strict_train" / "tokens.npy")
    assert tokens.shape[1] == 16 and np.load(feats / "text_strict_train" / "token_mask.npy").shape == tokens.shape[:2]
    assert "skip image_train" in run_script("extract_clip.py", "--parts", "image", "--set", *sets).stdout

    run_script("zero_shot_clip.py", "--set", *sets)
    m = json.loads((work / "clip" / "runs" / "zeroshot" / "metrics_test.json").read_text())
    assert m["run"] == "zeroshot" and m["n"] == 12 and 0 <= m["acc"] <= 1
    assert "already finished" in run_script("zero_shot_clip.py", "--set", *sets).stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_extract.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.extract'`.

- [ ] **Step 3: Implement `src/foodmm/clip/extract.py`**

```python
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
```

- [ ] **Step 4: Append to `src/foodmm/clip/extract.py` (running jobs)**

```python
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
```

- [ ] **Step 5: Append to `src/foodmm/clip/zero_shot.py`**

```python
def run_zero_shot(cfg: dict, force: bool = False, encoder=None) -> dict:
    from ..config import save_config, work_paths
    from ..data.prepare import load_manifest
    from ..metrics import compute_metrics
    from ..utils import ensure_dir, load_json, save_json
    from .encoder import ClipEncoder
    from .features import check_ids, clip_paths, image_set_name, load_feature_set

    cp, paths = clip_paths(cfg), work_paths(cfg)
    run_dir = cp["runs"] / "zeroshot"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "metrics_test.json").exists():
        print("zeroshot is already finished (use --force to rerun)")
        return load_json(run_dir / "metrics_test.json")
    classes = load_json(paths["classes"])
    test = load_manifest(paths["manifest"]).query("split == 'test'")
    fs = load_feature_set(cp["features"] / image_set_name("test"), keys=["pooled"])
    check_ids(fs["ids"], test["id"].to_numpy(), image_set_name("test"))
    cc = cfg["clip"]
    encoder = encoder or ClipEncoder(cc["model_name"], fp16=bool(cc["fp16"]), n_tokens=int(cc["n_tokens"]))
    prompts = class_prompts(classes, cc["prompt"])
    prompt_embeds = np.concatenate([encoder.encode_texts(prompts[i:i + 64])[0] for i in range(0, len(prompts), 64)])
    logits = zero_shot_logits(fs["pooled"], prompt_embeds).astype(np.float32)
    labels = test["label_idx"].to_numpy()
    ensure_dir(run_dir)
    np.savez(run_dir / "preds_test.npz", logits=logits, labels=labels, ids=test["id"].to_numpy().astype(str))
    cfg["run"] = {"name": "zeroshot", "head": "zeroshot"}
    save_config(cfg, run_dir / "config.yaml")
    metrics = compute_metrics(logits, labels)
    metrics.update({"run": "zeroshot", "modality": "zeroshot", "head": "zeroshot", "text_mask": "-",
                    "modality_dropout": None, "train_frac": 1.0, "best_val_acc": None})
    save_json(metrics, run_dir / "metrics_test.json")
    return metrics
```

- [ ] **Step 6: Implement `scripts/extract_clip.py`**

```python
#!/usr/bin/env python
"""Extract frozen CLIP features (pooled + tokens) for clean and corrupted inputs; resumable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.extract import PARTS, run_extraction  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default=",".join(PARTS))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    results = run_extraction(cfg, args.parts.split(","), args.splits.split(","), force=args.force)
    print(f"{sum(r['status'] == 'done' for r in results)} extracted, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Implement `scripts/zero_shot_clip.py`**

```python
#!/usr/bin/env python
"""CLIP zero-shot baseline on the test split (prompt from clip.prompt)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.zero_shot import run_zero_shot  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    print(json.dumps(run_zero_shot(config_from_args(args), force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_extract.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add src/foodmm/clip/extract.py src/foodmm/clip/zero_shot.py scripts/extract_clip.py scripts/zero_shot_clip.py tests/test_clip_extract.py
git commit -m "feat(clip): resumable feature extraction and zero-shot baseline"
```

---

### Task 5: Fusion heads

**Files:**
- Create: `src/foodmm/clip/heads.py`
- Test: `tests/test_clip_heads.py`

**Interfaces:**
- Produces: `MODAL_HEADS = ("image", "text")`, `FUSION_HEADS = ("concat", "gated", "xattn")`, `HEADS`, `TOKEN_HEADS = ("xattn",)`, `head_inputs(name) -> tuple[use_img, use_txt, use_tokens]`, `masked_mean(x, mask)`, `random_modality_drop(n, p, generator=None) -> (drop_img, drop_txt)` (bool `[n]`, never both), `drop_modalities(batch, drop_img, drop_txt) -> dict` (new dict; zeroes `img`/`img_tok` or `txt`/`txt_tok` and `txt_mask`), `ModalHead`, `ConcatHead`, `GatedHead`, `CrossAttnHead`, `build_head(name, dims, num_classes, hcfg)`; `dims` keys `img, txt, img_tok, txt_tok, n_tokens`.

- [ ] **Step 1: Write the failing test `tests/test_clip_heads.py`**

```python
import pytest
import torch

from foodmm.clip.heads import (
    HEADS, build_head, drop_modalities, head_inputs, masked_mean, random_modality_drop,
)
from foodmm.config import load_config

DIMS = {"img": 8, "txt": 8, "img_tok": 6, "txt_tok": 5, "n_tokens": 4}


def _batch(b=3):
    g = torch.Generator().manual_seed(0)
    return {"img": torch.randn(b, 8, generator=g), "txt": torch.randn(b, 8, generator=g),
            "img_tok": torch.randn(b, 4, 6, generator=g), "txt_tok": torch.randn(b, 4, 5, generator=g),
            "txt_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.float32)[:b],
            "label": torch.zeros(b, dtype=torch.long)}


def _hcfg(**kw):
    h = dict(load_config()["head"])
    h.update({"hidden": 16, "xattn_dim": 8, "xattn_heads": 2, **kw})
    return h


@pytest.mark.parametrize("name", HEADS)
def test_heads_output_shapes(name):
    model = build_head(name, DIMS, 7, _hcfg()).eval()
    out = model(_batch())
    assert out["logits"].shape == (3, 7)
    assert out["features"].shape[0] == 3 and out["features"].ndim == 2
    if name == "gated":
        assert out["gate"].shape == (3,) and ((out["gate"] >= 0) & (out["gate"] <= 1)).all()


def test_build_head_rejects_unknown():
    with pytest.raises(ValueError):
        build_head("bilinear", DIMS, 7, _hcfg())


def test_head_inputs():
    assert head_inputs("image") == (True, False, False)
    assert head_inputs("text") == (False, True, False)
    assert head_inputs("gated") == (True, True, False)
    assert head_inputs("xattn") == (True, True, True)


def test_masked_mean_all_zero_mask_gives_zero():
    x = torch.ones(2, 3, 4)
    out = masked_mean(x, torch.tensor([[1.0, 0, 0], [0, 0, 0]]))
    assert out[0].tolist() == [1.0] * 4 and out[1].tolist() == [0.0] * 4


def test_drop_modalities():
    b = _batch()
    out = drop_modalities(b, torch.tensor([True, False, False]), torch.tensor([False, True, False]))
    assert out["img"][0].abs().sum() == 0 and out["img_tok"][0].abs().sum() == 0
    assert out["txt"][1].abs().sum() == 0 and out["txt_mask"][1].sum() == 0 and out["txt_tok"][1].abs().sum() == 0
    assert torch.equal(out["img"][1], b["img"][1]) and torch.equal(out["txt"][0], b["txt"][0])
    assert b["img"][0].abs().sum() > 0  # input not modified


def test_random_modality_drop_never_both():
    g = torch.Generator().manual_seed(0)
    di, dt = random_modality_drop(10000, 0.5, g)
    assert not (di & dt).any()
    assert 0.2 < di.float().mean() < 0.3 and 0.2 < dt.float().mean() < 0.3
    di0, dt0 = random_modality_drop(10, 0.0, g)
    assert not di0.any() and not dt0.any()


def test_xattn_ignores_padded_text_tokens():
    model = build_head("xattn", DIMS, 7, _hcfg()).eval()
    b = _batch()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["txt_tok"][0, 2:] = 99.0  # padded positions of sample 0
    with torch.no_grad():
        assert torch.allclose(model(b)["logits"][0], model(b2)["logits"][0], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_heads.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.heads'`.

- [ ] **Step 3: Implement `src/foodmm/clip/heads.py` (part 1: helpers and simple heads)**

```python
"""Light classification heads on frozen CLIP features."""
from __future__ import annotations

import torch
from torch import nn

MODAL_HEADS = ("image", "text")
FUSION_HEADS = ("concat", "gated", "xattn")
HEADS = MODAL_HEADS + FUSION_HEADS
TOKEN_HEADS = ("xattn",)


def head_inputs(name: str) -> tuple[bool, bool, bool]:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}; expected one of {HEADS}")
    return name != "text", name != "image", name in TOKEN_HEADS


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def random_modality_drop(n: int, p: float, generator: torch.Generator | None = None):
    drop = torch.rand(n, generator=generator) < p
    pick_img = torch.rand(n, generator=generator) < 0.5
    return drop & pick_img, drop & ~pick_img


def drop_modalities(batch: dict, drop_img: torch.Tensor, drop_txt: torch.Tensor) -> dict:
    """Zero the dropped modality per sample; keys that are absent (unimodal banks) are ignored."""
    out = dict(batch)
    keep = {"img": ~drop_img.bool(), "txt": ~drop_txt.bool()}
    for key, which in (("img", "img"), ("img_tok", "img"), ("txt", "txt"), ("txt_tok", "txt"), ("txt_mask", "txt")):
        if key in out:
            v = out[key]
            k = keep[which].to(v.device)
            out[key] = v * k.view(-1, *([1] * (v.ndim - 1))).to(v.dtype)
    return out


def _proj(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())


class ModalHead(nn.Module):
    """MLP probe on one pooled embedding (`key` = "img" or "txt")."""

    def __init__(self, key: str, in_dim: int, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.key = key
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, batch: dict) -> dict:
        f = self.body(batch[self.key])
        return {"logits": self.out(f), "features": f}


class ConcatHead(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.proj_i, self.proj_t = _proj(img_dim, hidden), _proj(txt_dim, hidden)
        self.fuse = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, batch: dict) -> dict:
        f = self.fuse(torch.cat([self.proj_i(batch["img"]), self.proj_t(batch["txt"])], dim=1))
        return {"logits": self.out(f), "features": f}


class GatedHead(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.proj_i, self.proj_t = _proj(img_dim, hidden), _proj(txt_dim, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, batch: dict) -> dict:
        hi, ht = self.proj_i(batch["img"]), self.proj_t(batch["txt"])
        g = torch.sigmoid(self.gate(torch.cat([hi, ht], dim=1)))
        f = g * hi + (1.0 - g) * ht
        return {"logits": self.out(f), "features": f, "gate": g.mean(dim=1)}
```

- [ ] **Step 4: Append to `src/foodmm/clip/heads.py` (part 2: cross-attention and factory)**

```python
class CrossAttnBlock(nn.Module):
    """Pre-norm block: text tokens (queries) attend to image tokens (keys/values)."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.ln_q(q), kv, kv, need_weights=False)
        q = q + a
        return q + self.ffn(self.ln_ffn(q))


class CrossAttnHead(nn.Module):
    def __init__(self, dims: dict, num_classes: int, hidden: int, dropout: float,
                 dim: int = 256, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        n = int(dims["n_tokens"])
        self.tok_i = nn.Sequential(nn.LayerNorm(dims["img_tok"]), nn.Linear(dims["img_tok"], dim))
        self.tok_t = nn.Sequential(nn.LayerNorm(dims["txt_tok"]), nn.Linear(dims["txt_tok"], dim))
        self.pos_i = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.ln_kv = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([CrossAttnBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.proj_i, self.proj_t = _proj(dims["img"], hidden), _proj(dims["txt"], hidden)
        self.fuse = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim + 2 * hidden, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, batch: dict) -> dict:
        kv = self.ln_kv(self.tok_i(batch["img_tok"]) + self.pos_i)
        q = self.tok_t(batch["txt_tok"]) + self.pos_t
        for block in self.blocks:
            q = block(q, kv)
        z = masked_mean(q, batch["txt_mask"])
        f = self.fuse(torch.cat([z, self.proj_i(batch["img"]), self.proj_t(batch["txt"])], dim=1))
        return {"logits": self.out(f), "features": f}


def build_head(name: str, dims: dict, num_classes: int, hcfg: dict) -> nn.Module:
    head_inputs(name)  # validates the name
    hidden, dropout = int(hcfg["hidden"]), float(hcfg["dropout"])
    if name == "image":
        return ModalHead("img", dims["img"], num_classes, hidden, dropout)
    if name == "text":
        return ModalHead("txt", dims["txt"], num_classes, hidden, dropout)
    if name == "concat":
        return ConcatHead(dims["img"], dims["txt"], num_classes, hidden, dropout)
    if name == "gated":
        return GatedHead(dims["img"], dims["txt"], num_classes, hidden, dropout)
    return CrossAttnHead(dims, num_classes, hidden, dropout, dim=int(hcfg["xattn_dim"]),
                         n_heads=int(hcfg["xattn_heads"]), n_layers=int(hcfg["xattn_layers"]))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_heads.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/foodmm/clip/heads.py tests/test_clip_heads.py
git commit -m "feat(clip): image/text/concat/gated/cross-attention heads with modality dropping"
```

---

### Task 6: Head training, robustness evaluation and late fusion

**Files:**
- Create: `src/foodmm/clip/train_heads.py`
- Test: `tests/test_clip_train.py`

**Interfaces:**
- Consumes: M1 `save_config`, `work_paths`, `load_manifest`, `build_scheduler`, `check_aligned`, `combine`, `load_preds`, `search_weight`, `compute_metrics`, `softmax`, utils; Tasks 1, 2, 5.
- Produces: `@dataclass FeatureBank(ids, labels, arrays)` with `__len__`, `subset(idx)`, `batch(idx, device) -> dict`, `dims() -> dict`; `load_bank(cfg, df, split, head, text_mask, image_corruption=None, text_corruption=None) -> FeatureBank`; `run_name_for(cfg) -> str`; `stratified_fraction(labels, frac, seed) -> ndarray`; `fit_head(model, train, val, hcfg, device, seed, use_md) -> (state_dict, history, best_val_acc)`; `predict_head(model, bank, device, batch_size=1024, drop_img=False, drop_txt=False) -> dict[logits f32, features f16, gate?]`; `robust_conditions(cfg, head, text_mask) -> list[dict]`; `evaluate_robust(model, cfg, df, head, text_mask, device, test_bank) -> (rows, logits_by_condition)`; `train_head_run(cfg, force=False, device=None) -> dict`; `run_late(cfg, text_mask, force=False) -> dict`.

- [ ] **Step 1: Write the failing test `tests/test_clip_train.py`**

```python
import json

import numpy as np
import pytest

from foodmm.clip.features import FeatureWriter, MissingFeatures, clip_paths, image_set_name, text_set_name
from foodmm.clip.train_heads import run_late, run_name_for, stratified_fraction, train_head_run
from foodmm.config import load_config, set_by_path, work_paths
from foodmm.data.prepare import load_manifest
from helpers import clip_smoke_overrides, make_fake_dataset, run_script

E, DI, DT, N_TOK = 8, 6, 5, 16


def _write(cfg, name, ids, arrays):
    w = FeatureWriter(clip_paths(cfg)["features"] / name, ids, shard_size=64)
    for i, idx in w.pending_shards():
        w.write_shard(i, {k: v[idx] for k, v in arrays.items()})
    w.finalize()


def _fake_features(cfg, df, rng):
    """Class-dependent pooled features so heads can learn; tokens are noise."""
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        ids, y = sub["id"].tolist(), sub["label_idx"].to_numpy()
        n = len(ids)
        signal = np.zeros((n, E), dtype=np.float32)
        signal[np.arange(n), y] = 3.0
        img = {"pooled": (signal + rng.normal(0, 0.5, (n, E))).astype(np.float16),
               "tokens": rng.normal(0, 1, (n, N_TOK, DI)).astype(np.float16)}
        mask = np.zeros((n, N_TOK), dtype=bool)
        mask[:, :4] = True
        txt = {"pooled": (signal + rng.normal(0, 0.5, (n, E))).astype(np.float16),
               "tokens": rng.normal(0, 1, (n, N_TOK, DT)).astype(np.float16), "token_mask": mask}
        _write(cfg, image_set_name(split), ids, img)
        for m in ("none", "strict"):
            _write(cfg, text_set_name(m, split), ids, txt)
        if split == "test":
            _write(cfg, image_set_name("test", "blur2"), ids, img)
            _write(cfg, image_set_name("test", "noise0.1"), ids, img)
            _write(cfg, text_set_name("strict", "test", "drop0.5"), ids, txt)


@pytest.fixture()
def env(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work) + ["head.epochs=20", "head.lr=0.01", "head.patience=20"]
    run_script("prepare_data.py", "--set", *sets)
    cfg = load_config(overrides=sets)
    _fake_features(cfg, load_manifest(work_paths(cfg)["manifest"]), np.random.default_rng(0))
    return cfg


def _with(cfg, **kv):
    import copy

    c = copy.deepcopy(cfg)
    for k, v in kv.items():
        set_by_path(c, k.replace("__", "."), v)
    return c


def test_run_name_for():
    cfg = load_config()
    assert run_name_for(_with(cfg, head__name="image", data__text_mask="strict")) == "image"
    assert run_name_for(_with(cfg, head__name="text", data__text_mask="none")) == "text_none"
    assert run_name_for(_with(cfg, head__name="xattn", data__text_mask="strict")) == "xattn_strict"
    assert run_name_for(_with(cfg, head__name="gated", head__modality_dropout=0.0)) == "gated_none_md0"
    assert run_name_for(_with(cfg, head__name="concat", data__text_mask="strict", head__train_frac=0.25)) == \
        "concat_strict_frac0.25"
    assert run_name_for(_with(cfg, head__name="text", head__modality_dropout=0.3)) == "text_none"


def test_stratified_fraction():
    labels = np.repeat(np.arange(4), [10, 10, 10, 1])
    idx = stratified_fraction(labels, 0.3, seed=0)
    counts = np.bincount(labels[idx], minlength=4)
    assert counts.tolist() == [3, 3, 3, 1]
    assert np.array_equal(idx, stratified_fraction(labels, 0.3, seed=0)) and np.all(np.diff(idx) > 0)


def test_train_heads_robustness_and_late(env):
    cfg = env
    runs = clip_paths(cfg)["runs"]
    m_img = train_head_run(_with(cfg, head__name="image"))
    assert m_img["run"] == "image" and m_img["text_mask"] == "-" and m_img["acc"] >= 0.5
    rob = json.loads((runs / "image" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob] == ["full", "blur2", "noise0.1"]

    for mask in ("none", "strict"):
        train_head_run(_with(cfg, head__name="text", data__text_mask=mask))
    rob_none = json.loads((runs / "text_none" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob_none] == ["full"]  # word drop only exists for the main mask

    m_x = train_head_run(_with(cfg, head__name="xattn", data__text_mask="strict"))
    assert m_x["modality_dropout"] == 0.1 and m_x["train_frac"] == 1.0
    rob_x = json.loads((runs / "xattn_strict" / "metrics_robust.json").read_text())
    assert [r["condition"] for r in rob_x] == ["full", "no_image", "no_text", "blur2", "noise0.1", "drop0.5"]
    with np.load(runs / "xattn_strict" / "preds_test.npz") as z:
        assert z["logits"].shape == (12, 3) and z["features"].dtype == np.float16
    with np.load(runs / "xattn_strict" / "preds_robust.npz") as z:
        assert {"full", "no_image", "drop0.5", "ids", "labels"} <= set(z.files)

    m_g = train_head_run(_with(cfg, head__name="gated", data__text_mask="strict", head__train_frac=0.5))
    assert m_g["run"] == "gated_strict_frac0.5" and 0.0 <= m_g["mean_gate"] <= 1.0

    late = run_late(cfg, "strict")
    assert late["run"] == "late_strict" and 0.0 <= late["w"] <= 1.0
    rob_late = json.loads((runs / "late_strict" / "metrics_robust.json").read_text())
    assert {r["condition"] for r in rob_late} == {"full", "no_image", "no_text", "blur2", "noise0.1", "drop0.5"}
    assert train_head_run(_with(cfg, head__name="image"))["acc"] == m_img["acc"]  # already finished


def test_missing_features_and_runs(env):
    with pytest.raises(MissingFeatures):
        train_head_run(_with(env, head__name="text", data__text_mask="exact"))
    with pytest.raises(SystemExit, match="image"):
        run_late(env, "none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_train.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.train_heads'`.

- [ ] **Step 3: Implement `src/foodmm/clip/train_heads.py` (part 1: data)**

```python
"""Train / evaluate heads on cached CLIP features; robustness conditions; late fusion."""
from __future__ import annotations

import copy
import math
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import save_config, work_paths
from ..data.prepare import load_manifest
from ..engine import build_scheduler
from ..late_fusion import check_aligned, combine, load_preds, search_weight
from ..metrics import compute_metrics, softmax
from ..utils import ensure_dir, get_device, load_json, save_json, seed_everything
from .corrupt import IMAGE_KINDS, corruption_specs, fmt_level
from .features import MissingFeatures, check_ids, clip_paths, image_set_name, load_feature_set, text_set_name
from .heads import FUSION_HEADS, build_head, drop_modalities, head_inputs, random_modality_drop


@dataclass
class FeatureBank:
    ids: np.ndarray
    labels: np.ndarray
    arrays: dict  # subset of img, txt, img_tok, txt_tok, txt_mask (float16 / bool numpy arrays)

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, idx: np.ndarray) -> "FeatureBank":
        return FeatureBank(self.ids[idx], self.labels[idx], {k: v[idx] for k, v in self.arrays.items()})

    def batch(self, idx: np.ndarray, device: torch.device) -> dict:
        out = {k: torch.from_numpy(np.asarray(v[idx], dtype=np.float32)).to(device) for k, v in self.arrays.items()}
        out["label"] = torch.from_numpy(self.labels[idx].astype(np.int64)).to(device)
        return out

    def dims(self) -> dict:
        a, d = self.arrays, {}
        for key in ("img", "txt"):
            if key in a:
                d[key] = int(a[key].shape[1])
        for key in ("img_tok", "txt_tok"):
            if key in a:
                d[key], d["n_tokens"] = int(a[key].shape[2]), int(a[key].shape[1])
        return d


def load_bank(cfg: dict, df: pd.DataFrame, split: str, head: str, text_mask: str,
              image_corruption: str | None = None, text_corruption: str | None = None) -> FeatureBank:
    use_img, use_txt, use_tok = head_inputs(head)
    sub = df[df["split"] == split]
    ids, labels = sub["id"].astype(str).to_numpy(), sub["label_idx"].to_numpy()
    root = clip_paths(cfg)["features"]
    arrays: dict = {}
    if use_img:
        name = image_set_name(split, image_corruption)
        fs = load_feature_set(root / name, ["pooled"] + (["tokens"] if use_tok else []))
        check_ids(fs["ids"], ids, name)
        arrays["img"] = fs["pooled"]
        if use_tok:
            arrays["img_tok"] = fs["tokens"]
    if use_txt:
        name = text_set_name(text_mask, split, text_corruption)
        fs = load_feature_set(root / name, ["pooled"] + (["tokens", "token_mask"] if use_tok else []))
        check_ids(fs["ids"], ids, name)
        arrays["txt"] = fs["pooled"]
        if use_tok:
            arrays["txt_tok"], arrays["txt_mask"] = fs["tokens"], fs["token_mask"]
    return FeatureBank(ids, labels, arrays)


def run_name_for(cfg: dict) -> str:
    h = cfg["head"]
    name, mask = h["name"], cfg["data"]["text_mask"]
    head_inputs(name)
    run = "image" if name == "image" else f"{name}_{mask}"
    md = float(h["modality_dropout"])
    if name in FUSION_HEADS and abs(md - float(h["default_modality_dropout"])) > 1e-12:
        run += f"_md{fmt_level(md)}"
    if float(h["train_frac"]) < 1.0:
        run += f"_frac{fmt_level(h['train_frac'])}"
    return run


def stratified_fraction(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(1, int(round(frac * len(idx))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))
```

- [ ] **Step 4: Append to `src/foodmm/clip/train_heads.py` (part 2: fit / predict / robustness)**

```python
@torch.no_grad()
def predict_head(model: nn.Module, bank: FeatureBank, device: torch.device, batch_size: int = 1024,
                 drop_img: bool = False, drop_txt: bool = False) -> dict:
    model.eval()
    logits, feats, gates = [], [], []
    for start in range(0, len(bank), batch_size):
        idx = np.arange(start, min(len(bank), start + batch_size))
        b = bank.batch(idx, device)
        if drop_img or drop_txt:
            b = drop_modalities(b, torch.full((len(idx),), drop_img), torch.full((len(idx),), drop_txt))
        out = model(b)
        logits.append(out["logits"].float().cpu().numpy())
        feats.append(out["features"].float().cpu().numpy().astype(np.float16))
        if "gate" in out:
            gates.append(out["gate"].float().cpu().numpy())
    res = {"logits": np.concatenate(logits), "features": np.concatenate(feats)}
    if gates:
        res["gate"] = np.concatenate(gates)
    return res


def fit_head(model: nn.Module, train: FeatureBank, val: FeatureBank, hcfg: dict, device: torch.device,
             seed: int, use_md: bool) -> tuple[dict, list[dict], float]:
    bs, epochs = int(hcfg["batch_size"]), int(hcfg["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(hcfg["lr"]), weight_decay=float(hcfg["weight_decay"]))
    sched = build_scheduler(opt, epochs * math.ceil(len(train) / bs), float(hcfg["warmup_ratio"]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(hcfg["label_smoothing"]))
    md = float(hcfg["modality_dropout"]) if use_md else 0.0
    g = torch.Generator().manual_seed(int(seed))
    best_acc, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        t0, model_losses = time.time(), []
        model.train()
        perm = torch.randperm(len(train), generator=g).numpy()
        for start in range(0, len(train), bs):
            idx = perm[start:start + bs]
            b = train.batch(idx, device)
            if md > 0:
                di, dt = random_modality_drop(len(idx), md, g)
                b = drop_modalities(b, di, dt)
            loss = loss_fn(model(b)["logits"], b["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model_losses.append(float(loss))
        val_acc = float((predict_head(model, val, device)["logits"].argmax(1) == val.labels).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(model_losses)), "val_acc": val_acc,
                        "seconds": round(time.time() - t0, 2)})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc, best_state, bad = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(hcfg["patience"]):
                break
    return best_state, history, best_acc


def robust_conditions(cfg: dict, head: str, text_mask: str) -> list[dict]:
    use_img, use_txt, _ = head_inputs(head)
    conds = [{"condition": "full"}]
    if use_img and use_txt:
        conds += [{"condition": "no_image", "drop_img": True}, {"condition": "no_text", "drop_txt": True}]
    for name, kind, _level in corruption_specs(cfg):
        if kind in IMAGE_KINDS and use_img:
            conds.append({"condition": name, "image_corruption": name})
        elif kind == "drop" and use_txt and text_mask == cfg["clip"]["main_mask"]:
            conds.append({"condition": name, "text_corruption": name})
    return conds


def evaluate_robust(model, cfg, df, head, text_mask, device, test_bank) -> tuple[list[dict], dict]:
    rows, logits = [], {}
    for c in robust_conditions(cfg, head, text_mask):
        bank = test_bank
        if c.get("image_corruption") or c.get("text_corruption"):
            try:
                bank = load_bank(cfg, df, "test", head, text_mask, c.get("image_corruption"), c.get("text_corruption"))
            except MissingFeatures:
                print(f"skip condition {c['condition']}: features not extracted", flush=True)
                continue
        out = predict_head(model, bank, device, drop_img=c.get("drop_img", False), drop_txt=c.get("drop_txt", False))
        rows.append({"condition": c["condition"], **compute_metrics(out["logits"], bank.labels)})
        logits[c["condition"]] = out["logits"]
    return rows, logits
```

- [ ] **Step 5: Append to `src/foodmm/clip/train_heads.py` (part 3: runs)**

```python
def _prepare_run(run_dir, force: bool, name: str):
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "metrics_test.json").exists():
        print(f"{name} is already finished (use --force to retrain)", flush=True)
        return load_json(run_dir / "metrics_test.json")
    return None


def train_head_run(cfg: dict, force: bool = False, device: torch.device | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    name = run_name_for(cfg)
    run_dir = clip_paths(cfg)["runs"] / name
    done = _prepare_run(run_dir, force, name)
    if done is not None:
        return done
    paths = work_paths(cfg)
    if not paths["manifest"].exists():
        raise SystemExit(f"Missing {paths['manifest']}: run scripts/prepare_data.py first")
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = device or get_device()
    df, classes = load_manifest(paths["manifest"]), load_json(paths["classes"])
    hc = cfg["head"]
    head, mask, frac = hc["name"], cfg["data"]["text_mask"], float(hc["train_frac"])
    banks = {s: load_bank(cfg, df, s, head, mask) for s in ("train", "val", "test")}
    if frac < 1.0:
        banks["train"] = banks["train"].subset(stratified_fraction(banks["train"].labels, frac, seed))
    dims = banks["train"].dims()
    model = build_head(head, dims, len(classes), hc).to(device)
    print(f"{name}: {len(banks['train'])} train samples, "
          f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)
    state, history, best_acc = fit_head(model, banks["train"], banks["val"], hc, device, seed,
                                        use_md=head in FUSION_HEADS)
    model.load_state_dict(state)
    ensure_dir(run_dir)
    torch.save({"model": state, "dims": dims, "head": head, "num_classes": len(classes)}, run_dir / "best.pt")
    save_json(history, run_dir / "history.json")
    cfg["run"] = {"name": name, "head": head}
    save_config(cfg, run_dir / "config.yaml")
    outs = {}
    for s in ("val", "test"):
        outs[s] = predict_head(model, banks[s], device)
        np.savez(run_dir / f"preds_{s}.npz", logits=outs[s]["logits"], labels=banks[s].labels,
                 ids=banks[s].ids, features=outs[s]["features"])
    metrics = compute_metrics(outs["test"]["logits"], banks["test"].labels)
    metrics.update({"run": name, "modality": head, "head": head, "text_mask": "-" if head == "image" else mask,
                    "modality_dropout": float(hc["modality_dropout"]) if head in FUSION_HEADS else None,
                    "train_frac": frac, "best_val_acc": best_acc})
    if "gate" in outs["test"]:
        metrics["mean_gate"] = float(outs["test"]["gate"].mean())
    rows, rob = evaluate_robust(model, cfg, df, head, mask, device, banks["test"])
    save_json(rows, run_dir / "metrics_robust.json")
    np.savez(run_dir / "preds_robust.npz", ids=banks["test"].ids, labels=banks["test"].labels,
             **{k: v.astype(np.float16) for k, v in rob.items()})
    save_json(metrics, run_dir / "metrics_test.json")  # written last: marks the run as finished
    print(f"{name}: test acc={metrics['acc']:.4f}", flush=True)
    return metrics


def _load_robust(run_dir) -> dict[str, np.ndarray]:
    with np.load(run_dir / "preds_robust.npz") as z:
        return {k: softmax(z[k].astype(np.float32)) for k in z.files if k not in ("ids", "labels")}


def run_late(cfg: dict, text_mask: str, force: bool = False) -> dict:
    runs = clip_paths(cfg)["runs"]
    img_dir, txt_dir = runs / "image", runs / f"text_{text_mask}"
    for d in (img_dir, txt_dir):
        if not (d / "metrics_test.json").exists():
            raise SystemExit(f"Missing run '{d.name}': train it first (scripts/train_head.py)")
    name = f"late_{text_mask}"
    run_dir = runs / name
    done = _prepare_run(run_dir, force, name)
    if done is not None:
        return done
    preds = {(m, s): load_preds(d, s) for m, d in (("img", img_dir), ("txt", txt_dir)) for s in ("val", "test")}
    for s in ("val", "test"):
        check_aligned(preds[("img", s)], preds[("txt", s)])
    probs = {k: softmax(v["logits"].astype(np.float32)) for k, v in preds.items()}
    w, curve = search_weight(probs[("img", "val")], probs[("txt", "val")], preds[("img", "val")]["labels"],
                             step=float(cfg["late_fusion"]["w_step"]))
    ensure_dir(run_dir)
    fused = {}
    for s in ("val", "test"):
        fused[s] = combine(probs[("img", s)], probs[("txt", s)], w)
        np.savez(run_dir / f"preds_{s}.npz", logits=np.log(fused[s] + 1e-12).astype(np.float32),
                 labels=preds[("img", s)]["labels"], ids=preds[("img", s)]["ids"])
    p_img, p_txt = probs[("img", "test")], probs[("txt", "test")]
    conds = {"full": fused["test"], "no_image": p_txt, "no_text": p_img}
    for k, v in _load_robust(img_dir).items():
        if k != "full":
            conds[k] = combine(v, p_txt, w)
    for k, v in _load_robust(txt_dir).items():
        if k != "full":
            conds[k] = combine(p_img, v, w)
    labels = preds[("img", "test")]["labels"]
    rows = [{"condition": k, **compute_metrics(v, labels)} for k, v in conds.items()]
    save_json(rows, run_dir / "metrics_robust.json")
    np.savez(run_dir / "preds_robust.npz", ids=preds[("img", "test")]["ids"], labels=labels,
             **{k: np.log(v + 1e-12).astype(np.float16) for k, v in conds.items()})
    save_json(curve, run_dir / "weight_curve.json")
    cfg = copy.deepcopy(cfg)
    cfg["data"]["text_mask"] = text_mask
    cfg["run"] = {"name": name, "head": "late"}
    save_config(cfg, run_dir / "config.yaml")
    metrics = compute_metrics(fused["test"], labels)
    metrics.update({"run": name, "modality": "late", "head": "late", "text_mask": text_mask,
                    "modality_dropout": None, "train_frac": 1.0,
                    "best_val_acc": max(r["acc"] for r in curve), "w": w})
    save_json(metrics, run_dir / "metrics_test.json")
    return metrics
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_train.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/foodmm/clip/train_heads.py tests/test_clip_train.py
git commit -m "feat(clip): head training, robustness evaluation and late fusion runs"
```

---

### Task 7: Run scripts and the experiment suite

**Files:**
- Create: `src/foodmm/clip/suite.py`, `scripts/train_head.py`, `scripts/run_clip_suite.py`
- Test: `tests/test_clip_suite.py`

**Interfaces:**
- Consumes: `train_head_run`, `run_late` (Task 6), `run_zero_shot` (Task 4), `set_by_path`, `parse_value` (M1).
- Produces: `STAGES = ("main", "missing", "frac", "all")`, `suite_jobs(cfg, stage) -> list[tuple[kind, overrides]]` (`kind` in `zeroshot|head|late`), `job_config(cfg, overrides) -> dict`, `run_suite(cfg, stage, force=False, skip_zeroshot=False) -> pandas.DataFrame[kind, run, acc]`.
- CLI: `train_head.py --head {image,text,concat,gated,xattn,late} [--force] --set ...`; `run_clip_suite.py --stage {main,missing,frac,all} [--skip_zeroshot] [--force] --set ...` (prints the final table).

- [ ] **Step 1: Write the failing test `tests/test_clip_suite.py`**

```python
import json

from foodmm.clip.features import clip_paths
from foodmm.clip.suite import job_config, run_suite, suite_jobs
from foodmm.config import load_config
from helpers import run_script
from test_clip_train import env  # noqa: F401  (fixture with a fake feature store)


def test_suite_jobs_default_counts():
    cfg = load_config()
    assert len(suite_jobs(cfg, "main")) == 17
    assert len(suite_jobs(cfg, "missing")) == 6
    assert len(suite_jobs(cfg, "frac")) == 12
    assert len(suite_jobs(cfg, "all")) == 35
    assert suite_jobs(cfg, "main")[0] == ("zeroshot", [])
    kind, ov = suite_jobs(cfg, "missing")[0]
    c = job_config(cfg, ov)
    assert kind == "head" and c["head"]["name"] == "concat" and c["head"]["modality_dropout"] == 0.0
    assert c["data"]["text_mask"] == "strict" and cfg["head"]["modality_dropout"] == 0.1  # base untouched


def test_run_suite_on_fake_features(env):  # noqa: F811
    cfg = job_config(env, ["head.epochs=2"])
    table = run_suite(cfg, "all", skip_zeroshot=True)
    runs = set(table["run"])
    assert {"image", "text_none", "text_strict", "concat_none", "gated_strict", "xattn_strict",
            "late_none", "late_strict", "xattn_strict_md0.3", "image_frac0.5", "xattn_strict_frac0.5"} <= runs
    assert len(table) == 16  # 11 main (zero-shot skipped) + 3 missing + 2 fraction runs
    rob = json.loads((clip_paths(cfg)["runs"] / "gated_strict_md0.3" / "metrics_robust.json").read_text())
    assert rob[0]["condition"] == "full"


def test_train_head_script(env, tmp_path):  # noqa: F811
    work = env["paths"]["work_dir"]
    sets = [f"paths.data_root={env['paths']['data_root']}", f"paths.work_dir={work}", "head.epochs=1",
            "head.hidden=16", "head.batch_size=16", "data.text_mask=strict",
            "clip.corruptions.blur=[2]", "clip.corruptions.noise=[0.1]", "clip.corruptions.word_drop=[0.5]"]
    run_script("train_head.py", "--head", "image", "--set", *sets)
    run_script("train_head.py", "--head", "text", "--set", *sets)
    out = run_script("train_head.py", "--head", "late", "--set", *sets).stdout
    assert '"run": "late_strict"' in out
    assert "already finished" in run_script("train_head.py", "--head", "image", "--set", *sets).stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_suite.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.suite'`.

- [ ] **Step 3: Implement `src/foodmm/clip/suite.py`**

```python
"""The Milestone 2 experiment list: main runs, missing-modality and data-fraction ablations."""
from __future__ import annotations

import copy

import pandas as pd

from ..config import parse_value, set_by_path
from .train_heads import run_late, train_head_run
from .zero_shot import run_zero_shot

STAGES = ("main", "missing", "frac", "all")


def suite_jobs(cfg: dict, stage: str) -> list[tuple[str, list[str]]]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    s, main = cfg["clip_suite"], cfg["clip"]["main_mask"]
    jobs: list[tuple[str, list[str]]] = []
    if stage in ("main", "all"):
        jobs.append(("zeroshot", []))
        jobs.append(("head", ["head.name=image"]))
        jobs += [("head", ["head.name=text", f"data.text_mask={m}"]) for m in s["masks"]]
        jobs += [("head", [f"head.name={h}", f"data.text_mask={m}"]) for m in s["masks"] for h in s["fusion_heads"]]
        jobs += [("late", [f"data.text_mask={m}"]) for m in s["masks"]]
    if stage in ("missing", "all"):
        jobs += [("head", [f"head.name={h}", f"data.text_mask={main}", f"head.modality_dropout={md}"])
                 for h in s["fusion_heads"] for md in s["missing_md"]]
    if stage in ("frac", "all"):
        jobs += [("head", [f"head.name={h}", f"data.text_mask={main}", f"head.train_frac={f}"])
                 for h in s["frac_heads"] for f in s["fracs"]]
    return jobs


def job_config(cfg: dict, overrides: list[str]) -> dict:
    c = copy.deepcopy(cfg)
    for item in overrides:
        key, raw = item.split("=", 1)
        set_by_path(c, key, parse_value(raw))
    return c


def run_suite(cfg: dict, stage: str, force: bool = False, skip_zeroshot: bool = False) -> pd.DataFrame:
    rows = []
    jobs = suite_jobs(cfg, stage)
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
```

- [ ] **Step 4: Implement `scripts/train_head.py`**

```python
#!/usr/bin/env python
"""Train one head on cached CLIP features (or compute late fusion with --head late)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.heads import HEADS  # noqa: E402
from foodmm.clip.train_heads import run_late, train_head_run  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, choices=[*HEADS, "late"])
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if args.head == "late":
        metrics = run_late(cfg, cfg["data"]["text_mask"], force=args.force)
    else:
        cfg["head"]["name"] = args.head
        metrics = train_head_run(cfg, force=args.force)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement `scripts/run_clip_suite.py`**

```python
#!/usr/bin/env python
"""Run every Milestone 2 experiment of a stage; finished runs are skipped."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.clip.suite import STAGES, run_suite  # noqa: E402
from foodmm.config import add_config_args, config_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--skip_zeroshot", action="store_true")
    parser.add_argument("--force", action="store_true", help="retrain every run of the stage")
    add_config_args(parser)
    args = parser.parse_args(argv)
    table = run_suite(config_from_args(args), args.stage, force=args.force, skip_zeroshot=args.skip_zeroshot)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_suite.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/foodmm/clip/suite.py scripts/train_head.py scripts/run_clip_suite.py tests/test_clip_suite.py
git commit -m "feat(clip): experiment suite and head training scripts"
```

---

### Task 8: Report tables, plots and `summarize_clip.py`

**Files:**
- Create: `src/foodmm/clip/report.py`, `scripts/summarize_clip.py`
- Test: `tests/test_clip_report.py`

**Interfaces:**
- Consumes: `MASK_ORDER` (M1 analysis), `load_json`, `clip_paths`.
- Produces: `HEAD_ORDER`, `CLIP_COLUMNS = ["run", "head", "text_mask", "modality_dropout", "train_frac", "acc", "top5", "macro_f1", "n"]`, `collect_clip_results(runs_dir) -> DataFrame`, `main_table(df, default_md) -> DataFrame`, `collect_robust(runs_dir) -> DataFrame[run, head, text_mask, modality_dropout, train_frac, condition, acc, top5, macro_f1]`, `missing_table(robust, mask) -> DataFrame[run, head, modality_dropout, full, no_image, no_text]`, `robustness_table(robust, mask, default_md) -> DataFrame[run, head, kind, level, acc]`, `fraction_table(df, mask, default_md) -> DataFrame[head, train_frac, acc, run]`, `table_to_markdown(df, pct_cols) -> str`, `plot_robustness(table)`, `plot_fraction(table)`, `sample_for_tsne(labels, n_classes=20, per_class=50, seed=0) -> ndarray`, `tsne_2d(x, seed=0, perplexity=30.0) -> ndarray`, `plot_tsne(emb, labels, classes, title="")`.
- CLI: `summarize_clip.py --set ...` → `clip/results/{main,missing,robust,frac}.csv`, `main.md`, `missing.md`, `robust.png`, `frac.png`; exits non-zero with `No finished runs` when empty.

- [ ] **Step 1: Write the failing test `tests/test_clip_report.py`**

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from foodmm.clip.report import (  # noqa: E402
    collect_clip_results, collect_robust, fraction_table, main_table, missing_table, plot_fraction,
    plot_robustness, plot_tsne, robustness_table, sample_for_tsne, table_to_markdown, tsne_2d,
)
from foodmm.utils import save_json  # noqa: E402
from helpers import run_script  # noqa: E402

RUNS = [  # run, head, mask, md, frac, acc
    ("zeroshot", "zeroshot", "-", None, 1.0, 0.5), ("image", "image", "-", None, 1.0, 0.6),
    ("text_strict", "text", "strict", None, 1.0, 0.4), ("late_strict", "late", "strict", None, 1.0, 0.65),
    ("xattn_strict", "xattn", "strict", 0.1, 1.0, 0.7), ("xattn_strict_md0.3", "xattn", "strict", 0.3, 1.0, 0.69),
    ("xattn_strict_frac0.5", "xattn", "strict", 0.1, 0.5, 0.6), ("image_frac0.5", "image", "-", None, 0.5, 0.5),
]


def _write(runs_dir):
    for run, head, mask, md, frac, acc in RUNS:
        save_json({"run": run, "modality": head, "head": head, "text_mask": mask, "modality_dropout": md,
                   "train_frac": frac, "acc": acc, "top5": 0.9, "macro_f1": acc, "n": 10}, runs_dir / run / "metrics_test.json")
        if head == "zeroshot":
            continue
        conds = [("full", acc), ("blur2", acc - 0.1), ("noise0.1", acc - 0.2)]
        if head in ("xattn", "late"):
            conds += [("no_image", 0.3), ("no_text", acc - 0.05), ("drop0.5", acc - 0.05)]
        save_json([{"condition": c, "acc": a, "top5": 0.9, "macro_f1": a, "n": 10} for c, a in conds],
                  runs_dir / run / "metrics_robust.json")


def test_tables(tmp_path):
    _write(tmp_path)
    df = collect_clip_results(tmp_path)
    assert df["run"].tolist()[:4] == ["zeroshot", "image", "image_frac0.5", "text_strict"]
    main = main_table(df, 0.1)
    assert main["run"].tolist() == ["zeroshot", "image", "text_strict", "late_strict", "xattn_strict"]
    rob = collect_robust(tmp_path)
    miss = missing_table(rob, "strict")
    assert miss["run"].tolist() == ["late_strict", "xattn_strict", "xattn_strict_md0.3"]
    assert miss.set_index("run").loc["xattn_strict", "no_image"] == 0.3
    rt = robustness_table(rob, "strict", 0.1)
    blur = rt[(rt["run"] == "image") & (rt["kind"] == "blur")].sort_values("level")
    assert blur["level"].tolist() == [0.0, 2.0] and blur["acc"].round(2).tolist() == [0.6, 0.5]
    assert set(rt["run"]) == {"image", "text_strict", "late_strict", "xattn_strict"}
    fr = fraction_table(df, "strict", 0.1)
    assert fr[fr["head"] == "xattn"]["train_frac"].tolist() == [0.5, 1.0]
    assert set(fr["head"]) == {"image", "xattn"}
    md = table_to_markdown(main, ["acc", "top5", "macro_f1"])
    assert md.splitlines()[0].startswith("| run |") and "| 70.00 |" in md


def test_plots_and_tsne():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(25), 8)
    idx = sample_for_tsne(labels, n_classes=20, per_class=5, seed=0)
    assert len(idx) == 100 and len(np.unique(labels[idx])) == 20
    emb = tsne_2d(rng.normal(size=(len(idx), 6)), seed=0)
    assert emb.shape == (100, 2)
    import pandas as pd

    rob = pd.DataFrame({"run": ["a", "a"], "head": ["image"] * 2, "kind": ["blur"] * 2, "level": [0.0, 2.0], "acc": [0.6, 0.5]})
    fr = pd.DataFrame({"head": ["image", "image"], "train_frac": [0.5, 1.0], "acc": [0.5, 0.6], "run": ["x", "y"]})
    figs = [plot_robustness(rob), plot_fraction(fr), plot_tsne(emb, labels[idx], [f"c{i}" for i in range(25)], "t")]
    assert all(isinstance(f, Figure) for f in figs)
    plt.close("all")


def test_summarize_clip_script(tmp_path):
    work = tmp_path / "work"
    res = run_script("summarize_clip.py", "--set", f"paths.work_dir={work}", check=False)
    assert res.returncode != 0 and "No finished runs" in res.stderr
    _write(work / "clip" / "runs")
    run_script("summarize_clip.py", "--set", f"paths.work_dir={work}")
    out = work / "clip" / "results"
    for name in ("main.csv", "main.md", "missing.csv", "missing.md", "robust.csv", "robust.png", "frac.csv", "frac.png"):
        assert (out / name).exists(), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clip_report.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.clip.report'`.

- [ ] **Step 3: Implement `src/foodmm/clip/report.py` (part 1: tables)**

```python
"""Milestone 2 result tables, plots and t-SNE helpers."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import MASK_ORDER
from ..utils import load_json

HEAD_ORDER = {"zeroshot": 0, "image": 1, "text": 2, "late": 3, "concat": 4, "gated": 5, "xattn": 6}
CLIP_COLUMNS = ["run", "head", "text_mask", "modality_dropout", "train_frac", "acc", "top5", "macro_f1", "n"]
_COND = re.compile(r"^(blur|noise|drop)([\d.]+)$")


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame({
        "h": df["head"].map(HEAD_ORDER).fillna(99), "m": df["text_mask"].map(MASK_ORDER).fillna(9),
        "md": pd.to_numeric(df["modality_dropout"], errors="coerce").fillna(-1.0),
        "f": -pd.to_numeric(df["train_frac"], errors="coerce").fillna(1.0), "r": df["run"],
    })
    order = keys.sort_values(["h", "m", "md", "f", "r"]).index
    return df.loc[order].reset_index(drop=True)


def collect_clip_results(runs_dir: str | Path) -> pd.DataFrame:
    rows = [load_json(p) for p in sorted(Path(runs_dir).glob("*/metrics_test.json"))]
    if not rows:
        return pd.DataFrame(columns=CLIP_COLUMNS)
    return _sort(pd.DataFrame(rows).reindex(columns=CLIP_COLUMNS))


def _default_md(df: pd.DataFrame, default_md: float) -> pd.Series:
    md = pd.to_numeric(df["modality_dropout"], errors="coerce")
    return md.isna() | np.isclose(md.fillna(-1.0), default_md)


def main_table(df: pd.DataFrame, default_md: float) -> pd.DataFrame:
    keep = np.isclose(pd.to_numeric(df["train_frac"]), 1.0) & _default_md(df, default_md)
    return df[keep].reset_index(drop=True)


def collect_robust(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir).glob("*/metrics_robust.json")):
        meta_path = p.parent / "metrics_test.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        for r in load_json(p):
            rows.append({**{k: meta.get(k) for k in CLIP_COLUMNS[:5]}, "condition": r["condition"],
                         "acc": r["acc"], "top5": r["top5"], "macro_f1": r["macro_f1"]})
    cols = CLIP_COLUMNS[:5] + ["condition", "acc", "top5", "macro_f1"]
    return pd.DataFrame(rows, columns=cols)


def missing_table(robust: pd.DataFrame, mask: str) -> pd.DataFrame:
    sub = robust[robust["head"].isin(["late", "concat", "gated", "xattn"]) & (robust["text_mask"] == mask)
                 & np.isclose(pd.to_numeric(robust["train_frac"]), 1.0)
                 & robust["condition"].isin(["full", "no_image", "no_text"])]
    if sub.empty:
        return pd.DataFrame(columns=["run", "head", "modality_dropout", "full", "no_image", "no_text"])
    wide = sub.pivot_table(index="run", columns="condition", values="acc").reset_index()
    meta = sub.drop_duplicates("run").set_index("run")
    wide["head"] = wide["run"].map(meta["head"])
    wide["modality_dropout"] = wide["run"].map(meta["modality_dropout"])
    wide["text_mask"], wide["train_frac"] = mask, 1.0
    out = _sort(wide)
    return out.reindex(columns=["run", "head", "modality_dropout", "full", "no_image", "no_text"])


def robustness_table(robust: pd.DataFrame, mask: str, default_md: float) -> pd.DataFrame:
    sub = robust[robust["text_mask"].isin([mask, "-"]) & np.isclose(pd.to_numeric(robust["train_frac"]), 1.0)
                 & _default_md(robust, default_md)]
    rows = []
    for run, g in sub.groupby("run", sort=False):
        full = g.loc[g["condition"] == "full", "acc"]
        kinds = {}
        for r in g.itertuples(index=False):
            m = _COND.match(r.condition)
            if m:
                kinds.setdefault(m.group(1), []).append((float(m.group(2)), r.acc))
        for kind, points in kinds.items():
            if len(full):
                points = [(0.0, float(full.iloc[0]))] + points
            rows += [{"run": run, "head": g["head"].iloc[0], "kind": kind, "level": lv, "acc": a} for lv, a in points]
    return pd.DataFrame(rows, columns=["run", "head", "kind", "level", "acc"])


def fraction_table(df: pd.DataFrame, mask: str, default_md: float) -> pd.DataFrame:
    frac = pd.to_numeric(df["train_frac"])
    heads = set(df.loc[frac < 1.0, "head"])
    sub = df[df["head"].isin(heads) & df["text_mask"].isin([mask, "-"]) & _default_md(df, default_md)]
    out = sub[["head", "train_frac", "acc", "run"]].copy()
    out["_h"] = out["head"].map(HEAD_ORDER)
    return out.sort_values(["_h", "train_frac"]).drop(columns="_h").reset_index(drop=True)


def table_to_markdown(df: pd.DataFrame, pct_cols: Sequence[str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif c in pct_cols:
                cells.append(f"{float(v) * 100:.2f}")
            else:
                cells.append(f"{v:g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Append to `src/foodmm/clip/report.py` (part 2: plots and t-SNE)**

```python
def plot_robustness(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    kinds = [k for k in ("blur", "noise", "drop") if k in set(table["kind"])] or ["blur"]
    titles = {"blur": "Ảnh bị làm mờ (bán kính)", "noise": "Ảnh thêm nhiễu (độ lệch chuẩn)",
              "drop": "Text bị bỏ từ (tỷ lệ)"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for run, g in table[table["kind"] == kind].groupby("run", sort=False):
            g = g.sort_values("level")
            ax.plot(g["level"], g["acc"] * 100, marker="o", label=run)
        ax.set_title(titles[kind])
        ax.set_xlabel("mức nhiễu (0 = sạch)")
        ax.set_ylabel("test acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_fraction(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for head, g in table.groupby("head", sort=False):
        ax.plot(g["train_frac"] * 100, g["acc"] * 100, marker="o", label=head)
    ax.set_xscale("log")
    ax.set_xlabel("dữ liệu train (%)")
    ax.set_ylabel("test acc (%)")
    ax.set_title("Accuracy theo lượng dữ liệu train")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def sample_for_tsne(labels: np.ndarray, n_classes: int = 20, per_class: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen = np.sort(rng.choice(classes, size=min(n_classes, len(classes)), replace=False))
    idx = []
    for c in chosen:
        pool = np.flatnonzero(labels == c)
        idx.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def tsne_2d(x: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE

    x = np.asarray(x, dtype=np.float32)
    perplexity = float(min(perplexity, max(2.0, (len(x) - 1) / 3.0)))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(x)


def plot_tsne(emb: np.ndarray, labels: np.ndarray, classes: Sequence[str], title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(i), label=str(classes[int(c)]).replace("_", " "))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Implement `scripts/summarize_clip.py`**

```python
#!/usr/bin/env python
"""Write the Milestone 2 tables and plots into work_dir/clip/results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from foodmm.clip.features import clip_paths  # noqa: E402
from foodmm.clip.report import (  # noqa: E402
    collect_clip_results, collect_robust, fraction_table, main_table, missing_table, plot_fraction,
    plot_robustness, robustness_table, table_to_markdown,
)
from foodmm.config import add_config_args, config_from_args  # noqa: E402
from foodmm.utils import ensure_dir  # noqa: E402

PCT = ["acc", "top5", "macro_f1", "full", "no_image", "no_text"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cp = clip_paths(cfg)
    df = collect_clip_results(cp["runs"])
    if df.empty:
        raise SystemExit(f"No finished runs found in {cp['runs']}")
    out = ensure_dir(cp["results"])
    mask, md = cfg["clip"]["main_mask"], float(cfg["head"]["default_modality_dropout"])
    robust = collect_robust(cp["runs"])
    tables = {"main": main_table(df, md), "missing": missing_table(robust, mask),
              "robust": robustness_table(robust, mask, md), "frac": fraction_table(df, mask, md)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    for name in ("main", "missing"):
        md_text = table_to_markdown(tables[name], PCT)
        (out / f"{name}.md").write_text(md_text + "\n", encoding="utf-8")
        print(f"\n## {name}\n{md_text}")
    plot_robustness(tables["robust"]).savefig(out / "robust.png", dpi=150)
    plot_fraction(tables["frac"]).savefig(out / "frac.png", dpi=150)
    print(f"\nWrote results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clip_report.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/foodmm/clip/report.py scripts/summarize_clip.py tests/test_clip_report.py
git commit -m "feat(clip): result tables, robustness/fraction plots and t-SNE helpers"
```

---

### Task 9: Shared notebook builder and notebook 02

**Files:**
- Create: `tools/nb_utils.py`, `tools/build_notebook_m2.py`, `notebooks/02_milestone2.ipynb` (generated)
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Produces: `NotebookBuilder` with `.md(src)`, `.code(src)`, `.build() -> dict`, `.write(path)`, `.main(default_out, argv=None) -> int`; `add_setup(nb)` (GPU check, mount Drive, clone/pull, `pip install`, `run()` helper; defines `REPO_DIR`, `WORK_DIR`, `DATA_ROOT`, `BASE`); `add_data(nb)` (Kaggle download → Drive zip → extract to `/content/data` → `prepare_data.py`, which skips when the manifest exists).
- `python tools/build_notebook_m2.py [--out PATH]` writes `notebooks/02_milestone2.ipynb` deterministically.
- `tests/test_notebooks.py` has a `NOTEBOOKS` list that later plans extend.

- [ ] **Step 1: Write the failing test `tests/test_notebooks.py`**

```python
import json
import subprocess
import sys

import pytest

from helpers import REPO_ROOT

NOTEBOOKS = [("build_notebook_m2.py", "02_milestone2.ipynb")]


@pytest.mark.parametrize("builder,name", NOTEBOOKS)
def test_notebook_is_generated_and_up_to_date(tmp_path, builder, name):
    out = tmp_path / name
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / builder), "--out", str(out)], check=True)
    assert out.read_text(encoding="utf-8") == (REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("builder,name", NOTEBOOKS)
def test_notebook_cells_compile_and_imports_resolve(builder, name):
    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) > 10
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = [ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("!", "%"))]
        compile("\n".join(lines), "<cell>", "exec")
        for ln in lines:
            if ln.startswith(("from foodmm", "import foodmm")):
                exec(ln, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebooks.py -v`
Expected: FAIL (`CalledProcessError`: `tools/build_notebook_m2.py` does not exist).

- [ ] **Step 3: Implement `tools/nb_utils.py`**

```python
"""Shared builder and setup cells for notebooks 02-04 (notebook 01 keeps tools/build_notebook.py)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


class NotebookBuilder:
    def __init__(self) -> None:
        self.cells: list[tuple[str, str]] = []

    def md(self, src: str) -> None:
        self.cells.append(("markdown", src.strip("\n")))

    def code(self, src: str) -> None:
        self.cells.append(("code", src.strip("\n")))

    def build(self) -> dict:
        cells = []
        for i, (kind, src) in enumerate(self.cells):
            cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
            if kind == "code":
                cell.update({"execution_count": None, "outputs": []})
            cells.append(cell)
        return {"cells": cells,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def main(self, default_out: str | Path, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description="Generate a Colab notebook")
        parser.add_argument("--out", default=str(default_out))
        args = parser.parse_args(argv)
        out = self.write(args.out)
        print(f"Wrote {out} ({len(self.cells)} cells)")
        return 0


def add_setup(nb: NotebookBuilder) -> None:
    nb.code("!nvidia-smi")
    nb.code(r'''
import os, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

REPO_URL = "https://github.com/viethoang2503/food-recognize-extract.git"
REPO_DIR = Path("/content/food-recognize-extract")
WORK_DIR = Path("/content/drive/MyDrive/foodmm")
DATA_ROOT = Path("/content/data/upmc_food101")

if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
BASE = [f"paths.data_root={DATA_ROOT}", f"paths.work_dir={WORK_DIR}"]
print("repo:", REPO_DIR, "| work dir:", WORK_DIR)
''')
    nb.code("!pip install -q -r requirements-colab.txt")
    nb.code(r'''
def run(script, *args):
    """Run a repo script and stream its output into the notebook."""
    cmd = [sys.executable, "-u", f"scripts/{script}", *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"{script} failed with exit code {proc.returncode}")
''')


def add_data(nb: NotebookBuilder) -> None:
    nb.md(r'''
## Dữ liệu
Giải nén bản zip trên Drive về `/content` (lần đầu tải từ Kaggle, cần Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`). `prepare_data.py` bỏ qua nếu manifest đã có từ Mốc 1.
''')
    nb.code(r'''
import zipfile
from google.colab import userdata

ZIP_PATH = WORK_DIR / "data" / "upmcfood101.zip"
ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
if not ZIP_PATH.exists():
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    subprocess.run(["kaggle", "datasets", "download", "-d", "gianmarco96/upmcfood101", "-p", str(ZIP_PATH.parent)], check=True)
if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
run("prepare_data.py", "--set", *BASE)
''')
```

- [ ] **Step 4: Implement `tools/build_notebook_m2.py`**

```python
#!/usr/bin/env python
"""Generate notebooks/02_milestone2.ipynb (CLIP features, fusion heads, ablations)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Mốc 2: Đặc trưng CLIP, các kiểu fusion và ablation

CLIP ViT-B/16 được đóng băng. Đặc trưng được trích xuất một lần và lưu trên Drive (`foodmm/clip/features`), sau đó các head nhẹ được huấn luyện trên đặc trưng này.

**Thứ tự chạy:** setup → dữ liệu → chạy thử → trích xuất → các run chính → ablation → kết quả. Mọi bước đều chạy lại được: phần đã xong sẽ được bỏ qua.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Chạy thử nhanh (5 lớp)
Ghi vào `/content/smoke_clip`, không đụng tới Drive. Dùng để kiểm tra pipeline và ước lượng thời gian.
''')
nb.code(r'''
SMOKE = ["--set", f"paths.data_root={DATA_ROOT}", "paths.work_dir=/content/smoke_clip",
         "data.subset_classes=5", "data.max_per_class=60", "head.epochs=3",
         "clip_suite.masks=[strict]", "clip_suite.fracs=[0.5]", "clip_suite.missing_md=[0.3]"]
run("prepare_data.py", "--force", *SMOKE)
run("extract_clip.py", "--parts", "image,text_strict,corrupt", "--force", *SMOKE)
run("zero_shot_clip.py", "--force", *SMOKE)
run("run_clip_suite.py", "--stage", "all", *SMOKE)
run("summarize_clip.py", *SMOKE)
''')

nb.md(r'''
## 2. Trích xuất đặc trưng CLIP
Ảnh train/val/test, text theo 3 chế độ che, và các bản test bị nhiễu. Nếu mất session, chạy lại cell này để tiếp tục từ shard cuối cùng.
''')
nb.code(r'''
run("extract_clip.py", "--set", *BASE)
''')

nb.md(r'''
## 3. Zero-shot và các run chính
`image`, `text_{none,exact,strict}`, `{concat,gated,xattn}_{none,exact,strict}`, `late_{none,exact,strict}`.
''')
nb.code(r'''
run("run_clip_suite.py", "--stage", "main", "--set", *BASE)
''')

nb.md(r'''
## 4. Ablation: thiếu modality và tỷ lệ dữ liệu
Head fusion được train với modality dropout 0 và 0.3 (0.1 lấy từ run chính). Sau đó train với 10/25/50% dữ liệu train.
''')
nb.code(r'''
run("run_clip_suite.py", "--stage", "missing", "--set", *BASE)
run("run_clip_suite.py", "--stage", "frac", "--set", *BASE)
''')

nb.md(r'''
## 5. Kết quả
''')
nb.code(r'''
from IPython.display import Image, Markdown, display

run("summarize_clip.py", "--set", *BASE)
RESULTS = WORK_DIR / "clip" / "results"
display(Markdown("### Kết quả chính\n" + (RESULTS / "main.md").read_text()))
display(Markdown("### Thiếu modality (accuracy)\n" + (RESULTS / "missing.md").read_text()))
display(Image(str(RESULTS / "robust.png")))
display(Image(str(RESULTS / "frac.png")))
''')

nb.md(r'''
### Gated fusion: giá trị cổng trung bình
Giá trị gần 1 nghĩa là head dựa vào ảnh nhiều hơn, gần 0 là dựa vào text nhiều hơn.
''')
nb.code(r'''
import json

from foodmm.clip.features import clip_paths
from foodmm.config import load_config

CFG = load_config(overrides=BASE)
CP = clip_paths(CFG)
for p in sorted(CP["runs"].glob("gated_*/metrics_test.json")):
    m = json.loads(p.read_text())
    print(f"{m['run']:28s} acc={m['acc'] * 100:5.2f}%  mean_gate={m['mean_gate']:.3f}")
''')

nb.md(r'''
## 6. t-SNE trên 20 lớp của tập test
So sánh đặc trưng ảnh CLIP, đặc trưng text CLIP (`strict`) và đặc trưng fusion của `xattn_strict`.
''')
nb.code(r'''
import matplotlib.pyplot as plt

from foodmm.clip.features import image_set_name, load_feature_set, text_set_name
from foodmm.clip.report import plot_tsne, sample_for_tsne, tsne_2d
from foodmm.late_fusion import load_preds
from foodmm.utils import load_json

classes = load_json(WORK_DIR / "data" / "classes.json")
xattn = load_preds(CP["runs"] / "xattn_strict", "test")
idx = sample_for_tsne(xattn["labels"], n_classes=20, per_class=50, seed=0)
sources = {
    "Ảnh (CLIP pooled)": load_feature_set(CP["features"] / image_set_name("test"), ["pooled"])["pooled"],
    "Text strict (CLIP pooled)": load_feature_set(CP["features"] / text_set_name("strict", "test"), ["pooled"])["pooled"],
    "Fusion xattn_strict": xattn["features"],
}
for title, feats in sources.items():
    plot_tsne(tsne_2d(feats[idx], seed=0), xattn["labels"][idx], classes, title)
    plt.show()
''')

nb.md(r'''
## 7. So sánh với Mốc 1
Chỉ hiện khi `results/summary.md` của Mốc 1 đã có. Lưu ý: DistilBERT ở Mốc 1 đọc 256 token, còn CLIP chỉ đọc 77 token.
''')
nb.code(r'''
m1 = WORK_DIR / "results" / "summary.md"
if m1.exists():
    display(Markdown("### Mốc 1\n" + m1.read_text()))
    display(Markdown("### Mốc 2\n" + (RESULTS / "main.md").read_text()))
else:
    print("Chưa có kết quả Mốc 1 (notebooks/01_milestone1.ipynb).")
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "02_milestone2.ipynb"))
```

- [ ] **Step 5: Generate the notebook and run the tests**

Run: `.venv/bin/python tools/build_notebook_m2.py`
Expected: `Wrote .../notebooks/02_milestone2.ipynb (N cells)`.
Run: `.venv/bin/pytest tests/test_notebooks.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/nb_utils.py tools/build_notebook_m2.py notebooks/02_milestone2.ipynb tests/test_notebooks.py
git commit -m "feat(clip): shared notebook builder and Milestone 2 notebook"
```

---

### Task 10: End-to-end pipeline test and final verification

**Files:**
- Test: `tests/test_clip_pipeline.py`

- [ ] **Step 1: Write the test `tests/test_clip_pipeline.py`**

```python
import json

import pandas as pd
import pytest

from helpers import clip_smoke_overrides, make_fake_dataset, run_script


@pytest.mark.network
def test_clip_pipeline_end_to_end(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = clip_smoke_overrides(data_root, work)
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_none,text_strict,corrupt", "--set", *sets)
    out = run_script("run_clip_suite.py", "--stage", "all", "--set", *sets).stdout
    assert "zeroshot" in out and "xattn_strict_frac0.5" in out
    run_script("summarize_clip.py", "--set", *sets)
    main = pd.read_csv(work / "clip" / "results" / "main.csv")
    assert main["run"].tolist()[0] == "zeroshot" and "xattn_strict" in set(main["run"])
    missing = pd.read_csv(work / "clip" / "results" / "missing.csv")
    assert {"late_strict", "xattn_strict", "xattn_strict_md0.3"} <= set(missing["run"])
    robust = pd.read_csv(work / "clip" / "results" / "robust.csv")
    assert set(robust["kind"]) == {"blur", "noise", "drop"}
    m = json.loads((work / "clip" / "runs" / "gated_strict" / "metrics_test.json").read_text())
    assert m["head"] == "gated" and "mean_gate" in m
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_clip_pipeline.py -v`
Expected: 1 passed. (No new code: if it fails, fix the responsible module and re-run its task's tests.)

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: every test passes (Milestone 1 tests included).

- [ ] **Step 4: Commit**

```bash
git add tests/test_clip_pipeline.py
git commit -m "test(clip): end-to-end Milestone 2 pipeline on the fake dataset"
```

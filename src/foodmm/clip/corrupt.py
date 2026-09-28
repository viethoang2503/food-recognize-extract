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

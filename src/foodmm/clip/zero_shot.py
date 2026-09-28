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

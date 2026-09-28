#!/usr/bin/env python
"""Train an image-only, text-only or early-fusion (multimodal) classifier."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from foodmm.config import add_config_args, config_from_args, load_config, save_config, work_paths  # noqa: E402
from foodmm.data.datasets import Collator, FoodDataset, build_image_transform  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.data.text_utils import text_column  # noqa: E402
from foodmm.engine import FitConfig, build_optimizer, build_scheduler, fit, predict  # noqa: E402
from foodmm.metrics import compute_metrics  # noqa: E402
from foodmm.models import build_model  # noqa: E402
from foodmm.utils import get_device, load_json, save_json, seed_everything  # noqa: E402

RUN_MODALITY = {"image": "image", "text": "text", "multimodal": "early"}


def default_run_name(modality: str, text_mask: str) -> str:
    return "image" if modality == "image" else f"{RUN_MODALITY[modality]}_{text_mask}"


def forward_fn_for(modality: str):
    def forward(model, batch, return_features):
        if modality == "image":
            out = model(batch["image"], return_features=return_features)
        elif modality == "text":
            out = model(batch["input_ids"], batch["attention_mask"], return_features=return_features)
        else:
            out = model(batch["image"], batch["input_ids"], batch["attention_mask"], return_features=return_features)
        return out if return_features else (out, None)

    return forward


def load_weights(module: torch.nn.Module, ckpt_path: Path, device) -> None:
    module.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])


def require_run(run_dir: Path, what: str) -> None:
    if not (run_dir / "best.pt").exists():
        raise SystemExit(f"Missing {run_dir / 'best.pt'}: train the {what} run '{run_dir.name}' first.")


def prepare_multimodal_inits(cfg: dict, runs_dir: Path, text_mask: str) -> tuple[Path, Path]:
    image_run = runs_dir / cfg["fusion"]["image_init"]
    text_run = runs_dir / (cfg["fusion"]["text_init"] or f"text_{text_mask}")
    require_run(image_run, "image-only")
    require_run(text_run, "text-only")
    trained_mask = load_config(text_run / "config.yaml")["data"]["text_mask"]
    if trained_mask != text_mask:
        raise SystemExit(f"{text_run.name} was trained with text_mask={trained_mask!r} "
                         f"but this run uses text_mask={text_mask!r}")
    cfg["image"]["pretrained"] = False  # weights come from the image-only checkpoint
    return image_run, text_run


def make_loaders(cfg: dict, df, modality: str, model, tokenizer, device):
    use_image = modality in ("image", "multimodal")
    use_text = modality in ("text", "multimodal")
    img_size = int(cfg["image"]["img_size"])
    image_model = model if modality == "image" else getattr(model, "image_model", None)
    mean, std = image_model.data_mean_std() if use_image else (None, None)
    tcfg = cfg["train"][modality]
    workers = int(cfg["data"]["num_workers"])
    collate = Collator(tokenizer, int(cfg["text"]["max_len"]))
    loaders, datasets = {}, {}
    for split in ("train", "val", "test"):
        tf = build_image_transform(img_size, train=split == "train", mean=mean, std=std) if use_image else None
        datasets[split] = FoodDataset(
            df[df["split"] == split], cfg["paths"]["data_root"], text_col=text_column(cfg["data"]["text_mask"]),
            use_image=use_image, use_text=use_text, transform=tf, img_size=img_size,
            max_bad_images=int(cfg["data"]["max_bad_images"]),
        )
        loaders[split] = DataLoader(
            datasets[split], batch_size=int(tcfg["batch_size"]), shuffle=split == "train",
            num_workers=workers, collate_fn=collate, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
    return loaders, datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", required=True, choices=sorted(RUN_MODALITY))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--force", action="store_true", help="Delete the run directory and retrain from scratch")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    seed_everything(int(cfg["seed"]))
    paths = work_paths(cfg)
    modality, text_mask = args.modality, cfg["data"]["text_mask"]
    text_column(text_mask)  # validate early
    run_name = args.run_name or default_run_name(modality, text_mask)
    run_dir = paths["runs"] / run_name
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "metrics_test.json").exists():
        print(f"{run_name} is already finished (use --force to retrain)")
        return 0

    device = get_device()
    df = load_manifest(paths["manifest"])
    classes = load_json(paths["classes"])
    inits = prepare_multimodal_inits(cfg, paths["runs"], text_mask) if modality == "multimodal" else None
    model = build_model(modality, cfg, len(classes))
    if inits is not None:
        load_weights(model.image_model, inits[0] / "best.pt", "cpu")
        load_weights(model.text_model, inits[1] / "best.pt", "cpu")
    model.to(device)

    tokenizer = None
    if modality in ("text", "multimodal"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    loaders, datasets = make_loaders(cfg, df, modality, model, tokenizer, device)


    tcfg, tr = cfg["train"][modality], cfg["train"]
    wd = float(tcfg["weight_decay"])
    if modality == "multimodal":
        groups = model.param_groups(float(tcfg["lr_image"]), float(tcfg["lr_text"]), float(tcfg["lr_head"]), wd)
    else:
        groups = model.param_groups(float(tcfg["lr_backbone"]), float(tcfg["lr_head"]), wd)
    optimizer = build_optimizer(groups)
    epochs = int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, epochs * len(loaders["train"]), float(tr["warmup_ratio"]))
    fit_cfg = FitConfig(epochs=epochs, grad_clip=float(tr["grad_clip"]), label_smoothing=float(tr["label_smoothing"]),
                        patience=int(tr["patience"]), amp=bool(tr["amp"]), log_every=int(tr["log_every"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"name": run_name, "modality": RUN_MODALITY[modality]}
    save_config(cfg, run_dir / "config.yaml")
    forward = forward_fn_for(modality)
    state = fit(model, loaders["train"], loaders["val"], forward, optimizer, scheduler, fit_cfg, run_dir, device)

    load_weights(model, run_dir / "best.pt", device)
    outputs = {}
    for split in ("val", "test"):
        out = predict(model, loaders[split], forward, device, amp=fit_cfg.amp)
        ids = datasets[split].df["id"].to_numpy()[out["idx"]].astype(str)
        np.savez(run_dir / f"preds_{split}.npz", logits=out["logits"], labels=out["labels"], ids=ids,
                 features=out["features"].astype(np.float16))
        outputs[split] = out
    metrics = compute_metrics(outputs["test"]["logits"], outputs["test"]["labels"])
    metrics.update({"run": run_name, "modality": RUN_MODALITY[modality],
                    "text_mask": "-" if modality == "image" else text_mask,
                    "best_val_acc": float(state["best_acc"])})
    save_json(metrics, run_dir / "metrics_test.json")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

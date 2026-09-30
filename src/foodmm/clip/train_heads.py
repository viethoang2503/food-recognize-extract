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
    ids, labels = sub["id"].to_numpy().astype(str), sub["label_idx"].to_numpy()
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

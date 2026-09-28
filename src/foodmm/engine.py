"""Generic training / evaluation loop with AMP, checkpointing, resume and early stopping."""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .utils import save_json


@dataclass
class FitConfig:
    epochs: int
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    patience: int = 2
    amp: bool = True
    log_every: int = 100


def build_optimizer(param_groups: list[dict]) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _use_amp(amp: bool, device: torch.device) -> bool:
    return bool(amp) and device.type == "cuda"


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16) if enabled else contextlib.nullcontext()


@torch.no_grad()
def predict(model: nn.Module, loader, forward_fn, device: torch.device, amp: bool = True,
            return_features: bool = True) -> dict[str, np.ndarray]:
    model.eval()
    use_amp = _use_amp(amp, device)
    logits, feats, labels, idx = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with _autocast(device, use_amp):
            out, f = forward_fn(model, batch, return_features)
        logits.append(out.float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        idx.append(batch["idx"].cpu().numpy())
        if return_features and f is not None:
            feats.append(f.float().cpu().numpy())
    result = {"logits": np.concatenate(logits), "labels": np.concatenate(labels), "idx": np.concatenate(idx)}
    if feats:
        result["features"] = np.concatenate(feats)
    return result


def _train_one_epoch(model, loader, forward_fn, optimizer, scheduler, scaler, criterion,
                     cfg: FitConfig, device: torch.device, use_amp: bool) -> float:
    model.train()
    total_loss, n_seen = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with _autocast(device, use_amp):
            logits, _ = forward_fn(model, batch, False)
            loss = criterion(logits.float(), batch["label"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        bs = int(batch["label"].shape[0])
        total_loss += loss.item() * bs
        n_seen += bs
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"  step {step}/{len(loader)} loss={total_loss / n_seen:.4f}", flush=True)
    return total_loss / max(1, n_seen)


def fit(model: nn.Module, train_loader, val_loader, forward_fn, optimizer, scheduler, cfg: FitConfig,
        run_dir: str | Path, device: torch.device) -> dict:
    """Train with early stopping on val accuracy. Resumes from run_dir/last.pt if present."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    use_amp = _use_amp(cfg.amp, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    state = {"epoch": 0, "best_acc": -1.0, "bad_epochs": 0, "history": []}
    last_path = run_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"Resumed from {last_path} after epoch {state['epoch']}", flush=True)

    for epoch in range(state["epoch"], cfg.epochs):
        if state["bad_epochs"] >= cfg.patience:
            print(f"Early stopping: no val improvement for {cfg.patience} epochs", flush=True)
            break
        start = time.time()
        train_loss = _train_one_epoch(model, train_loader, forward_fn, optimizer, scheduler, scaler,
                                      criterion, cfg, device, use_amp)
        val = predict(model, val_loader, forward_fn, device, amp=cfg.amp, return_features=False)
        val_acc = float((val["logits"].argmax(axis=1) == val["labels"]).mean())
        record = {"epoch": epoch + 1, "train_loss": float(train_loss), "val_acc": val_acc,
                  "seconds": round(time.time() - start, 1)}
        state["history"].append(record)
        if val_acc > state["best_acc"]:
            state["best_acc"], state["bad_epochs"] = val_acc, 0
            torch.save({"model": model.state_dict()}, run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch + 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state}, last_path)
        save_json(state["history"], run_dir / "history.json")
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={train_loss:.4f} val_acc={val_acc:.4f} "
              f"({record['seconds']}s)", flush=True)
    return state

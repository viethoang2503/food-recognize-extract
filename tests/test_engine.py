import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from foodmm.engine import FitConfig, build_optimizer, build_scheduler, fit, predict

CPU = torch.device("cpu")


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x, return_features=False):
        logits = self.fc(x)
        return (logits, x) if return_features else logits


def forward_fn(model, batch, return_features):
    out = model(batch["x"], return_features=return_features)
    return out if return_features else (out, None)


def make_loader(n=90, seed=0, batch_size=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    items = [{"x": x[i], "label": y[i], "idx": i} for i in range(n)]

    def collate(batch):
        return {"x": torch.stack([b["x"] for b in batch]), "label": torch.stack([b["label"] for b in batch]),
                "idx": torch.tensor([b["idx"] for b in batch])}

    return DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=collate)


def setup(lr=0.1, epochs=8):
    torch.manual_seed(0)
    model = TinyNet()
    opt = build_optimizer([{"params": list(model.parameters()), "lr": lr, "weight_decay": 0.0}])
    train = make_loader(seed=0)
    sched = build_scheduler(opt, total_steps=epochs * len(train), warmup_ratio=0.1)
    return model, opt, sched, train, make_loader(seed=1)


def test_fit_trains_and_writes_checkpoints(tmp_path):
    model, opt, sched, train, val = setup()
    state = fit(model, train, val, forward_fn, opt, sched,
                FitConfig(epochs=8, amp=False, patience=10, label_smoothing=0.0), tmp_path, CPU)
    for name in ("best.pt", "last.pt", "history.json"):
        assert (tmp_path / name).exists()
    assert len(state["history"]) == 8
    assert state["best_acc"] > 0.6
    assert json.loads((tmp_path / "history.json").read_text())[0]["epoch"] == 1


def test_fit_resumes_from_last_checkpoint(tmp_path):
    model, opt, sched, train, val = setup(epochs=4)
    first = fit(model, train, val, forward_fn, opt, sched, FitConfig(epochs=2, amp=False, patience=10), tmp_path, CPU)
    model, opt, sched, train, val = setup(epochs=4)
    second = fit(model, train, val, forward_fn, opt, sched, FitConfig(epochs=4, amp=False, patience=10), tmp_path, CPU)
    assert len(second["history"]) == 4
    assert second["history"][:2] == first["history"]


def test_early_stopping(tmp_path):
    model, opt, sched, train, val = setup(lr=0.0, epochs=10)
    state = fit(model, train, val, forward_fn, opt, sched, FitConfig(epochs=10, amp=False, patience=2), tmp_path, CPU)
    assert len(state["history"]) == 3


def test_predict_returns_arrays():
    model, _, _, _, val = setup()
    out = predict(model, val, forward_fn, CPU, amp=False)
    assert out["logits"].shape == (90, 3) and out["features"].shape == (90, 4)
    assert out["labels"].shape == (90,) and out["idx"].tolist() == list(range(90))
    assert "features" not in predict(model, val, forward_fn, CPU, amp=False, return_features=False)


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, total_steps=10, warmup_ratio=0.2)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)

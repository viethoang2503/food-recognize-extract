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

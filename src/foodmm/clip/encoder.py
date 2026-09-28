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

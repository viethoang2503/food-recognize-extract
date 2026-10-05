"""Image / text / fusion predictions from one frozen CLIP encoder and three Milestone 2 heads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..clip.features import clip_paths
from ..clip.heads import build_head, drop_modalities
from ..config import load_config, work_paths
from ..data.text_utils import build_mask_pattern, class_to_phrase, clean_text, mask_text
from ..utils import get_device, load_json


def load_head(run_dir: str | Path, device: torch.device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    hcfg = load_config(run_dir / "config.yaml")["head"]
    model = build_head(ckpt["head"], ckpt["dims"], int(ckpt["num_classes"]), hcfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


class DemoPredictor:
    def __init__(self, cfg: dict, encoder=None, device: torch.device | None = None):
        d = cfg["demo"]
        self.cfg = cfg
        self.device = device or get_device()
        self.top_k = int(d["top_k"])
        self.max_chars = int(cfg["data"]["max_chars"])
        self.run_names = {"image": d["image_run"], "text": d["text_run"] or f"text_{d['text_mask']}",
                          "fusion": d["fusion_run"]}
        runs = clip_paths(cfg)["runs"]
        missing = [r for r in self.run_names.values() if not (runs / r / "best.pt").exists()]
        if missing:
            raise SystemExit(f"Missing Milestone 2 runs {missing} in {runs}: run notebooks/02_milestone2.ipynb first")
        self.classes = load_json(work_paths(cfg)["classes"])
        self.pattern = build_mask_pattern(self.classes, d["text_mask"])
        self.heads = {k: load_head(runs / r, self.device) for k, r in self.run_names.items()}
        if encoder is None:
            from ..clip.encoder import ClipEncoder

            cc = cfg["clip"]
            encoder = ClipEncoder(cc["model_name"], device=self.device, fp16=bool(cc["fp16"]),
                                  n_tokens=int(cc["n_tokens"]))
        self.encoder = encoder

    def _labels(self, probs: np.ndarray) -> dict[str, float]:
        top = np.argsort(-probs)[:self.top_k]
        return {class_to_phrase(self.classes[int(i)]): float(probs[i]) for i in top}

    @torch.no_grad()
    def _probs(self, key: str, batch: dict) -> np.ndarray:
        return torch.softmax(self.heads[key](batch)["logits"].float(), dim=1)[0].cpu().numpy()

    def predict(self, image, text: str | None) -> dict:
        text = (text or "").strip()
        if image is None and not text:
            raise ValueError("Please provide an image or a text")
        enc, n = self.encoder, self.encoder.n_tokens
        masked = mask_text(clean_text(text, self.max_chars), self.pattern) if text else ""
        if image is not None:
            img, img_tok = enc.encode_images([image.convert("RGB")])
        else:
            img, img_tok = np.zeros((1, enc.embed_dim)), np.zeros((1, n, enc.image_dim))
        if text:
            txt, txt_tok, txt_mask = enc.encode_texts([masked])
        else:
            txt, txt_tok, txt_mask = np.zeros((1, enc.embed_dim)), np.zeros((1, n, enc.text_dim)), np.zeros((1, n))
        batch = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
                 for k, v in {"img": img, "img_tok": img_tok, "txt": txt, "txt_tok": txt_tok, "txt_mask": txt_mask}.items()}
        fusion_batch = drop_modalities(batch, torch.tensor([image is None]), torch.tensor([not text]))
        p_fusion = self._probs("fusion", fusion_batch)
        return {
            "image": self._labels(self._probs("image", batch)) if image is not None else None,
            "text": self._labels(self._probs("text", batch)) if text else None,
            "fusion": self._labels(p_fusion),
            "masked_text": masked,
        }

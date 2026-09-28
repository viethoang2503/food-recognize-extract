"""PyTorch dataset over the manifest, image transforms and a tokenizing collator."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(img_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(round(img_size / 0.875))),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class TooManyBadImages(RuntimeError):
    """Raised when more than `max_bad_images` images fail to load."""


class FoodDataset(Dataset):
    """Items: idx, label and optionally image (tensor) and text (str).

    `bad_images` is counted per process: with DataLoader workers each worker keeps its own count.
    """

    def __init__(self, df: pd.DataFrame, data_root: str | Path, *, text_col: str = "text",
                 use_image: bool = True, use_text: bool = True, transform=None,
                 img_size: int = 224, max_bad_images: int = 50):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.use_image, self.use_text = use_image, use_text
        self.transform = transform
        self.img_size = img_size
        self.max_bad_images = max_bad_images
        self.bad_images = 0
        self._paths = self.df["image_path"].tolist()
        self._labels = self.df["label_idx"].astype(int).tolist()
        self._texts = self.df[text_col].astype(str).tolist() if use_text else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, i: int) -> torch.Tensor:
        path = self.data_root / self._paths[i]
        try:
            with Image.open(path) as im:
                return self.transform(im.convert("RGB"))
        except (OSError, ValueError) as err:
            self.bad_images += 1
            logger.warning("Could not read %s (%s); using a blank image", path, err)
            if self.bad_images > self.max_bad_images:
                raise TooManyBadImages(f"{self.bad_images} unreadable images (max {self.max_bad_images})") from err
            return torch.zeros(3, self.img_size, self.img_size)

    def __getitem__(self, i: int) -> dict:
        item = {"idx": i, "label": self._labels[i]}
        if self.use_image:
            item["image"] = self._load_image(i)
        if self.use_text:
            item["text"] = self._texts[i]
        return item


class Collator:
    """Stack images and tokenize texts per batch (dynamic padding)."""

    def __init__(self, tokenizer=None, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict:
        out = {
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }
        if "image" in batch[0]:
            out["image"] = torch.stack([b["image"] for b in batch])
        if "text" in batch[0] and self.tokenizer is not None:
            enc = self.tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            out["input_ids"] = enc["input_ids"]
            out["attention_mask"] = enc["attention_mask"]
        return out

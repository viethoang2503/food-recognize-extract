import pytest
import torch

from foodmm.data.datasets import Collator, FoodDataset, TooManyBadImages, build_image_transform
from foodmm.data.prepare import build_manifest
from helpers import TINY_TEXT_MODEL, make_fake_dataset


@pytest.fixture
def manifest(tmp_path):
    root = make_fake_dataset(tmp_path / "ds")
    df, _, _ = build_manifest(root, val_ratio=0.3, seed=0)
    return root, df


def test_dataset_image_and_text(manifest):
    root, df = manifest
    ds = FoodDataset(df, root, text_col="text_strict", transform=build_image_transform(32, train=False), img_size=32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["text"] == df["text_strict"].iloc[0]
    assert item["label"] == int(df["label_idx"].iloc[0])
    assert len(ds) == len(df)


def test_train_transform_shape(manifest):
    root, df = manifest
    ds = FoodDataset(df, root, use_text=False, transform=build_image_transform(32, train=True), img_size=32)
    assert ds[1]["image"].shape == (3, 32, 32)
    assert "text" not in ds[1]


def test_bad_image_replaced_with_zeros(manifest):
    root, df = manifest
    (root / df["image_path"].iloc[0]).write_bytes(b"not an image")
    ds = FoodDataset(df, root, use_text=False, transform=build_image_transform(32, train=False),
                     img_size=32, max_bad_images=5)
    assert torch.count_nonzero(ds[0]["image"]) == 0
    assert ds.bad_images == 1


def test_too_many_bad_images_raises(manifest):
    root, df = manifest
    (root / df["image_path"].iloc[0]).write_bytes(b"not an image")
    ds = FoodDataset(df, root, use_text=False, transform=build_image_transform(32, train=False),
                     img_size=32, max_bad_images=0)
    with pytest.raises(TooManyBadImages):
        ds[0]


def test_collator_images_only(manifest):
    root, df = manifest
    ds = FoodDataset(df, root, use_text=False, transform=build_image_transform(32, train=False), img_size=32)
    batch = Collator()([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["label"].dtype == torch.long
    assert batch["idx"].tolist() == [0, 1]
    assert "input_ids" not in batch


@pytest.mark.network
def test_collator_tokenizes():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY_TEXT_MODEL)
    batch = Collator(tok, max_len=8)([
        {"idx": 0, "label": 1, "text": "apple pie [MASK]"},
        {"idx": 1, "label": 2, "text": "a much longer text about fries and salad and more words"},
    ])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= 8
    assert batch["attention_mask"].shape == batch["input_ids"].shape

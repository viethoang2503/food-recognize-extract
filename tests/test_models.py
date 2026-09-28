import pytest
import torch

from foodmm.config import load_config
from foodmm.models import (
    ImageClassifier, MultimodalClassifier, TextClassifier, build_model, drop_modalities,
)
from helpers import TINY_TEXT_MODEL


def test_image_classifier_shapes_and_groups():
    m = ImageClassifier("resnet18", num_classes=3, pretrained=False)
    logits, feats = m(torch.randn(2, 3, 32, 32), return_features=True)
    assert logits.shape == (2, 3)
    assert m.feat_dim == 512 and feats.shape == (2, 512)
    groups = m.param_groups(1e-4, 1e-3, 0.01)
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    mean, std = m.data_mean_std()
    assert len(mean) == 3 and len(std) == 3


def test_drop_modalities_never_drops_both():
    torch.manual_seed(0)
    zi, zt = torch.ones(2000, 4), torch.ones(2000, 4)
    a, b = drop_modalities(zi, zt, p=0.5)
    img_off = a.abs().sum(1) == 0
    txt_off = b.abs().sum(1) == 0
    assert not (img_off & txt_off).any()
    assert 0.4 < (img_off | txt_off).float().mean().item() < 0.6
    same_a, same_b = drop_modalities(zi, zt, p=0.5, training=False)
    assert torch.equal(same_a, zi) and torch.equal(same_b, zt)


def test_build_model_image_and_unknown():
    cfg = load_config(overrides=["image.backbone=resnet18", "image.pretrained=false"])
    assert isinstance(build_model("image", cfg, 3), ImageClassifier)
    with pytest.raises(ValueError):
        build_model("audio", cfg, 3)


@pytest.mark.network
def test_text_and_multimodal():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY_TEXT_MODEL)
    enc = tok(["apple pie", "french fries with salt"], padding=True, return_tensors="pt")
    txt = TextClassifier(TINY_TEXT_MODEL, 3, pretrained=False)
    logits, feats = txt(enc["input_ids"], enc["attention_mask"], return_features=True)
    assert logits.shape == (2, 3) and feats.shape == (2, txt.feat_dim)

    img = ImageClassifier("resnet18", 3, pretrained=False)
    mm = MultimodalClassifier(img, txt, 3, hidden=16, dropout=0.0, modality_dropout=0.0)
    logits, fused = mm(torch.randn(2, 3, 32, 32), enc["input_ids"], enc["attention_mask"], return_features=True)
    assert logits.shape == (2, 3) and fused.shape == (2, 32)
    groups = mm.param_groups(1e-5, 2e-5, 1e-3, 0.0)
    assert [g["lr"] for g in groups] == [1e-5, 2e-5, 1e-3]
    grouped = {id(p) for g in groups for p in g["params"]}
    assert id(img.head.weight) not in grouped and id(txt.head.weight) not in grouped

    cfg = load_config(overrides=["image.backbone=resnet18", "image.pretrained=false",
                                 f"text.model_name={TINY_TEXT_MODEL}", "text.pretrained=false", "fusion.hidden=16"])
    assert isinstance(build_model("multimodal", cfg, 3), MultimodalClassifier)

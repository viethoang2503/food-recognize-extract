import pytest
import torch

from foodmm.clip.heads import (
    HEADS, build_head, drop_modalities, head_inputs, masked_mean, random_modality_drop,
)
from foodmm.config import load_config

DIMS = {"img": 8, "txt": 8, "img_tok": 6, "txt_tok": 5, "n_tokens": 4}


def _batch(b=3):
    g = torch.Generator().manual_seed(0)
    return {"img": torch.randn(b, 8, generator=g), "txt": torch.randn(b, 8, generator=g),
            "img_tok": torch.randn(b, 4, 6, generator=g), "txt_tok": torch.randn(b, 4, 5, generator=g),
            "txt_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.float32)[:b],
            "label": torch.zeros(b, dtype=torch.long)}


def _hcfg(**kw):
    h = dict(load_config()["head"])
    h.update({"hidden": 16, "xattn_dim": 8, "xattn_heads": 2, **kw})
    return h


@pytest.mark.parametrize("name", HEADS)
def test_heads_output_shapes(name):
    model = build_head(name, DIMS, 7, _hcfg()).eval()
    out = model(_batch())
    assert out["logits"].shape == (3, 7)
    assert out["features"].shape[0] == 3 and out["features"].ndim == 2
    if name == "gated":
        assert out["gate"].shape == (3,) and ((out["gate"] >= 0) & (out["gate"] <= 1)).all()


def test_build_head_rejects_unknown():
    with pytest.raises(ValueError):
        build_head("bilinear", DIMS, 7, _hcfg())


def test_head_inputs():
    assert head_inputs("image") == (True, False, False)
    assert head_inputs("text") == (False, True, False)
    assert head_inputs("gated") == (True, True, False)
    assert head_inputs("xattn") == (True, True, True)


def test_masked_mean_all_zero_mask_gives_zero():
    x = torch.ones(2, 3, 4)
    out = masked_mean(x, torch.tensor([[1.0, 0, 0], [0, 0, 0]]))
    assert out[0].tolist() == [1.0] * 4 and out[1].tolist() == [0.0] * 4


def test_drop_modalities():
    b = _batch()
    out = drop_modalities(b, torch.tensor([True, False, False]), torch.tensor([False, True, False]))
    assert out["img"][0].abs().sum() == 0 and out["img_tok"][0].abs().sum() == 0
    assert out["txt"][1].abs().sum() == 0 and out["txt_mask"][1].sum() == 0 and out["txt_tok"][1].abs().sum() == 0
    assert torch.equal(out["img"][1], b["img"][1]) and torch.equal(out["txt"][0], b["txt"][0])
    assert b["img"][0].abs().sum() > 0  # input not modified


def test_random_modality_drop_never_both():
    g = torch.Generator().manual_seed(0)
    di, dt = random_modality_drop(10000, 0.5, g)
    assert not (di & dt).any()
    assert 0.2 < di.float().mean() < 0.3 and 0.2 < dt.float().mean() < 0.3
    di0, dt0 = random_modality_drop(10, 0.0, g)
    assert not di0.any() and not dt0.any()


def test_xattn_ignores_padded_text_tokens():
    model = build_head("xattn", DIMS, 7, _hcfg()).eval()
    b = _batch()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["txt_tok"][0, 2:] = 99.0  # padded positions of sample 0
    with torch.no_grad():
        assert torch.allclose(model(b)["logits"][0], model(b2)["logits"][0], atol=1e-5)

import numpy as np
import pytest
import torch
from PIL import Image

from foodmm.clip.encoder import chunk_pool, grid_pool
from foodmm.clip.zero_shot import class_prompts, zero_shot_logits
from helpers import TINY_CLIP_MODEL


def test_chunk_pool_splits_valid_tokens():
    hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1).repeat(2, 1, 1)
    attn = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]])
    tokens, mask = chunk_pool(hidden, attn, 3)
    assert tokens.shape == (2, 3, 1)
    assert tokens[0, :, 0].tolist() == [0.5, 2.5, 4.5]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, True, False]  # 2 valid tokens -> chunks 0 and 1
    assert tokens[1, 2, 0].item() == 0.0


def test_grid_pool():
    patches = torch.ones(2, 9, 5)
    assert grid_pool(patches, 4).shape == (2, 4, 5)
    with pytest.raises(ValueError):
        grid_pool(patches, 5)


def test_zero_shot_math():
    prompts = class_prompts(["apple_pie", "french_fries"], "a photo of {}, a type of food")
    assert prompts == ["a photo of apple pie, a type of food", "a photo of french fries, a type of food"]
    img = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float16)
    txt = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = zero_shot_logits(img, txt)
    assert logits.shape == (2, 2) and logits.argmax(1).tolist() == [0, 1]
    assert logits[1, 1] == pytest.approx(100.0)


@pytest.mark.network
def test_clip_encoder_shapes():
    from foodmm.clip.encoder import ClipEncoder

    enc = ClipEncoder(TINY_CLIP_MODEL, device=torch.device("cpu"), n_tokens=16)
    imgs = [Image.new("RGB", (40, 32), (i * 50, 100, 30)) for i in range(3)]
    pooled, tokens = enc.encode_images(imgs)
    assert pooled.shape == (3, enc.embed_dim) and tokens.shape == (3, 16, enc.image_dim)
    assert pooled.dtype == np.float16
    assert np.allclose(np.linalg.norm(pooled.astype(np.float32), axis=1), 1.0, atol=1e-2)
    tp, tt, tm = enc.encode_texts(["apple pie with cream and a long list of words " * 3, "", "fries"])
    assert tp.shape == (3, enc.embed_dim) and tt.shape == (3, 16, enc.text_dim) and tm.shape == (3, 16)
    assert tm[0].all() and tm[1].sum() < 16
    assert enc.preprocess(imgs[0]).shape[0] == 3

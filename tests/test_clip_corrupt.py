import numpy as np
from PIL import Image

from foodmm.clip.corrupt import (
    apply_image_corruption, blur_image, corruption_specs, fmt_level, noise_image, sample_seed, word_drop,
)
from foodmm.config import load_config


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8))


def test_fmt_level_and_seed():
    assert fmt_level(1) == "1" and fmt_level(0.05) == "0.05" and fmt_level(2.0) == "2"
    assert sample_seed(42, "a") == sample_seed(42, "a") != sample_seed(42, "b")


def test_blur_and_noise_are_deterministic():
    img = _img()
    b = np.asarray(blur_image(img, 2), dtype=float)
    assert b.shape == (20, 20, 3) and b.std() < np.asarray(img, dtype=float).std()
    n1, n2 = noise_image(img, 0.1, seed=3), noise_image(img, 0.1, seed=3)
    assert np.array_equal(np.asarray(n1), np.asarray(n2))
    assert not np.array_equal(np.asarray(n1), np.asarray(img))
    same = apply_image_corruption(img, "noise", 0.1, 3)
    assert np.array_equal(np.asarray(same), np.asarray(n1))


def test_word_drop():
    text = " ".join(f"w{i}" for i in range(200))
    out = word_drop(text, 0.5, seed=1)
    assert out == word_drop(text, 0.5, seed=1)
    assert 60 < len(out.split()) < 140
    assert len(word_drop("one two", 0.999, seed=0).split()) == 1  # never empty
    assert word_drop("", 0.5, seed=0) == ""


def test_corruption_specs_from_default_config():
    specs = corruption_specs(load_config())
    names = [s[0] for s in specs]
    assert names == ["blur1", "blur2", "blur4", "noise0.05", "noise0.1", "noise0.2",
                     "drop0.25", "drop0.5", "drop0.75"]
    assert specs[0][1:] == ("blur", 1.0)

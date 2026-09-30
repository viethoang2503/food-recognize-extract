import pytest
from PIL import Image

from foodmm.config import load_config
from foodmm.vlm.backend import FakeBackend, HFVLMBackend, make_backend, resize_for_vlm, resolve_quantize
from helpers import TINY_VLM_MODEL


def test_resize_for_vlm():
    img = Image.new("RGB", (2000, 1000))
    assert resize_for_vlm(img, 768).size == (768, 384)
    small = Image.new("L", (100, 50))
    out = resize_for_vlm(small, 768)
    assert out.size == (100, 50) and out.mode == "RGB"


def test_resolve_quantize():
    assert resolve_quantize("auto", cuda=False, capability_major=0) == "none"
    assert resolve_quantize("auto", cuda=True, capability_major=7) == "4bit"  # T4
    assert resolve_quantize("auto", cuda=True, capability_major=8) == "none"  # L4 / A100
    assert resolve_quantize("none", cuda=True, capability_major=7) == "none"
    with pytest.raises(RuntimeError):
        resolve_quantize("4bit", cuda=False, capability_major=0)
    with pytest.raises(ValueError):
        resolve_quantize("8bit", cuda=True, capability_major=8)


def test_fake_backend_and_factory():
    fb = FakeBackend(["a", "b"])
    img = Image.new("RGB", (4, 4))
    assert [fb.generate(img, "p", "s") for _ in range(2)] + [fb.generate(None, "p", "s")] == ["a", "b", "a"]
    assert len(fb.calls) == 3 and fb.calls[0] == ("p", "s") and fb.images == [img, img, None]
    assert '"dish_name"' in FakeBackend().generate(img, "p", "s")
    assert isinstance(make_backend(load_config(overrides=["vlm.backend=fake"])), FakeBackend)
    hf = make_backend(load_config())
    assert isinstance(hf, HFVLMBackend) and not hf.loaded  # lazy: nothing downloaded yet


def test_hf_backend_requires_gpu_unless_allowed():
    import torch

    if torch.cuda.is_available():
        pytest.skip("machine has a GPU")
    with pytest.raises(RuntimeError, match="GPU"):
        HFVLMBackend(TINY_VLM_MODEL).load()


@pytest.mark.network
def test_hf_backend_tiny_model_on_cpu():
    backend = HFVLMBackend(TINY_VLM_MODEL, max_new_tokens=5, allow_cpu=True)
    out = backend.generate(Image.new("RGB", (64, 48), (200, 100, 50)), "Describe the dish.", "Return JSON.")
    assert isinstance(out, str) and backend.loaded
    assert isinstance(backend.generate(None, "Text only: crispy [MASK].", "Return JSON."), str)

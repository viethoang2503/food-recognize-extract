import gradio as gr
import pytest
from PIL import Image

from foodmm.demo.app import build_app, make_handlers
from foodmm.demo.vlm_service import LazyVLM, choose_mode
from foodmm.vlm.backend import FakeBackend

IMG = Image.new("RGB", (16, 16), (200, 120, 40))
TOP = {"apple pie": 0.7, "french fries": 0.2}


class FakePredictor:
    def predict(self, image, text):
        if image is None and not (text or "").strip():
            raise ValueError("Please provide an image or a text")
        return {"image": TOP if image is not None else None, "text": TOP if text else None, "fusion": TOP,
                "masked_text": "best [MASK]" if text else ""}


EXAMPLES = [{"name": "1. apple pie", "id": "x", "label": "apple_pie", "image_path": None, "text": "best apple pie",
             "pred": {"image": TOP, "text": TOP, "fusion": TOP, "masked_text": "best [MASK]"}, "vlm": {"dish_name": "pie"}}]


def test_choose_mode():
    assert choose_mode(True, True) == "image_text"
    assert choose_mode(True, False) == "image"
    assert choose_mode(False, True) == "text"
    assert choose_mode(False, False) is None


def test_lazy_vlm_loads_once():
    fb = FakeBackend()
    calls = []
    vlm = LazyVLM(lambda: calls.append(1) or fb)
    assert vlm.extract(None, "  ") == {"error": "The VLM needs an image or a text"} and not calls
    out = vlm.extract(IMG)
    assert out["dish_name"] == "apple pie" and out["mode"] == "image"
    assert vlm.extract(IMG, "best [MASK]")["mode"] == "image_text"
    assert vlm.extract(None, "best [MASK]")["mode"] == "text"
    assert len(calls) == 1 and vlm.factory_calls == 1
    assert "photo" in fb.calls[0][0] and "[MASK]" in fb.calls[1][0]
    assert fb.images[1] is not None and fb.images[2] is None  # text mode sends no image


def test_lazy_vlm_remembers_load_error():
    def boom():
        raise RuntimeError("no GPU")

    vlm = LazyVLM(boom)
    assert "no GPU" in vlm.extract(IMG)["error"]
    assert "no GPU" in vlm.extract(IMG)["error"] and vlm.factory_calls == 1


def test_lazy_vlm_invalid_output():
    out = LazyVLM(lambda: FakeBackend(["garbage"])).extract(IMG)
    assert "error" in out and out["raw"] == "garbage"


def test_on_predict():
    on_predict, _ = make_handlers(FakePredictor(), LazyVLM(lambda: FakeBackend()), EXAMPLES)
    img, txt, fus, masked, vlm = on_predict(IMG, "best apple pie", False)
    assert img == TOP and txt == TOP and fus == TOP and masked == "best [MASK]" and vlm is None
    *_, vlm = on_predict(IMG, "", True)
    assert vlm["dish_name"] == "apple pie" and vlm["mode"] == "image"
    assert on_predict(IMG, "best apple pie", True)[4]["mode"] == "image_text"
    assert on_predict(None, "best apple pie", True)[4]["mode"] == "text"
    img, txt, fus, masked, vlm = on_predict(None, "  ", True)
    assert img is None and fus is None and "provide an image" in vlm["error"]


def test_on_predict_without_vlm_service():
    on_predict, _ = make_handlers(FakePredictor(), None, None)
    assert on_predict(IMG, "", True)[4] is None


def test_on_example(tmp_path):
    path = tmp_path / "ex.jpg"
    IMG.save(path)
    examples = [{**EXAMPLES[0], "image_path": str(path)}]
    _, on_example = make_handlers(FakePredictor(), None, examples)
    image, text, img, txt, fus, masked, vlm = on_example("1. apple pie")
    assert image.size == (16, 16) and text == "best apple pie" and fus == TOP and vlm == {"dish_name": "pie"}
    _, on_example_empty = make_handlers(FakePredictor(), None, None)
    assert "build_demo_examples.py" in on_example_empty("x")[6]["info"]


@pytest.mark.parametrize("examples", [None, EXAMPLES])
def test_build_app(examples):
    app = build_app(FakePredictor(), LazyVLM(lambda: FakeBackend()), examples)
    assert isinstance(app, gr.Blocks)

"""VLM backends: a lazy Hugging Face backend (Qwen3-VL) and a fake one for tests."""
from __future__ import annotations

import itertools
import json
from collections.abc import Sequence

from PIL import Image

FAKE_OUTPUT = json.dumps({"dish_name": "apple pie", "cuisine": "American", "main_ingredients": ["apple", "sugar"],
                          "cooking_method": "baked", "confidence": 0.7})


def resize_for_vlm(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def resolve_quantize(quantize: str, cuda: bool, capability_major: int) -> str:
    if quantize not in ("auto", "4bit", "none"):
        raise ValueError(f"vlm.quantize must be auto, 4bit or none, got {quantize!r}")
    if quantize == "4bit" and not cuda:
        raise RuntimeError("4-bit loading needs a CUDA GPU")
    if quantize == "auto":
        return "4bit" if cuda and capability_major < 8 else "none"
    return quantize


class HFVLMBackend:
    def __init__(self, model_name: str, quantize: str = "auto", max_new_tokens: int = 256, allow_cpu: bool = False):
        self.model_name, self.quantize = model_name, quantize
        self.max_new_tokens, self.allow_cpu = int(max_new_tokens), bool(allow_cpu)
        self.model = self.processor = None
        self.quant_mode = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cuda = torch.cuda.is_available()
        if not cuda and not self.allow_cpu:
            raise RuntimeError("The VLM needs a CUDA GPU (Colab: Runtime > Change runtime type > GPU)")
        major = torch.cuda.get_device_capability()[0] if cuda else 0
        self.quant_mode = resolve_quantize(self.quantize, cuda, major)
        kwargs: dict = {}
        if self.quant_mode == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        elif cuda:
            kwargs.update(dtype=torch.bfloat16, device_map="auto")
        else:
            kwargs["dtype"] = torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **kwargs).eval()

    def generate(self, image: Image.Image | None, prompt: str, system: str) -> str:
        import torch

        self.load()
        user = [{"type": "image", "image": image}] if image is not None else []
        user.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": user}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class FakeBackend:
    """Returns canned outputs in a cycle (tests and dry runs only)."""

    def __init__(self, outputs: Sequence[str] | None = None):
        self._outputs = itertools.cycle(list(outputs) if outputs else [FAKE_OUTPUT])
        self.calls: list[tuple[str, str]] = []
        self.images: list[Image.Image | None] = []
        self.loaded = True

    def generate(self, image: Image.Image | None, prompt: str, system: str) -> str:
        self.calls.append((prompt, system))
        self.images.append(image)
        return next(self._outputs)


def make_backend(cfg: dict):
    v = cfg["vlm"]
    if v["backend"] == "fake":
        return FakeBackend()
    if v["backend"] != "hf":
        raise ValueError(f"vlm.backend must be hf or fake, got {v['backend']!r}")
    return HFVLMBackend(v["model_name"], quantize=v["quantize"], max_new_tokens=v["max_new_tokens"],
                        allow_cpu=v["allow_cpu"])

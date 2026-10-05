"""Load the VLM backend on first use and turn every failure into an error dict."""
from __future__ import annotations

from collections.abc import Callable

from ..vlm.backend import resize_for_vlm
from ..vlm.extract import extract_one
from ..vlm.prompts import build_prompt


def choose_mode(has_image: bool, has_text: bool) -> str | None:
    """Same modes as sub-project 3: image_text if both inputs are given, else image or text."""
    if has_image and has_text:
        return "image_text"
    if has_image:
        return "image"
    return "text" if has_text else None


class LazyVLM:
    def __init__(self, factory: Callable[[], object], max_image_side: int = 768):
        self._factory = factory
        self._backend = None
        self._error: str | None = None
        self.max_image_side = int(max_image_side)
        self.factory_calls = 0

    def extract(self, image, text: str | None = None) -> dict:
        """`text` is the already masked text shown to the user."""
        text = (text or "").strip()
        mode = choose_mode(image is not None, bool(text))
        if mode is None:
            return {"error": "The VLM needs an image or a text"}
        if self._error:
            return {"error": self._error}
        if self._backend is None:
            self.factory_calls += 1
            try:
                backend = self._factory()
                if hasattr(backend, "load"):
                    backend.load()
                self._backend = backend
            except Exception as e:  # noqa: BLE001 - shown to the user instead of crashing the app
                self._error = f"Could not load the VLM: {e}. See the 'Examples' tab."
                return {"error": self._error}
        prompt = build_prompt(mode, text=text)
        img = resize_for_vlm(image, self.max_image_side) if mode != "text" else None
        rec = extract_one(self._backend, img, prompt)
        return {"mode": mode, **rec["output"]} if rec["valid"] else \
            {"mode": mode, "error": rec["error"], "raw": rec["raw"]}

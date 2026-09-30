# Sub-project 3 (VLM information extraction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Qwen3-VL-4B-Instruct locally in Colab to extract a fixed JSON schema (dish, cuisine, ingredients, cooking method, confidence) from 200 test images under 3 prompt modes (`image`, `text`, `image_text`), so image-only, text-only and multimodal extraction can be compared, and evaluate validity, latency, dish accuracy with a bootstrap CI (next to the classifiers scored on the same ids), ingredient grounding (reference only) and a blind manual-grading sheet.

**Architecture:** Sub-package `src/foodmm/vlm/` (parse, prompts, backend, extract, evaluate). The backend exposes one method `generate(image | None, prompt, system) -> str` (`None` for the text-only mode), so all logic is tested with a fake backend; the Hugging Face backend is exercised once with a tiny Qwen3-VL on CPU. Outputs are resumable JSONL files under `work_dir/vlm/<vlm.tag>/`.

**Tech Stack:** `transformers==5.17.0` (`AutoModelForImageTextToText`, `AutoProcessor`), `bitsandbytes==0.50.2` (Colab only, 4-bit), stdlib `difflib`.

**Spec:** `docs/superpowers/specs/2026-09-27-vlm-extraction-design.md`
**Depends on:** Milestone 1 (config, manifest, text utils, metrics, late_fusion `load_preds`); Milestone 2 plan including Task 8b (`foodmm/stats.py`); Milestone 2 runs `image`, `text_strict`, `xattn_strict` for the classifier comparison (configurable; missing runs are skipped, so extraction itself needs no Milestone 2 run).

**Scope reduction (2026-09-28):** the `image_hint` mode was removed; 200 samples (not 500) and 20 graded ids (not 50).

## Global Constraints

- Same CLI/test/commit conventions as Milestones 1–2.
- `MODES = ("image", "text", "image_text")`; `TEXT_MODES = ("text", "image_text")` (modes that see the sample text). `SCHEMA_KEYS = ("dish_name", "cuisine", "main_ingredients", "cooking_method", "confidence")`.
- JSONL record keys: `id, label, mode, valid, attempts, latency_s, output, raw, error` (`output` is the normalised dict or `null`).
- Output directory: `vlm_dir(cfg) = work_dir / "vlm" / vlm.tag`.
- `vlm.backend: hf | fake`. `fake` exists only so scripts can be tested end-to-end without a model.
- Tests marked `network` download `trl-internal-testing/tiny-Qwen3VLForConditionalGeneration` (~7 MB weights + tokenizer).

## File Map

| File | Responsibility |
|---|---|
| `configs/default.yaml` (append) | `vlm` section |
| `requirements-colab.txt` (replace) | add `bitsandbytes==0.50.2` |
| `src/foodmm/vlm/parse.py` | JSON extraction, validation, normalisation |
| `src/foodmm/vlm/prompts.py` | system prompt, 3 modes, retry prompt |
| `src/foodmm/vlm/backend.py` | `resize_for_vlm`, `resolve_quantize`, `HFVLMBackend`, `FakeBackend`, `make_backend` |
| `src/foodmm/vlm/extract.py` | sample selection, retry loop, resumable JSONL |
| `src/foodmm/vlm/evaluate.py` | dish mapping, grounding, summary with CI, classifiers on the same sample, blind grading sheet |
| `scripts/vlm_extract.py`, `scripts/vlm_evaluate.py` | CLI |
| `tools/build_notebook_vlm.py`, `notebooks/03_vlm.ipynb` | notebook |
| `tests/test_vlm_*.py`, `tests/test_notebooks.py` (replace) | tests |

---

### Task 1: Config, requirements and JSON parsing

**Files:**
- Modify: `configs/default.yaml` (append), `requirements-colab.txt` (replace)
- Create: `src/foodmm/vlm/__init__.py`, `src/foodmm/vlm/parse.py`
- Test: `tests/test_vlm_parse.py`

**Interfaces:**
- Produces: `SCHEMA_KEYS`, `MAX_INGREDIENTS = 15`, `class ExtractionError(ValueError)`, `extract_json(raw) -> dict`, `validate_output(obj) -> dict`, `parse_output(raw) -> dict`.

- [ ] **Step 1: Append to `configs/default.yaml`**

```yaml

vlm:
  model_name: Qwen/Qwen3-VL-4B-Instruct   # Qwen/Qwen3-VL-8B-Instruct also works
  tag: qwen3vl4b                          # output folder name under work_dir/vlm/
  backend: hf                             # hf | fake (tests only)
  quantize: auto                          # auto: bf16 on compute capability >= 8 (L4/A100), else 4-bit
  allow_cpu: false                        # true only for tests with a tiny model
  max_new_tokens: 256
  max_image_side: 768
  n_samples: 200
  text_mask: strict                       # text given to the text and image_text modes
  max_context_chars: 1000
  match_threshold: 0.6
  n_grading: 20
  n_boot: 1000                            # bootstrap resamples for the dish-accuracy CI
  compare_runs:                           # classifiers scored on the same sample ids (relative to paths.work_dir)
    image: clip/runs/image
    text: clip/runs/text_strict
    image_text: clip/runs/xattn_strict
  log_every: 20
```

- [ ] **Step 2: Replace `requirements-colab.txt`**

```txt
# torch, torchvision, scikit-learn, pandas, matplotlib, pyyaml, pillow, accelerate and kaggle are preinstalled on Colab
timm==1.0.30
transformers==5.17.0
bitsandbytes==0.50.2
```

- [ ] **Step 3: Write the failing test `tests/test_vlm_parse.py`**

```python
import pytest

from foodmm.vlm.parse import ExtractionError, extract_json, parse_output, validate_output

GOOD = {"dish_name": "Apple Pie", "cuisine": "American", "main_ingredients": ["Apples", " sugar ", ""],
        "cooking_method": "baked", "confidence": 0.8}


def test_extract_json_with_fence_and_extra_text():
    raw = 'Sure! Here it is:\n```json\n{"dish_name": "pie", "nested": {"a": [1, 2]}}\n```\nHope it helps.'
    assert extract_json(raw) == {"dish_name": "pie", "nested": {"a": [1, 2]}}


def test_extract_json_skips_broken_braces():
    assert extract_json('{broken {"dish_name": "x"}') == {"dish_name": "x"}
    with pytest.raises(ExtractionError, match="no JSON object"):
        extract_json("no braces here")
    with pytest.raises(ExtractionError):
        extract_json("{not json}")


def test_validate_normalises():
    out = validate_output({**GOOD, "extra": 1})
    assert out == {"dish_name": "Apple Pie", "cuisine": "American", "main_ingredients": ["apples", "sugar"],
                   "cooking_method": "baked", "confidence": 0.8}


def test_validate_conversions():
    out = validate_output({**GOOD, "main_ingredients": "eggs, Flour ,milk", "confidence": "85%", "cuisine": None})
    assert out["main_ingredients"] == ["eggs", "flour", "milk"]
    assert out["confidence"] == pytest.approx(0.85) and out["cuisine"] == ""
    assert validate_output({**GOOD, "confidence": 90})["confidence"] == pytest.approx(0.9)
    many = validate_output({**GOOD, "main_ingredients": [f"i{k}" for k in range(30)]})
    assert len(many["main_ingredients"]) == 15


@pytest.mark.parametrize("bad, reason", [
    ({k: v for k, v in GOOD.items() if k != "cuisine"}, "missing keys"),
    ({**GOOD, "dish_name": "  "}, "dish_name"),
    ({**GOOD, "main_ingredients": [1, 2]}, "main_ingredients"),
    ({**GOOD, "confidence": 250}, "confidence"),
    ({**GOOD, "confidence": True}, "confidence"),
    ({**GOOD, "cooking_method": ["a"]}, "cooking_method"),
])
def test_validate_rejects(bad, reason):
    with pytest.raises(ExtractionError, match=reason):
        validate_output(bad)


def test_parse_output_roundtrip():
    import json

    assert parse_output(json.dumps(GOOD))["dish_name"] == "Apple Pie"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vlm_parse.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.vlm'`.

- [ ] **Step 5: Implement `src/foodmm/vlm/__init__.py`**

```python
"""Structured food information extraction with a local vision-language model."""
```

- [ ] **Step 6: Implement `src/foodmm/vlm/parse.py`**

```python
"""Find, validate and normalise the JSON object returned by the VLM."""
from __future__ import annotations

import json
import re

SCHEMA_KEYS = ("dish_name", "cuisine", "main_ingredients", "cooking_method", "confidence")
MAX_INGREDIENTS = 15
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


class ExtractionError(ValueError):
    pass


def extract_json(raw: str) -> dict:
    """Return the first JSON object in `raw`; tolerates code fences and text around it."""
    text = _FENCE.sub("", str(raw))
    starts = [m.start() for m in re.finditer(r"\{", text)]
    if not starts:
        raise ExtractionError("no JSON object found")
    decoder = json.JSONDecoder()
    for i in starts:
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ExtractionError("invalid JSON")


def _optional_str(obj: dict, key: str) -> str:
    v = obj[key]
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ExtractionError(f"{key} must be a string")
    return v.strip()


def _confidence(v) -> float:
    percent = False
    if isinstance(v, str):
        s = v.strip()
        percent = s.endswith("%")
        try:
            v = float(s.rstrip("%"))
        except ValueError as e:
            raise ExtractionError("confidence must be a number") from e
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ExtractionError("confidence must be a number")
    v = float(v)
    if percent or 1.0 < v <= 100.0:
        v /= 100.0
    if not 0.0 <= v <= 1.0:
        raise ExtractionError("confidence must be in [0, 1]")
    return round(v, 4)


def validate_output(obj: dict) -> dict:
    missing = [k for k in SCHEMA_KEYS if k not in obj]
    if missing:
        raise ExtractionError(f"missing keys: {', '.join(missing)}")
    dish = obj["dish_name"]
    if not isinstance(dish, str) or not dish.strip():
        raise ExtractionError("dish_name must be a non-empty string")
    ingredients = obj["main_ingredients"]
    if isinstance(ingredients, str):
        ingredients = ingredients.split(",")
    if not isinstance(ingredients, list) or not all(isinstance(x, str) for x in ingredients):
        raise ExtractionError("main_ingredients must be a list of strings")
    ingredients = [x.strip().lower() for x in ingredients if x.strip()][:MAX_INGREDIENTS]
    return {"dish_name": dish.strip(), "cuisine": _optional_str(obj, "cuisine"), "main_ingredients": ingredients,
            "cooking_method": _optional_str(obj, "cooking_method"), "confidence": _confidence(obj["confidence"])}


def parse_output(raw: str) -> dict:
    return validate_output(extract_json(raw))
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vlm_parse.py tests/test_config.py -v`
Expected: 17 passed.

- [ ] **Step 8: Commit**

```bash
git add configs/default.yaml requirements-colab.txt src/foodmm/vlm/__init__.py src/foodmm/vlm/parse.py tests/test_vlm_parse.py
git commit -m "feat(vlm): config section and JSON output parsing"
```

---

### Task 2: Prompts and backends

**Files:**
- Create: `src/foodmm/vlm/prompts.py`, `src/foodmm/vlm/backend.py`
- Modify: `tests/helpers.py` (append `TINY_VLM_MODEL`)
- Test: `tests/test_vlm_prompts.py`, `tests/test_vlm_backend.py`

**Interfaces:**
- Produces (`prompts`): `MODES`, `TEXT_MODES`, `SYSTEM_PROMPT`, `RETRY_PROMPT` (format key `error`), `build_prompt(mode, text=None, max_chars=1000) -> str`.
- Produces (`backend`): `resize_for_vlm(img, max_side) -> Image`, `resolve_quantize(quantize, cuda, capability_major) -> "none" | "4bit"`, `class HFVLMBackend(model_name, quantize="auto", max_new_tokens=256, allow_cpu=False)` with `.load()`, `.loaded`, `.generate(image | None, prompt, system) -> str`; `class FakeBackend(outputs=None)` with `.calls` (prompt, system) and `.images`; `make_backend(cfg) -> backend`.

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
TINY_VLM_MODEL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_vlm_prompts.py`:
```python
import pytest

from foodmm.vlm.prompts import MODES, RETRY_PROMPT, SYSTEM_PROMPT, TEXT_MODES, build_prompt


def test_system_prompt_lists_schema():
    for key in ("dish_name", "cuisine", "main_ingredients", "cooking_method", "confidence"):
        assert key in SYSTEM_PROMPT
    assert "{error}" in RETRY_PROMPT


def test_modes():
    assert MODES == ("image", "text", "image_text") and TEXT_MODES == ("text", "image_text")
    base = build_prompt("image")
    assert "JSON" in base and "[MASK]" not in base and "photo" in base
    text_only = build_prompt("text", text="crispy [MASK] with salt")
    assert "photo" not in text_only and "crispy [MASK] with salt" in text_only
    with_text = build_prompt("image_text", text="x" * 5000, max_chars=100)
    assert "x" * 100 in with_text and "x" * 101 not in with_text and "[MASK]" in with_text
    assert "(no text)" in build_prompt("image_text", text="")


def test_bad_mode():
    for mode in ("text_only", "image_hint"):
        with pytest.raises(ValueError, match="unknown mode"):
            build_prompt(mode)
```

`tests/test_vlm_backend.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_vlm_prompts.py tests/test_vlm_backend.py -v`
Expected: ERROR `ModuleNotFoundError` for `foodmm.vlm.prompts` / `foodmm.vlm.backend`.

- [ ] **Step 4: Implement `src/foodmm/vlm/prompts.py`**

```python
"""Prompts for the three extraction modes."""
from __future__ import annotations

MODES = ("image", "text", "image_text")
TEXT_MODES = ("text", "image_text")  # modes whose prompt contains the sample text

SYSTEM_PROMPT = (
    "You are a food analysis assistant. You get a food photo, a text about a dish, or both. "
    "Answer with ONLY one JSON object "
    "(no markdown, no explanation) with exactly these keys:\n"
    '{"dish_name": string, "cuisine": string, "main_ingredients": [string, ...], '
    '"cooking_method": string, "confidence": number between 0 and 1}\n'
    "Write every value in English. List at most 10 main ingredients that are shown, mentioned or very likely."
)

RETRY_PROMPT = (
    "Your previous answer was not valid ({error}). Reply again with ONLY the JSON object with the keys "
    "dish_name, cuisine, main_ingredients, cooking_method and confidence."
)

_TEXT_CONTEXT = ("Text from a web page about this dish. It may be noisy, and dish names were "
                 "replaced by [MASK]:\n\"\"\"\n{text}\n\"\"\"")


def build_prompt(mode: str, text: str | None = None, max_chars: int = 1000) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    first = "Identify the dish described in this text and describe it." if mode == "text" else \
        "Identify the dish in this photo and describe it."
    parts = [first]
    if mode in TEXT_MODES:
        snippet = str(text or "").strip()[:max_chars] or "(no text)"
        parts.append(_TEXT_CONTEXT.format(text=snippet))
    parts.append("Answer with the JSON object only.")
    return "\n\n".join(parts)
```

- [ ] **Step 5: Implement `src/foodmm/vlm/backend.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vlm_prompts.py tests/test_vlm_backend.py -v`
Expected: 8 passed (the `network` test downloads the tiny Qwen3-VL; about 30 s on CPU).

- [ ] **Step 7: Commit**

```bash
git add src/foodmm/vlm/prompts.py src/foodmm/vlm/backend.py tests/helpers.py tests/test_vlm_prompts.py tests/test_vlm_backend.py
git commit -m "feat(vlm): prompt modes, lazy Qwen3-VL backend and fake backend"
```

---

### Task 3: Sample selection and the resumable extraction loop

**Files:**
- Create: `src/foodmm/vlm/extract.py`
- Test: `tests/test_vlm_extract.py`

**Interfaces:**
- Consumes: `work_paths`, `text_column`, `save_json`/`load_json` (M1); `parse_output`, `ExtractionError` (Task 1); `MODES`, `SYSTEM_PROMPT`, `RETRY_PROMPT`, `build_prompt`, `resize_for_vlm` (Task 2).
- Produces: `vlm_dir(cfg) -> Path`, `select_sample(df, n, seed, path) -> list[str]`, `read_records(path) -> list[dict]`, `extract_one(backend, image | None, prompt) -> dict[valid, attempts, latency_s, output, raw, error]`, `run_extraction(cfg, mode, backend, df, classes, n=None) -> dict[mode, path, n_total, n_new, n_valid]`.

- [ ] **Step 1: Write the failing test `tests/test_vlm_extract.py`**

```python
import json

import pandas as pd
import pytest
from PIL import Image

from foodmm.config import load_config
from foodmm.data.prepare import load_manifest
from foodmm.utils import load_json
from foodmm.vlm.backend import FAKE_OUTPUT, FakeBackend
from foodmm.vlm.extract import extract_one, read_records, run_extraction, select_sample, vlm_dir
from helpers import CLASSES, make_fake_dataset, run_script, smoke_overrides

IMG = Image.new("RGB", (8, 8))


def test_extract_one_retry_then_valid():
    fb = FakeBackend(["not json at all", FAKE_OUTPUT])
    rec = extract_one(fb, IMG, "prompt")
    assert rec["valid"] and rec["attempts"] == 2 and rec["output"]["dish_name"] == "apple pie"
    assert "not valid" in fb.calls[1][0] and rec["error"] is None


def test_extract_one_gives_up_after_retry():
    rec = extract_one(FakeBackend(["nope"]), IMG, "prompt")
    assert not rec["valid"] and rec["attempts"] == 2 and rec["output"] is None
    assert rec["raw"] == "nope" and "no JSON" in rec["error"] and rec["latency_s"] >= 0


def test_select_sample_is_stratified_and_cached(tmp_path):
    df = pd.DataFrame({"id": [f"{c}_{i}" for c in "abc" for i in range(10)], "label": [c for c in "abc" for _ in range(10)],
                       "split": "test"})
    path = tmp_path / "sample_ids.json"
    ids = select_sample(df, 6, seed=0, path=path)
    assert len(ids) == 6 and sorted(i[0] for i in ids) == ["a", "a", "b", "b", "c", "c"]
    assert load_json(path) == ids
    assert select_sample(df, 3, seed=1, path=path) == ids  # cached file wins
    assert len(select_sample(df, 7, seed=0, path=tmp_path / "x.json")) == 7


@pytest.fixture()
def env(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work) + ["vlm.backend=fake", "vlm.n_samples=6", "vlm.log_every=2"]
    run_script("prepare_data.py", "--set", *sets)
    cfg = load_config(overrides=sets)
    return cfg, load_manifest(work / "data" / "manifest.csv")


def test_run_extraction_resumes(env):
    cfg, df = env
    fb = FakeBackend()
    res = run_extraction(cfg, "image", fb, df, CLASSES)
    assert res["n_new"] == 6 and res["n_total"] == 6 and res["n_valid"] == 6
    recs = read_records(vlm_dir(cfg) / "extract_image.jsonl")
    assert [r["mode"] for r in recs] == ["image"] * 6 and all("label" in r for r in recs)
    assert "photo" in fb.calls[0][0] and all(im is not None for im in fb.images)
    again = run_extraction(cfg, "image", FakeBackend(), df, CLASSES)
    assert again["n_new"] == 0 and again["n_total"] == 6
    txt = run_extraction(cfg, "image_text", FakeBackend(), df, CLASSES, n=2)
    assert txt["n_new"] == 2
    line = (vlm_dir(cfg) / "extract_image_text.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["output"]["cuisine"] == "American"
    tb = FakeBackend()
    only_text = run_extraction(cfg, "text", tb, df, CLASSES, n=3)
    assert only_text["n_new"] == 3 and tb.images == [None] * 3  # the text-only mode never sends the image
    assert "Identify the dish described in this text" in tb.calls[0][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vlm_extract.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.vlm.extract'`.

- [ ] **Step 3: Implement `src/foodmm/vlm/extract.py`**

```python
"""Run the VLM over a fixed test sample; one JSONL line per image, resumable."""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import work_paths
from ..data.text_utils import text_column
from ..utils import ensure_dir, load_json, save_json
from .backend import resize_for_vlm
from .parse import ExtractionError, parse_output
from .prompts import MODES, RETRY_PROMPT, SYSTEM_PROMPT, build_prompt


def vlm_dir(cfg: dict) -> Path:
    return work_paths(cfg)["work_dir"] / "vlm" / str(cfg["vlm"]["tag"])


def select_sample(df: pd.DataFrame, n: int, seed: int, path: str | Path) -> list[str]:
    """Stratified sample of test ids (about n / n_classes per class), cached in `path`."""
    path = Path(path)
    if path.exists():
        return load_json(path)
    rng = np.random.default_rng(seed)
    test = df[df["split"] == "test"]
    groups = {c: g["id"].astype(str).tolist() for c, g in test.groupby("label", sort=True)}
    per_class = max(1, n // max(1, len(groups)))
    chosen = []
    for ids in groups.values():
        chosen += list(rng.choice(ids, size=min(per_class, len(ids)), replace=False))
    rest = sorted(set(test["id"].astype(str)) - set(chosen))
    if len(chosen) < n and rest:
        chosen += list(rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False))
    chosen = [str(i) for i in rng.permutation(chosen)[:n]]
    save_json(chosen, path)
    return chosen


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_one(backend, image: Image.Image | None, prompt: str) -> dict:
    t0 = time.time()
    raw, error, output, attempts = "", None, None, 0
    for attempt in range(2):
        attempts = attempt + 1
        p = prompt if attempt == 0 else f"{prompt}\n\n{RETRY_PROMPT.format(error=error)}"
        raw = backend.generate(image, p, SYSTEM_PROMPT)
        try:
            output, error = parse_output(raw), None
            break
        except ExtractionError as e:
            error = str(e)
    return {"valid": output is not None, "attempts": attempts, "latency_s": round(time.time() - t0, 3),
            "output": output, "raw": raw, "error": error}
```

- [ ] **Step 4: Append to `src/foodmm/vlm/extract.py`**

```python
def run_extraction(cfg: dict, mode: str, backend, df: pd.DataFrame, classes: Sequence[str],
                   n: int | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    v = cfg["vlm"]
    out_dir = ensure_dir(vlm_dir(cfg))
    ids = select_sample(df, int(v["n_samples"]), int(cfg["seed"]), out_dir / "sample_ids.json")
    if n is not None:
        ids = ids[:int(n)]
    out_path = out_dir / f"extract_{mode}.jsonl"
    done = {r["id"] for r in read_records(out_path)}
    todo = [i for i in ids if i not in done]
    rows = df.set_index(df["id"].astype(str))
    col = text_column(v["text_mask"])
    data_root = Path(cfg["paths"]["data_root"])
    n_valid, t_start = 0, time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for k, sid in enumerate(todo, 1):
            row = rows.loc[sid]
            prompt = build_prompt(mode, text=row[col], max_chars=int(v["max_context_chars"]))
            if mode == "text":
                res = extract_one(backend, None, prompt)
            else:
                try:
                    with Image.open(data_root / row["image_path"]) as im:
                        image = resize_for_vlm(im, int(v["max_image_side"]))
                except Exception:  # noqa: BLE001
                    res = {"valid": False, "attempts": 0, "latency_s": None, "output": None, "raw": "",
                           "error": "bad image"}
                else:
                    res = extract_one(backend, image, prompt)
            n_valid += bool(res["valid"])
            rec = {"id": sid, "label": row["label"], "mode": mode, **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % int(v["log_every"]) == 0 or k == len(todo):
                print(f"[{mode}] {k}/{len(todo)} new, valid {n_valid}/{k}, "
                      f"{(time.time() - t_start) / k:.2f}s per image", flush=True)
    all_recs = read_records(out_path)
    return {"mode": mode, "path": str(out_path), "n_total": len(all_recs), "n_new": len(todo),
            "n_valid": sum(bool(r["valid"]) for r in all_recs)}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vlm_extract.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/foodmm/vlm/extract.py tests/test_vlm_extract.py
git commit -m "feat(vlm): stratified sample and resumable extraction loop"
```

---

### Task 4: Evaluation (dish mapping, grounding, summary, classifiers on the sample, blind grading sheet)

**Files:**
- Create: `src/foodmm/vlm/evaluate.py`
- Test: `tests/test_vlm_evaluate.py`

**Interfaces:**
- Consumes: `class_to_phrase`, `phrase_variants`, `word_variants`, `STOPWORDS`, `clean_text` (M1 text utils); `load_preds` (M1); `bootstrap_ci`, `correct_vector` (M2 Task 8b, `foodmm/stats.py`); `read_records`, `vlm_dir`, `MODES`, `TEXT_MODES`.
- Produces: `normalize_name(s) -> str`, `map_dish_to_class(dish_name, classes, threshold=0.6) -> str | None`, `ingredient_grounding(ingredients, text) -> float | None`, `summarize_mode(records, classes, texts, threshold=0.6, n_boot=1000, seed=0) -> dict[mode, uses_text, n, valid_rate, retry_rate, latency_mean, latency_p90, dish_acc, dish_acc_lo, dish_acc_hi, unmapped_rate, ingredient_grounding]`, `SUMMARY_COLUMNS`, `CLASSIFIER_COLUMNS`, `classifier_on_sample(work_dir, runs, ids, n_boot=1000, seed=0) -> DataFrame[name, run, n, acc, acc_lo, acc_hi]`, `GRADING_COLUMNS`, `GRADES`, `grading_key_path(path) -> Path`, `make_grading_sheet(records_by_mode, ids, df, path, seed=0) -> bool`, `score_grading_sheet(path) -> DataFrame | None`, `evaluate_all(cfg, df, classes) -> DataFrame`.
- Notes:
  - `dish_acc` counts invalid and unmapped outputs as wrong; its 95% CI is a bootstrap over the sample.
  - `ingredient_grounding` is measured against the unmasked sample text, so it is inflated for modes that read that text (`uses_text = True`). It is reported as a reference only; manual grades are the metric used to compare modes.
  - The grading sheet is blind: rows are shuffled and the `mode` column lives in a separate key file (`manual_grading_key.csv`) that is joined back only when scoring.

- [ ] **Step 1: Write the failing test `tests/test_vlm_evaluate.py`**

```python
import math

import numpy as np
import pandas as pd
import pytest

from foodmm.vlm.evaluate import (
    GRADES, GRADING_COLUMNS, classifier_on_sample, grading_key_path, ingredient_grounding, make_grading_sheet,
    map_dish_to_class, normalize_name, score_grading_sheet, summarize_mode,
)

CLS = ["apple_pie", "caesar_salad", "french_fries", "spaghetti_bolognese", "pie"]


def test_normalize_name():
    assert normalize_name("  Apple-Pie!! (Homemade) ") == "apple pie homemade"


@pytest.mark.parametrize("name, expected", [
    ("Apple Pie", "apple_pie"),
    ("Classic homemade apple pies", "apple_pie"),        # plural, longest phrase wins over "pie"
    ("Caesar salad with croutons", "caesar_salad"),
    ("spagetti bolognese", "spaghetti_bolognese"),       # typo -> fuzzy
    ("fries", "french_fries"),                           # word overlap
    ("Beef Wellington", None),                           # unmapped
])
def test_map_dish_to_class(name, expected):
    assert map_dish_to_class(name, CLS, threshold=0.6) == expected


def test_ingredient_grounding():
    text = "Grandma's Apple Pie: slice the apples, add sugar and butter. http://x.com"
    assert ingredient_grounding(["apple", "brown sugar", "cinnamon"], text) == pytest.approx(2 / 3)
    assert ingredient_grounding([], text) is None


def _rec(i, label, valid=True, dish="apple pie", attempts=1, latency=1.0, ingr=("apple",), mode="image"):
    out = {"dish_name": dish, "cuisine": "x", "main_ingredients": list(ingr), "cooking_method": "baked",
           "confidence": 0.5} if valid else None
    return {"id": i, "label": label, "mode": mode, "valid": valid, "attempts": attempts,
            "latency_s": latency, "output": out, "raw": "", "error": None if valid else "bad"}


def test_summarize_mode():
    recs = [_rec("a", "apple_pie"), _rec("b", "apple_pie", dish="beef wellington", latency=3.0),
            _rec("c", "french_fries", valid=False, attempts=2, latency=2.0), _rec("d", "french_fries", dish="fries")]
    texts = {"a": "apple pie recipe", "b": "no match here", "d": "crispy fries"}
    s = summarize_mode(recs, CLS, texts, n_boot=200, seed=0)
    assert s["mode"] == "image" and s["n"] == 4 and s["uses_text"] is False
    assert s["valid_rate"] == 0.75 and s["retry_rate"] == 0.25
    assert s["dish_acc"] == 0.5 and s["unmapped_rate"] == pytest.approx(1 / 3)
    assert 0.0 <= s["dish_acc_lo"] <= 0.5 <= s["dish_acc_hi"] <= 1.0
    assert s["latency_mean"] == pytest.approx(1.75) and s["latency_p90"] == pytest.approx(2.7)
    assert s["ingredient_grounding"] == pytest.approx(1 / 3)
    assert summarize_mode([{**r, "mode": "text"} for r in recs], CLS, texts, n_boot=10)["uses_text"] is True
    empty = summarize_mode([], CLS, {})
    assert empty["n"] == 0 and math.isnan(empty["dish_acc"])


def test_classifier_on_sample(tmp_path, capsys):
    run = tmp_path / "clip" / "runs" / "image"
    run.mkdir(parents=True)
    logits = np.zeros((4, 3), dtype=np.float32)
    logits[[0, 1, 2, 3], [0, 1, 0, 2]] = 1.0
    np.savez(run / "preds_test.npz", logits=logits, labels=np.array([0, 1, 1, 2]), ids=np.array(["a", "b", "c", "d"]))
    out = classifier_on_sample(tmp_path, {"image": "clip/runs/image", "text": "clip/runs/missing"}, ["a", "c", "d"],
                               n_boot=100, seed=0)
    assert out["name"].tolist() == ["image"] and "Skipping classifier 'text'" in capsys.readouterr().out
    row = out.iloc[0]
    assert row["run"] == "clip/runs/image" and row["n"] == 3 and row["acc"] == pytest.approx(2 / 3)
    assert row["acc_lo"] <= row["acc"] <= row["acc_hi"]


def test_blind_grading_sheet_roundtrip(tmp_path):
    df = pd.DataFrame({"id": ["a", "b"], "image_path": ["x/a.jpg", "x/b.jpg"], "label": ["apple_pie"] * 2})
    recs = {"image": [_rec("a", "apple_pie"), _rec("b", "apple_pie", valid=False)],
            "text": [_rec("a", "apple_pie", mode="text")]}
    path = tmp_path / "manual_grading.csv"
    assert make_grading_sheet(recs, ["a", "b"], df, path, seed=0) is True
    sheet, key = pd.read_csv(path), pd.read_csv(grading_key_path(path))
    assert list(sheet.columns) == GRADING_COLUMNS and "mode" not in sheet.columns and len(sheet) == 3
    assert sorted(sheet["row"]) == [0, 1, 2] and list(key.columns) == ["row", "mode"]
    joined = sheet.merge(key, on="row")

    def where(sid, mode):
        row = joined.loc[(joined["id"] == sid) & (joined["mode"] == mode), "row"].iloc[0]
        return sheet["row"] == row

    assert sheet.loc[where("a", "image"), "main_ingredients"].iloc[0] == "apple"
    assert sheet.loc[where("a", "image"), "image_path"].iloc[0] == "x/a.jpg"
    assert make_grading_sheet(recs, ["a"], df, path) is False  # never overwrite a sheet
    assert score_grading_sheet(path) is None  # nothing graded yet
    sheet.loc[where("a", "image"), GRADES] = [1, 2, 1]
    sheet.loc[where("a", "text"), GRADES] = [0, 1, 1]
    sheet.to_csv(path, index=False)
    scores = score_grading_sheet(path).set_index("mode")
    assert scores.loc["image", "grade_dish"] == 1.0 and scores.loc["image", "n_graded"] == 1
    assert scores.loc["text", "grade_ingredients"] == 1.0
    grading_key_path(path).unlink()
    with pytest.raises(SystemExit, match="manual_grading_key.csv"):
        score_grading_sheet(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vlm_evaluate.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.vlm.evaluate'`.

- [ ] **Step 3: Implement `src/foodmm/vlm/evaluate.py`**

```python
"""Evaluate VLM extractions: validity, latency, dish accuracy, ingredient grounding, manual grades."""
from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.text_utils import STOPWORDS, class_to_phrase, clean_text, phrase_variants, word_variants
from ..late_fusion import load_preds
from ..stats import bootstrap_ci, correct_vector
from .extract import read_records, vlm_dir
from .prompts import MODES, TEXT_MODES

SUMMARY_COLUMNS = ["mode", "uses_text", "n", "valid_rate", "retry_rate", "latency_mean", "latency_p90", "dish_acc",
                   "dish_acc_lo", "dish_acc_hi", "unmapped_rate", "ingredient_grounding"]
CLASSIFIER_COLUMNS = ["name", "run", "n", "acc", "acc_lo", "acc_hi"]
GRADING_COLUMNS = ["row", "id", "label", "image_path", "dish_name", "cuisine", "main_ingredients",
                   "cooking_method", "confidence", "grade_dish", "grade_ingredients", "grade_method", "notes"]
GRADES = ["grade_dish", "grade_ingredients", "grade_method"]
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(s: str) -> str:
    return _NON_ALNUM.sub(" ", str(s).lower()).strip()


def _stem_words(s: str) -> set[str]:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split() if w not in STOPWORDS}


def map_dish_to_class(dish_name: str, classes: Sequence[str], threshold: float = 0.6) -> str | None:
    name = normalize_name(dish_name)
    if not name:
        return None
    hits = []
    for c in classes:
        for v in phrase_variants(class_to_phrase(c)):
            if re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", name):
                hits.append((len(v), c))
    if hits:
        return max(hits)[1]
    best, best_score = None, 0.0
    words = _stem_words(name)
    for c in classes:
        phrase = class_to_phrase(c)
        ratio = SequenceMatcher(None, name, phrase).ratio()
        cw = _stem_words(phrase)
        overlap = len(words & cw) / min(len(words), len(cw)) if words and cw else 0.0  # overlap coefficient
        score = max(ratio, overlap)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None


def ingredient_grounding(ingredients: Sequence[str], text: str) -> float | None:
    if not ingredients:
        return None
    tokens = set(re.findall(r"[a-z]+", clean_text(text)))
    matched = 0
    for ing in ingredients:
        words = [w for w in re.findall(r"[a-z]+", str(ing).lower()) if len(w) >= 3 and w not in STOPWORDS]
        if any(word_variants(w) & tokens for w in words):
            matched += 1
    return matched / len(ingredients)


def summarize_mode(records: Sequence[dict], classes: Sequence[str], texts: dict[str, str],
                   threshold: float = 0.6, n_boot: int = 1000, seed: int = 0) -> dict:
    n = len(records)
    mode = records[0]["mode"] if records else ""
    if n == 0:
        return {"mode": mode, "uses_text": mode in TEXT_MODES, "n": 0,
                **{k: float("nan") for k in SUMMARY_COLUMNS[3:]}}
    valid = [r for r in records if r["valid"]]
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    mapped = [map_dish_to_class(r["output"]["dish_name"], classes, threshold) if r["valid"] else None
              for r in records]
    correct = np.array([m is not None and m == r["label"] for m, r in zip(mapped, records)])
    acc_lo, acc_hi = bootstrap_ci(correct, n_boot, seed=seed)
    grounding = [g for r in valid
                 if (g := ingredient_grounding(r["output"]["main_ingredients"], texts.get(r["id"], ""))) is not None]
    return {
        "mode": mode, "uses_text": mode in TEXT_MODES, "n": n,
        "valid_rate": len(valid) / n,
        "retry_rate": sum(r["attempts"] > 1 for r in records) / n,
        "latency_mean": float(np.mean(lat)) if lat else float("nan"),
        "latency_p90": float(np.percentile(lat, 90)) if lat else float("nan"),
        "dish_acc": float(correct.mean()), "dish_acc_lo": acc_lo, "dish_acc_hi": acc_hi,
        "unmapped_rate": (sum(m is None for m, r in zip(mapped, records) if r["valid"]) / len(valid)
                          if valid else float("nan")),
        "ingredient_grounding": float(np.mean(grounding)) if grounding else float("nan"),
    }


def classifier_on_sample(work_dir: str | Path, runs: dict[str, str], ids: Sequence[str], n_boot: int = 1000,
                         seed: int = 0) -> pd.DataFrame:
    """Accuracy of trained classifiers on exactly the VLM sample ids; runs without predictions are skipped."""
    rows = []
    for name, rel in runs.items():
        try:
            preds = load_preds(Path(work_dir) / rel, "test")
        except FileNotFoundError:
            print(f"Skipping classifier '{name}': no predictions in {rel}")
            continue
        by_id = dict(zip(preds["ids"].astype(str), correct_vector(preds)))
        c = np.array([by_id[i] for i in ids if i in by_id], dtype=bool)
        lo, hi = bootstrap_ci(c, n_boot, seed=seed)
        rows.append({"name": name, "run": rel, "n": len(c), "acc": float(c.mean()) if len(c) else float("nan"),
                     "acc_lo": lo, "acc_hi": hi})
    return pd.DataFrame(rows, columns=CLASSIFIER_COLUMNS)
```

- [ ] **Step 4: Append to `src/foodmm/vlm/evaluate.py` (blind grading sheet and driver)**

```python
def grading_key_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_key.csv")


def make_grading_sheet(records_by_mode: dict[str, list[dict]], ids: Sequence[str], df: pd.DataFrame,
                       path: str | Path, seed: int = 0) -> bool:
    """Write the shuffled, mode-blind grading CSV plus its key file once (False if the sheet already exists)."""
    path = Path(path)
    if path.exists():
        return False
    meta = df.set_index(df["id"].astype(str))
    by_mode = {m: {r["id"]: r for r in records_by_mode.get(m, [])} for m in MODES}
    rows = []
    for sid in ids:
        for mode in MODES:
            rec = by_mode[mode].get(sid)
            if rec is None:
                continue
            out = rec["output"] or {}
            rows.append({"mode": mode, "id": sid, "label": rec["label"], "image_path": meta.loc[sid, "image_path"],
                         "dish_name": out.get("dish_name", ""), "cuisine": out.get("cuisine", ""),
                         "main_ingredients": "; ".join(out.get("main_ingredients", [])),
                         "cooking_method": out.get("cooking_method", ""), "confidence": out.get("confidence"),
                         "grade_dish": None, "grade_ingredients": None, "grade_method": None,
                         "notes": "" if rec["valid"] else f"invalid: {rec['error']}"})
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = [{**rows[i], "row": k} for k, i in enumerate(order)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=GRADING_COLUMNS).to_csv(path, index=False)
    pd.DataFrame(rows, columns=["row", "mode"]).to_csv(grading_key_path(path), index=False)
    return True


def score_grading_sheet(path: str | Path) -> pd.DataFrame | None:
    key_path = grading_key_path(path)
    if not key_path.exists():
        raise SystemExit(f"Missing {key_path}: it maps each graded row back to its prompt mode")
    sheet = pd.read_csv(path)
    graded = sheet[sheet[GRADES].notna().any(axis=1)].merge(pd.read_csv(key_path), on="row")
    if graded.empty:
        return None
    scores = graded.groupby("mode")[GRADES].mean()
    scores["n_graded"] = graded.groupby("mode").size()
    return scores.reset_index()


_PCT = {"valid_rate", "retry_rate", "dish_acc", "dish_acc_lo", "dish_acc_hi", "unmapped_rate",
        "ingredient_grounding", "acc", "acc_lo", "acc_hi"}


def _markdown(table: pd.DataFrame) -> str:
    def fmt(col, v):
        if col in ("latency_mean", "latency_p90"):
            return f"{v:.2f}s"
        return f"{v * 100:.1f}%" if col in _PCT else str(v)

    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, r[c]) for c in cols) + " |" for r in table.to_dict("records")]
    return "\n".join(lines)


def evaluate_all(cfg: dict, df: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    from ..config import work_paths
    from ..utils import load_json

    out_dir = vlm_dir(cfg)
    v, seed = cfg["vlm"], int(cfg["seed"])
    texts = dict(zip(df["id"].astype(str), df["text"].astype(str)))
    records = {m: read_records(out_dir / f"extract_{m}.jsonl") for m in MODES}
    records = {m: r for m, r in records.items() if r}
    if not records:
        raise SystemExit(f"No extraction results in {out_dir}: run scripts/vlm_extract.py first")
    summary = pd.DataFrame([summarize_mode(r, classes, texts, float(v["match_threshold"]), int(v["n_boot"]), seed)
                            for r in records.values()], columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    note = ("\n_ingredient_grounding is a reference only: modes with uses_text=True read the text it is measured "
            "against. Compare modes with the manual grades._\n")
    (out_dir / "eval_summary.md").write_text(_markdown(summary) + "\n" + note, encoding="utf-8")
    sample_ids = load_json(out_dir / "sample_ids.json")
    clf = classifier_on_sample(work_paths(cfg)["work_dir"], dict(v["compare_runs"]), sample_ids, int(v["n_boot"]), seed)
    clf.to_csv(out_dir / "classifier_on_sample.csv", index=False)
    (out_dir / "classifier_on_sample.md").write_text(_markdown(clf) + "\n", encoding="utf-8")
    sheet = out_dir / "manual_grading.csv"
    if make_grading_sheet(records, sample_ids[:int(v["n_grading"])], df, sheet, seed=seed):
        print(f"Wrote {sheet} (mode-blind, shuffled): fill the grade_* columns, then run vlm_evaluate.py again")
    scores = score_grading_sheet(sheet)
    if scores is not None:
        scores.to_csv(out_dir / "manual_scores.csv", index=False)
        print(scores.to_string(index=False))
    return summary
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vlm_evaluate.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/foodmm/vlm/evaluate.py tests/test_vlm_evaluate.py
git commit -m "feat(vlm): dish accuracy with CI, classifiers on the same sample and a blind grading sheet"
```

---

### Task 5: `vlm_extract.py`, `vlm_evaluate.py` and pipeline test

**Files:**
- Create: `scripts/vlm_extract.py`, `scripts/vlm_evaluate.py`
- Test: `tests/test_vlm_pipeline.py`

**Interfaces:**
- CLI: `vlm_extract.py [--mode image|text|image_text|all] [--n N] --set ...` (prints one summary line per mode); `vlm_evaluate.py --set ...` (prints the summary table, writes `eval_summary.{csv,md}`, `classifier_on_sample.{csv,md}`, `manual_grading.csv` + `manual_grading_key.csv`, and `manual_scores.csv` once grades exist).

- [ ] **Step 1: Write the failing test `tests/test_vlm_pipeline.py`**

```python
import numpy as np
import pandas as pd

from foodmm.data.prepare import load_manifest
from helpers import make_fake_dataset, run_script, smoke_overrides


def test_vlm_scripts_with_fake_backend(tmp_path):
    data_root = make_fake_dataset(tmp_path / "ds")
    work = tmp_path / "work"
    sets = smoke_overrides(data_root, work) + ["vlm.backend=fake", "vlm.n_samples=6", "vlm.n_grading=2"]
    run_script("prepare_data.py", "--set", *sets)

    test = load_manifest(work / "data" / "manifest.csv").query("split == 'test'")
    run_dir = work / "clip" / "runs" / "xattn_strict"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "preds_test.npz", logits=np.random.default_rng(0).normal(size=(len(test), 3)).astype(np.float32),
             labels=test["label_idx"].to_numpy(), ids=test["id"].to_numpy().astype(str))

    out = run_script("vlm_extract.py", "--mode", "all", "--set", *sets).stdout
    assert out.count("n_new=6") == 3
    assert run_script("vlm_extract.py", "--mode", "image", "--set", *sets).stdout.count("n_new=0") == 1

    run_script("vlm_evaluate.py", "--set", *sets)
    out_dir = work / "vlm" / "qwen3vl4b"
    summary = pd.read_csv(out_dir / "eval_summary.csv")
    assert summary["mode"].tolist() == ["image", "text", "image_text"]
    assert (summary["valid_rate"] == 1.0).all() and summary["uses_text"].tolist() == [False, True, True]
    assert len(pd.read_csv(out_dir / "manual_grading.csv")) == 6  # 2 ids x 3 modes
    assert (out_dir / "manual_grading_key.csv").exists()
    assert (out_dir / "eval_summary.md").read_text().startswith("| mode |")
    clf = pd.read_csv(out_dir / "classifier_on_sample.csv")
    assert clf["name"].tolist() == ["image_text"] and clf["n"].iloc[0] == 6  # only xattn_strict exists here
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vlm_pipeline.py -v`
Expected: FAIL (`vlm_extract.py` does not exist).

- [ ] **Step 3: Implement `scripts/vlm_extract.py`**

```python
#!/usr/bin/env python
"""Extract structured food information with a local VLM (resumable JSONL per prompt mode)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.utils import load_json  # noqa: E402
from foodmm.vlm.backend import make_backend  # noqa: E402
from foodmm.vlm.extract import run_extraction  # noqa: E402
from foodmm.vlm.prompts import MODES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=[*MODES, "all"])
    parser.add_argument("--n", type=int, default=None, help="only the first N ids of the sample")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    if not paths["manifest"].exists():
        raise SystemExit(f"Missing {paths['manifest']}: run scripts/prepare_data.py first")
    df, classes = load_manifest(paths["manifest"]), load_json(paths["classes"])
    modes = list(MODES) if args.mode == "all" else [args.mode]
    backend = make_backend(cfg)
    for mode in modes:
        res = run_extraction(cfg, mode, backend, df, classes, n=args.n)
        print(f"{mode}: n_new={res['n_new']} n_total={res['n_total']} n_valid={res['n_valid']} -> {res['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `scripts/vlm_evaluate.py`**

```python
#!/usr/bin/env python
"""Summarise VLM extractions and create / score the manual grading sheet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args, work_paths  # noqa: E402
from foodmm.data.prepare import load_manifest  # noqa: E402
from foodmm.utils import load_json  # noqa: E402
from foodmm.vlm.evaluate import evaluate_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    paths = work_paths(cfg)
    summary = evaluate_all(cfg, load_manifest(paths["manifest"]), load_json(paths["classes"]))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_vlm_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/vlm_extract.py scripts/vlm_evaluate.py tests/test_vlm_pipeline.py
git commit -m "feat(vlm): extraction and evaluation scripts"
```

---

### Task 6: Notebook 03 and final verification

**Files:**
- Create: `tools/build_notebook_vlm.py`, `notebooks/03_vlm.ipynb` (generated)
- Modify: `tests/test_notebooks.py` (replace the `NOTEBOOKS` line)

- [ ] **Step 1: Replace the `NOTEBOOKS` line in `tests/test_notebooks.py`**

Change `NOTEBOOKS = [("build_notebook_m2.py", "02_milestone2.ipynb")]` to:
```python
NOTEBOOKS = [("build_notebook_m2.py", "02_milestone2.ipynb"), ("build_notebook_vlm.py", "03_vlm.ipynb")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebooks.py -v`
Expected: the two `03_vlm` cases FAIL (builder missing).

- [ ] **Step 3: Implement `tools/build_notebook_vlm.py`**

```python
#!/usr/bin/env python
"""Generate notebooks/03_vlm.ipynb (Qwen3-VL structured extraction and evaluation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Sub-project 3: Trích xuất thông tin món ăn bằng Qwen3-VL

Model chạy trực tiếp trong Colab qua `transformers`. Không cần server hay API key. Mỗi ảnh cho ra một JSON gồm `dish_name`, `cuisine`, `main_ingredients`, `cooking_method`, `confidence`.

**GPU:** L4 hoặc A100 chạy bf16. T4 tự chuyển sang 4-bit (`bitsandbytes`). Bảng so sánh với classifier dùng các run `image`, `text_strict`, `xattn_strict` của Mốc 2 (`vlm.compare_runs`); run nào chưa có sẽ bị bỏ qua.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Chạy thử 5 ảnh
Nạp model (lần đầu tải khoảng 9 GB về `/root/.cache`) và kiểm tra output. Kết quả ghi vào `foodmm/vlm/smoke`, tách khỏi lần chạy thật.
''')
nb.code(r'''
import torch

print(torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))
SMOKE = ["--set", *BASE, "vlm.tag=smoke", "vlm.n_samples=5"]
run("vlm_extract.py", "--mode", "image", *SMOKE)
print((WORK_DIR / "vlm" / "smoke" / "extract_image.jsonl").read_text()[:1500])
''')

nb.md(r'''
## 2. Chạy 200 ảnh với 3 chế độ prompt
`image` (chỉ ảnh), `text` (chỉ text đã che `strict`, không gửi ảnh), `image_text` (ảnh + text đã che): bộ so sánh image-only / text-only / multimodal của đề bài. Mất session thì chạy lại cell, các ảnh đã xong được bỏ qua. Có thể thêm `vlm.model_name=Qwen/Qwen3-VL-8B-Instruct vlm.tag=qwen3vl8b` để thử bản 8B.
''')
nb.code(r'''
for mode in ("image", "text", "image_text"):
    run("vlm_extract.py", "--mode", mode, "--set", *BASE)
''')

nb.md(r'''
## 3. Đánh giá
`dish_acc_lo`/`dish_acc_hi` là khoảng tin cậy 95% (bootstrap). Bảng thứ hai là accuracy của các classifier trên đúng 200 ảnh này, để so sánh trực tiếp. `ingredient_grounding` chỉ để tham khảo: các chế độ có `uses_text=True` được đọc chính text dùng để đo nên điểm bị thổi phồng.
''')
nb.code(r'''
from IPython.display import Markdown, display

run("vlm_evaluate.py", "--set", *BASE)
VLM_DIR = WORK_DIR / "vlm" / "qwen3vl4b"
display(Markdown((VLM_DIR / "eval_summary.md").read_text()))
display(Markdown("### Classifier trên cùng mẫu\n" + (VLM_DIR / "classifier_on_sample.md").read_text()))
''')

nb.md(r'''
## 4. Xem vài ví dụ
''')
nb.code(r'''
import json

import matplotlib.pyplot as plt
from PIL import Image as PILImage

from foodmm.data.prepare import load_manifest
from foodmm.vlm.evaluate import map_dish_to_class
from foodmm.vlm.extract import read_records
from foodmm.utils import load_json

manifest = load_manifest(WORK_DIR / "data" / "manifest.csv").set_index("id")
classes = load_json(WORK_DIR / "data" / "classes.json")
recs = {m: {r["id"]: r for r in read_records(VLM_DIR / f"extract_{m}.jsonl")}
        for m in ("image", "text", "image_text")}
for sid in load_json(VLM_DIR / "sample_ids.json")[:6]:
    row = manifest.loc[sid]
    plt.figure(figsize=(3, 3))
    plt.imshow(PILImage.open(DATA_ROOT / row["image_path"]).convert("RGB"))
    plt.axis("off")
    plt.title(row["label"])
    plt.show()
    for mode, by_id in recs.items():
        r = by_id.get(sid)
        if r and r["valid"]:
            print(f"[{mode}] -> {map_dish_to_class(r['output']['dish_name'], classes)}")
            print(json.dumps(r["output"], ensure_ascii=False))
        elif r:
            print(f"[{mode}] invalid: {r['error']}")
''')

nb.md(r'''
## 5. Chấm tay
1. Mở `MyDrive/foodmm/vlm/qwen3vl4b/manual_grading.csv` (Google Sheets hoặc Excel). Các dòng đã được xáo trộn và **không ghi chế độ prompt**, để chấm mù. Đừng mở `manual_grading_key.csv` trước khi chấm xong.
2. Điền `grade_dish` (0/1), `grade_ingredients` (0: sai, 1: một phần, 2: tốt), `grade_method` (0/1). Có thể ghi chú ở `notes`.
3. Lưu lại dưới dạng CSV cùng tên, rồi chạy cell dưới để tính điểm trung bình theo chế độ.
''')
nb.code(r'''
run("vlm_evaluate.py", "--set", *BASE)
scores = VLM_DIR / "manual_scores.csv"
print(scores.read_text() if scores.exists() else "Chưa có dòng nào được chấm.")
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "03_vlm.ipynb"))
```

- [ ] **Step 4: Generate the notebook and run the tests**

Run: `.venv/bin/python tools/build_notebook_vlm.py`
Expected: `Wrote .../notebooks/03_vlm.ipynb (N cells)`.
Run: `.venv/bin/pytest tests/test_notebooks.py -v && .venv/bin/pytest -q`
Expected: 4 passed, then the whole suite passes.

- [ ] **Step 5: Commit**

```bash
git add tools/build_notebook_vlm.py notebooks/03_vlm.ipynb tests/test_notebooks.py
git commit -m "feat(vlm): Colab notebook for VLM extraction"
```

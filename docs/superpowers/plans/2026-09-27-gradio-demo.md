# Sub-project 4 (Gradio demo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A password-protected Gradio app on Colab: upload a food photo and optional text, see top-5 predictions of the image, text and fusion heads side by side, optionally extract JSON with the VLM, and browse precomputed examples.

**Architecture:** Sub-package `src/foodmm/demo/` (predictor, vlm_service, examples, app). `DemoPredictor` wraps one frozen `ClipEncoder` plus three Milestone 2 heads loaded from `work_dir/clip/runs`. `LazyVLM` wraps the sub-project 3 backend and loads it on first use. UI callbacks are plain functions so they are tested without a server.

**Tech Stack:** `gradio==6.28.0`, Milestone 2 CLIP heads, sub-project 3 VLM backend.

**Spec:** `docs/superpowers/specs/2026-09-27-gradio-demo-design.md`
**Depends on:** Milestones 1–2 and sub-project 3 plans implemented.

## Global Constraints

- Same CLI/test/commit conventions as before.
- Label dicts use display names (`class_to_phrase`, e.g. `apple pie`) → probability (float), sorted by probability, at most `demo.top_k` entries.
- `DemoPredictor.predict` returns keys `image`, `text`, `fusion` (label dict or `None`), `masked_text` (str), `top3_fusion` (list of `(class_name, prob)` with raw class names, for VLM hints).
- A public share link is never created without auth unless `--no_auth` is passed explicitly.
- Tests marked `network` download the tiny CLIP.

## File Map

| File | Responsibility |
|---|---|
| `configs/default.yaml` (append) | `demo` section |
| `requirements-colab.txt`, `requirements-dev.txt` (replace) | add `gradio==6.28.0` |
| `src/foodmm/demo/predictor.py` | `load_head`, `DemoPredictor` |
| `src/foodmm/demo/vlm_service.py` | `LazyVLM` |
| `src/foodmm/demo/examples.py` | pick / build / load precomputed examples |
| `src/foodmm/demo/app.py` | `make_handlers`, `build_app` |
| `scripts/build_demo_examples.py`, `scripts/demo.py` | CLI |
| `tools/build_notebook_demo.py`, `notebooks/04_demo.ipynb` | notebook |
| `tests/test_demo_*.py`, `tests/test_notebooks.py` | tests |

---

### Task 1: Config, requirements, `LazyVLM` and the app

**Files:**
- Modify: `configs/default.yaml` (append), `requirements-colab.txt`, `requirements-dev.txt` (replace)
- Create: `src/foodmm/demo/__init__.py`, `src/foodmm/demo/vlm_service.py`, `src/foodmm/demo/app.py`
- Test: `tests/test_demo_app.py`

**Interfaces:**
- Produces: `LazyVLM(factory, max_image_side=768)` with `.extract(image, hints=None) -> dict`, `.factory_calls`; `make_handlers(predictor, vlm, examples) -> (on_predict, on_example)`; `on_predict(image, text, use_vlm) -> (img_labels, txt_labels, fusion_labels, masked_text, vlm_json)`; `on_example(name) -> (PIL | None, text, img_labels, txt_labels, fusion_labels, masked_text, vlm_json)`; `build_app(predictor, vlm=None, examples=None) -> gr.Blocks`. `examples` items: `{name, id, label, image_path (absolute), text, pred: {image, text, fusion, masked_text}, vlm}`.

- [ ] **Step 1: Append to `configs/default.yaml`**

```yaml

demo:
  text_mask: strict        # user text is cleaned and masked like the training data of the heads
  image_run: image
  text_run: null           # null -> text_<demo.text_mask>
  fusion_run: xattn_strict # a Milestone 2 head run (trained with modality dropout)
  top_k: 5
  n_examples: 8
  vlm: true                # show the VLM checkbox
  server_port: 7860
```

- [ ] **Step 2: Replace `requirements-colab.txt`**

```txt
# torch, torchvision, scikit-learn, pandas, matplotlib, pyyaml, pillow, accelerate and kaggle are preinstalled on Colab
timm==1.0.30
transformers==5.17.0
bitsandbytes==0.50.2
gradio==6.28.0
```

- [ ] **Step 3: Replace `requirements-dev.txt`**

```txt
# Local CPU environment for tests. torch/torchvision: latest CPU wheels.
torch
torchvision
timm==1.0.30
transformers==5.17.0
scikit-learn
pandas
pyyaml
pillow
matplotlib
pytest
gradio==6.28.0
```

- [ ] **Step 4: Write the failing test `tests/test_demo_app.py`**

```python
import gradio as gr
import pytest
from PIL import Image

from foodmm.demo.app import build_app, make_handlers
from foodmm.demo.vlm_service import LazyVLM
from foodmm.vlm.backend import FakeBackend

IMG = Image.new("RGB", (16, 16), (200, 120, 40))
TOP = {"apple pie": 0.7, "french fries": 0.2}


class FakePredictor:
    def predict(self, image, text):
        if image is None and not (text or "").strip():
            raise ValueError("Cần ảnh hoặc text")
        return {"image": TOP if image is not None else None, "text": TOP if text else None, "fusion": TOP,
                "masked_text": "best [MASK]" if text else "", "top3_fusion": [("apple_pie", 0.7)]}


EXAMPLES = [{"name": "1. apple pie", "id": "x", "label": "apple_pie", "image_path": None, "text": "best apple pie",
             "pred": {"image": TOP, "text": TOP, "fusion": TOP, "masked_text": "best [MASK]"}, "vlm": {"dish_name": "pie"}}]


def test_lazy_vlm_loads_once():
    fb = FakeBackend()
    calls = []
    vlm = LazyVLM(lambda: calls.append(1) or fb)
    assert vlm.extract(None) == {"error": "VLM cần một ảnh"}
    assert vlm.extract(IMG, [("apple_pie", 0.7)])["dish_name"] == "apple pie"
    vlm.extract(IMG)
    assert len(calls) == 1 and vlm.factory_calls == 1
    assert "classifier suggests" in fb.calls[0][0] and "classifier" not in fb.calls[1][0]


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
    assert vlm["dish_name"] == "apple pie"
    img, txt, fus, masked, vlm = on_predict(None, "  ", True)
    assert img is None and fus is None and "Cần ảnh" in vlm["error"]


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
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/pip install -q -r requirements-dev.txt && .venv/bin/pytest tests/test_demo_app.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'foodmm.demo'`.

- [ ] **Step 6: Implement `src/foodmm/demo/__init__.py`**

```python
"""Gradio demo: image / text / fusion predictions and optional VLM extraction."""
```

- [ ] **Step 7: Implement `src/foodmm/demo/vlm_service.py`**

```python
"""Load the VLM backend on first use and turn every failure into an error dict."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ..vlm.backend import resize_for_vlm
from ..vlm.extract import extract_one
from ..vlm.prompts import build_prompt


class LazyVLM:
    def __init__(self, factory: Callable[[], object], max_image_side: int = 768):
        self._factory = factory
        self._backend = None
        self._error: str | None = None
        self.max_image_side = int(max_image_side)
        self.factory_calls = 0

    def extract(self, image, hints: Sequence[tuple[str, float]] | None = None) -> dict:
        if image is None:
            return {"error": "VLM cần một ảnh"}
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
                self._error = f"Không nạp được VLM: {e}. Hãy xem tab 'Ví dụ có sẵn'."
                return {"error": self._error}
        prompt = build_prompt("image_hint", hints=hints) if hints else build_prompt("image")
        rec = extract_one(self._backend, resize_for_vlm(image, self.max_image_side), prompt)
        return rec["output"] if rec["valid"] else {"error": rec["error"], "raw": rec["raw"]}
```

- [ ] **Step 8: Implement `src/foodmm/demo/app.py`**

```python
"""Gradio UI. The callbacks are plain functions (tested without a server)."""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

NO_EXAMPLES = {"info": "Chưa có ví dụ: chạy scripts/build_demo_examples.py để tạo examples.json"}


def make_handlers(predictor, vlm, examples: Sequence[dict] | None):
    by_name = {e["name"]: e for e in (examples or [])}

    def on_predict(image, text, use_vlm):
        try:
            res = predictor.predict(image, text)
        except ValueError as e:
            return None, None, None, "", {"error": str(e)}
        vlm_out = vlm.extract(image, res["top3_fusion"]) if (use_vlm and vlm is not None) else None
        return res["image"], res["text"], res["fusion"], res["masked_text"], vlm_out

    def on_example(name):
        ex = by_name.get(name)
        if ex is None:
            return None, "", None, None, None, "", NO_EXAMPLES
        image = Image.open(ex["image_path"]).convert("RGB") if ex.get("image_path") else None
        p = ex["pred"]
        return image, ex["text"], p["image"], p["text"], p["fusion"], p["masked_text"], ex.get("vlm")

    return on_predict, on_example


def build_app(predictor, vlm=None, examples: Sequence[dict] | None = None):
    import gradio as gr

    on_predict, on_example = make_handlers(predictor, vlm, examples)

    def ui_predict(image, text, use_vlm):
        out = on_predict(image, text, use_vlm)
        if out[2] is None and isinstance(out[4], dict) and "error" in out[4]:
            gr.Warning(out[4]["error"])
        return out

    with gr.Blocks(title="Nhận diện món ăn đa phương thức") as app:
        gr.Markdown("# Nhận diện món ăn từ ảnh và text\nCLIP ViT-B/16 đóng băng + các head của Mốc 2. "
                    "Text được làm sạch và che tên món trước khi đưa vào model.")
        with gr.Tab("Dự đoán"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Ảnh món ăn")
                    text = gr.Textbox(label="Text (tùy chọn): công thức, mô tả...", lines=4)
                    use_vlm = gr.Checkbox(label="Trích xuất thông tin bằng VLM (lần đầu mất vài phút)",
                                          value=False, visible=vlm is not None)
                    button = gr.Button("Dự đoán", variant="primary")
                with gr.Column():
                    with gr.Row():
                        l_img = gr.Label(num_top_classes=5, label="Chỉ ảnh")
                        l_txt = gr.Label(num_top_classes=5, label="Chỉ text")
                        l_fus = gr.Label(num_top_classes=5, label="Fusion")
                    masked = gr.Textbox(label="Text sau khi che tên món", interactive=False)
                    vlm_json = gr.JSON(label="Kết quả VLM")
            button.click(ui_predict, [image, text, use_vlm], [l_img, l_txt, l_fus, masked, vlm_json])
        with gr.Tab("Ví dụ có sẵn"):
            if examples:
                choice = gr.Dropdown(choices=[e["name"] for e in examples], label="Chọn ví dụ")
                with gr.Row():
                    ex_img = gr.Image(type="pil", label="Ảnh", interactive=False)
                    ex_text = gr.Textbox(label="Text", lines=6, interactive=False)
                with gr.Row():
                    e_img = gr.Label(num_top_classes=5, label="Chỉ ảnh")
                    e_txt = gr.Label(num_top_classes=5, label="Chỉ text")
                    e_fus = gr.Label(num_top_classes=5, label="Fusion")
                e_masked = gr.Textbox(label="Text sau khi che tên món", interactive=False)
                e_vlm = gr.JSON(label="Kết quả VLM (đã lưu)")
                choice.change(on_example, choice, [ex_img, ex_text, e_img, e_txt, e_fus, e_masked, e_vlm])
            else:
                gr.Markdown(NO_EXAMPLES["info"])
    return app
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_demo_app.py -v`
Expected: 8 passed.

- [ ] **Step 10: Commit**

```bash
git add configs/default.yaml requirements-colab.txt requirements-dev.txt src/foodmm/demo/__init__.py src/foodmm/demo/vlm_service.py src/foodmm/demo/app.py tests/test_demo_app.py
git commit -m "feat(demo): lazy VLM service and Gradio app"
```

---

### Task 2: Predictor, examples and scripts

**Files:**
- Create: `src/foodmm/demo/predictor.py`, `src/foodmm/demo/examples.py`, `scripts/build_demo_examples.py`, `scripts/demo.py`
- Test: `tests/test_demo_predictor.py`

**Interfaces:**
- Consumes: `ClipEncoder`, `build_head`, `drop_modalities`, `clip_paths` (M2); `clean_text`, `build_mask_pattern`, `mask_text`, `class_to_phrase` (M1); `LazyVLM`, `build_app` (Task 1); `make_backend` (sub-project 3).
- Produces: `load_head(run_dir, device) -> nn.Module`; `DemoPredictor(cfg, encoder=None, device=None)` with `.classes`, `.run_names`, `.predict(image, text) -> dict`; `demo_dir(cfg) -> Path`; `pick_examples(df, n, seed) -> list[str]`; `build_examples(cfg, predictor, vlm=None, n=None) -> list[dict]`; `load_examples(cfg) -> list[dict] | None`.
- CLI: `build_demo_examples.py [--n N] [--with_vlm] --set ...`; `demo.py [--share] [--no_auth] [--no_vlm] [--dry_run] --set ...` (reads `DEMO_USERNAME` / `DEMO_PASSWORD`; `--dry_run` checks auth, prints `dry run ok` and exits before loading models).

- [ ] **Step 1: Write the failing test `tests/test_demo_predictor.py`**

```python
import json
import os

import pytest
from PIL import Image

from foodmm.config import load_config
from helpers import REPO_ROOT, clip_smoke_overrides, make_fake_dataset, run_script


def test_demo_refuses_public_link_without_auth():
    env = {k: v for k, v in os.environ.items() if k not in ("DEMO_USERNAME", "DEMO_PASSWORD")}
    import subprocess
    import sys

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "demo.py"), "--share", "--dry_run"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert res.returncode != 0 and "DEMO_USERNAME" in res.stderr
    ok = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                        env={**env, "DEMO_USERNAME": "u", "DEMO_PASSWORD": "p"})
    assert ok.returncode == 0 and "dry run ok" in ok.stdout and "auth=on" in ok.stdout
    no_auth = subprocess.run(cmd + ["--no_auth"], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert no_auth.returncode == 0 and "auth=off" in no_auth.stdout


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    data_root = make_fake_dataset(tmp / "ds")
    sets = clip_smoke_overrides(data_root, tmp / "work") + ["head.epochs=1"]
    run_script("prepare_data.py", "--set", *sets)
    run_script("extract_clip.py", "--parts", "image,text_strict", "--set", *sets)
    for head in ("image", "text", "xattn"):
        run_script("train_head.py", "--head", head, "--set", *sets, "data.text_mask=strict")
    return sets


@pytest.mark.network
def test_predictor(trained):
    from foodmm.demo.predictor import DemoPredictor

    pred = DemoPredictor(load_config(overrides=trained))
    assert pred.run_names == {"image": "image", "text": "text_strict", "fusion": "xattn_strict"}
    img = Image.new("RGB", (40, 30), (200, 150, 60))
    both = pred.predict(img, "Best apple pie recipe from grandma")
    assert set(both) == {"image", "text", "fusion", "masked_text", "top3_fusion"}
    assert len(both["fusion"]) == 3 and abs(sum(both["fusion"].values()) - 1.0) < 1e-4
    assert "apple" not in both["masked_text"] and "[MASK]" in both["masked_text"]
    assert all(" " in k or k.isalpha() for k in both["image"])  # display names, no underscores
    assert both["top3_fusion"][0][0] in ("apple_pie", "caesar_salad", "french_fries")
    image_only = pred.predict(img, "   ")
    assert image_only["text"] is None and image_only["fusion"] is not None and image_only["masked_text"] == ""
    text_only = pred.predict(None, "crispy fries")
    assert text_only["image"] is None and text_only["fusion"] is not None
    with pytest.raises(ValueError):
        pred.predict(None, "")


@pytest.mark.network
def test_build_demo_examples_script(trained):
    run_script("build_demo_examples.py", "--n", "2", "--set", *trained)
    cfg = load_config(overrides=trained)
    from foodmm.demo.examples import demo_dir, load_examples

    items = load_examples(cfg)
    assert len(items) == 2 and items[0]["name"].startswith("1. ") and items[0]["vlm"] is None
    assert os.path.exists(items[0]["image_path"])
    raw = json.loads((demo_dir(cfg) / "examples.json").read_text())
    assert raw[0]["image"].startswith("examples/")  # stored relative, resolved on load


def test_demo_missing_runs(tmp_path):
    res = run_script("demo.py", "--no_auth", "--set", f"paths.work_dir={tmp_path}", check=False)
    assert res.returncode != 0 and "Missing Milestone 2 runs" in res.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_demo_predictor.py -v`
Expected: FAIL / ERROR (`scripts/demo.py` and `foodmm.demo.predictor` do not exist).

- [ ] **Step 3: Implement `src/foodmm/demo/predictor.py`**

```python
"""Image / text / fusion predictions from one frozen CLIP encoder and three Milestone 2 heads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..clip.features import clip_paths
from ..clip.heads import build_head, drop_modalities
from ..config import load_config, work_paths
from ..data.text_utils import build_mask_pattern, class_to_phrase, clean_text, mask_text
from ..utils import get_device, load_json


def load_head(run_dir: str | Path, device: torch.device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    hcfg = load_config(run_dir / "config.yaml")["head"]
    model = build_head(ckpt["head"], ckpt["dims"], int(ckpt["num_classes"]), hcfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


class DemoPredictor:
    def __init__(self, cfg: dict, encoder=None, device: torch.device | None = None):
        d = cfg["demo"]
        self.cfg = cfg
        self.device = device or get_device()
        self.top_k = int(d["top_k"])
        self.max_chars = int(cfg["data"]["max_chars"])
        self.run_names = {"image": d["image_run"], "text": d["text_run"] or f"text_{d['text_mask']}",
                          "fusion": d["fusion_run"]}
        runs = clip_paths(cfg)["runs"]
        missing = [r for r in self.run_names.values() if not (runs / r / "best.pt").exists()]
        if missing:
            raise SystemExit(f"Missing Milestone 2 runs {missing} in {runs}: run notebooks/02_milestone2.ipynb first")
        self.classes = load_json(work_paths(cfg)["classes"])
        self.pattern = build_mask_pattern(self.classes, d["text_mask"])
        self.heads = {k: load_head(runs / r, self.device) for k, r in self.run_names.items()}
        if encoder is None:
            from ..clip.encoder import ClipEncoder

            cc = cfg["clip"]
            encoder = ClipEncoder(cc["model_name"], device=self.device, fp16=bool(cc["fp16"]),
                                  n_tokens=int(cc["n_tokens"]))
        self.encoder = encoder

    def _labels(self, probs: np.ndarray) -> dict[str, float]:
        top = np.argsort(-probs)[:self.top_k]
        return {class_to_phrase(self.classes[int(i)]): float(probs[i]) for i in top}

    @torch.no_grad()
    def _probs(self, key: str, batch: dict) -> np.ndarray:
        return torch.softmax(self.heads[key](batch)["logits"].float(), dim=1)[0].cpu().numpy()

    def predict(self, image, text: str | None) -> dict:
        text = (text or "").strip()
        if image is None and not text:
            raise ValueError("Cần ảnh hoặc text để dự đoán")
        enc, n = self.encoder, self.encoder.n_tokens
        masked = mask_text(clean_text(text, self.max_chars), self.pattern) if text else ""
        if image is not None:
            img, img_tok = enc.encode_images([image.convert("RGB")])
        else:
            img, img_tok = np.zeros((1, enc.embed_dim)), np.zeros((1, n, enc.image_dim))
        if text:
            txt, txt_tok, txt_mask = enc.encode_texts([masked])
        else:
            txt, txt_tok, txt_mask = np.zeros((1, enc.embed_dim)), np.zeros((1, n, enc.text_dim)), np.zeros((1, n))
        batch = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
                 for k, v in {"img": img, "img_tok": img_tok, "txt": txt, "txt_tok": txt_tok, "txt_mask": txt_mask}.items()}
        fusion_batch = drop_modalities(batch, torch.tensor([image is None]), torch.tensor([not text]))
        p_fusion = self._probs("fusion", fusion_batch)
        top3 = np.argsort(-p_fusion)[:3]
        return {
            "image": self._labels(self._probs("image", batch)) if image is not None else None,
            "text": self._labels(self._probs("text", batch)) if text else None,
            "fusion": self._labels(p_fusion),
            "masked_text": masked,
            "top3_fusion": [(self.classes[int(i)], float(p_fusion[i])) for i in top3],
        }
```

- [ ] **Step 4: Implement `src/foodmm/demo/examples.py`**

```python
"""Precomputed demo examples (work without a GPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import work_paths
from ..data.prepare import load_manifest
from ..data.text_utils import class_to_phrase
from ..utils import ensure_dir, load_json, save_json


def demo_dir(cfg: dict) -> Path:
    return work_paths(cfg)["work_dir"] / "demo"


def pick_examples(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """One random test image from each of n random classes."""
    rng = np.random.default_rng(seed)
    test = df[df["split"] == "test"]
    labels = sorted(test["label"].unique())
    chosen = rng.choice(labels, size=min(n, len(labels)), replace=False)
    return [str(rng.choice(test.loc[test["label"] == c, "id"].to_numpy())) for c in chosen]


def build_examples(cfg: dict, predictor, vlm=None, n: int | None = None) -> list[dict]:
    df = load_manifest(work_paths(cfg)["manifest"]).set_index("id", drop=False)
    ids = pick_examples(df, int(n or cfg["demo"]["n_examples"]), int(cfg["seed"]))
    out = demo_dir(cfg)
    ensure_dir(out / "examples")
    items = []
    for k, sid in enumerate(ids, 1):
        row = df.loc[sid]
        with Image.open(Path(cfg["paths"]["data_root"]) / row["image_path"]) as im:
            img = im.convert("RGB")
        img.thumbnail((512, 512))
        rel = f"examples/{k:02d}.jpg"
        img.save(out / rel, quality=90)
        text = str(row["text"])[:500]
        pred = predictor.predict(img, text)
        items.append({"name": f"{k}. {class_to_phrase(row['label'])}", "id": sid, "label": row["label"],
                      "image": rel, "text": text,
                      "pred": {key: pred[key] for key in ("image", "text", "fusion", "masked_text")},
                      "vlm": vlm.extract(img, pred["top3_fusion"]) if vlm is not None else None})
        print(f"example {k}/{len(ids)}: {row['label']}", flush=True)
    save_json(items, out / "examples.json")
    return items


def load_examples(cfg: dict) -> list[dict] | None:
    path = demo_dir(cfg) / "examples.json"
    if not path.exists():
        return None
    items = load_json(path)
    for it in items:
        it["image_path"] = str(path.parent / it["image"])
    return items
```

- [ ] **Step 5: Implement `scripts/build_demo_examples.py`**

```python
#!/usr/bin/env python
"""Precompute demo examples (predictions and, optionally, VLM output) into work_dir/demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args  # noqa: E402
from foodmm.demo.examples import build_examples, demo_dir  # noqa: E402
from foodmm.demo.predictor import DemoPredictor  # noqa: E402
from foodmm.demo.vlm_service import LazyVLM  # noqa: E402
from foodmm.vlm.backend import make_backend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--with_vlm", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    vlm = LazyVLM(lambda: make_backend(cfg), cfg["vlm"]["max_image_side"]) if args.with_vlm else None
    items = build_examples(cfg, DemoPredictor(cfg), vlm=vlm, n=args.n)
    print(f"Wrote {len(items)} examples to {demo_dir(cfg) / 'examples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Implement `scripts/demo.py`**

```python
#!/usr/bin/env python
"""Launch the Gradio demo. A public --share link requires DEMO_USERNAME / DEMO_PASSWORD (or --no_auth)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args  # noqa: E402


def resolve_auth(share: bool, no_auth: bool) -> tuple[str, str] | None:
    user, password = os.environ.get("DEMO_USERNAME"), os.environ.get("DEMO_PASSWORD")
    if user and password and not no_auth:
        return user, password
    if share and not no_auth:
        raise SystemExit("Refusing to create a public share link without auth: set DEMO_USERNAME and "
                         "DEMO_PASSWORD (Colab Secrets), or pass --no_auth explicitly")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--no_auth", action="store_true", help="allow running without a password")
    parser.add_argument("--no_vlm", action="store_true", help="hide the VLM checkbox")
    parser.add_argument("--dry_run", action="store_true", help="check the auth settings and exit")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    auth = resolve_auth(args.share, args.no_auth)
    if args.dry_run:
        print(f"dry run ok: share={args.share} auth={'on' if auth else 'off'}")
        return 0

    from foodmm.demo.app import build_app
    from foodmm.demo.examples import load_examples
    from foodmm.demo.predictor import DemoPredictor
    from foodmm.demo.vlm_service import LazyVLM
    from foodmm.vlm.backend import make_backend

    predictor = DemoPredictor(cfg)
    use_vlm = bool(cfg["demo"]["vlm"]) and not args.no_vlm
    vlm = LazyVLM(lambda: make_backend(cfg), cfg["vlm"]["max_image_side"]) if use_vlm else None
    app = build_app(predictor, vlm, load_examples(cfg))
    app.queue(default_concurrency_limit=1).launch(share=args.share, auth=auth,
                                                  server_port=int(cfg["demo"]["server_port"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_demo_predictor.py tests/test_demo_app.py -v`
Expected: 12 passed.

- [ ] **Step 8: Commit**

```bash
git add src/foodmm/demo/predictor.py src/foodmm/demo/examples.py scripts/build_demo_examples.py scripts/demo.py tests/test_demo_predictor.py
git commit -m "feat(demo): CLIP-head predictor, precomputed examples and launch script with auth"
```

---

### Task 3: Notebook 04 and final verification

**Files:**
- Create: `tools/build_notebook_demo.py`, `notebooks/04_demo.ipynb` (generated)
- Modify: `tests/test_notebooks.py` (replace the `NOTEBOOKS` line)

- [ ] **Step 1: Replace the `NOTEBOOKS` line in `tests/test_notebooks.py`**

```python
NOTEBOOKS = [("build_notebook_m2.py", "02_milestone2.ipynb"), ("build_notebook_vlm.py", "03_vlm.ipynb"),
             ("build_notebook_demo.py", "04_demo.ipynb")]
```

- [ ] **Step 2: Implement `tools/build_notebook_demo.py`**

```python
#!/usr/bin/env python
"""Generate notebooks/04_demo.ipynb (Gradio demo on Colab)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_utils import NOTEBOOK_DIR, NotebookBuilder, add_data, add_setup  # noqa: E402

nb = NotebookBuilder()
nb.md(r'''
# Sub-project 4: Demo Gradio

Tải ảnh món ăn lên, có thể nhập thêm text, rồi xem top-5 của 3 nhánh: chỉ ảnh, chỉ text, fusion (`xattn_strict`). Có thể bật trích xuất JSON bằng Qwen3-VL.

**Cần có trước:** các run `image`, `text_strict`, `xattn_strict` của Mốc 2 trên Drive.

**Bảo mật:** `share=True` tạo link công khai. Thêm Colab Secrets `DEMO_USERNAME` và `DEMO_PASSWORD` trước khi chạy. Nếu thiếu, script sẽ từ chối tạo link.
''')
add_setup(nb)
add_data(nb)

nb.md(r'''
## 1. Tạo ví dụ có sẵn
Chạy một lần. Thêm `--with_vlm` nếu muốn lưu cả kết quả VLM (cần GPU, lâu hơn). Tab "Ví dụ có sẵn" vẫn xem được khi không có GPU.
''')
nb.code(r'''
if not (WORK_DIR / "demo" / "examples.json").exists():
    run("build_demo_examples.py", "--n", "8", "--set", *BASE)
''')

nb.md(r'''
## 2. Chạy demo
Mở link `*.gradio.live` in ra bên dưới và đăng nhập bằng tài khoản trong Colab Secrets. Dừng cell để tắt demo.
''')
nb.code(r'''
from google.colab import userdata

os.environ["DEMO_USERNAME"] = userdata.get("DEMO_USERNAME")
os.environ["DEMO_PASSWORD"] = userdata.get("DEMO_PASSWORD")
run("demo.py", "--share", "--set", *BASE)
''')

if __name__ == "__main__":
    raise SystemExit(nb.main(NOTEBOOK_DIR / "04_demo.ipynb"))
```

- [ ] **Step 3: Generate the notebook and run the whole suite**

Run: `.venv/bin/python tools/build_notebook_demo.py && .venv/bin/pytest -q`
Expected: every test passes.

- [ ] **Step 4: Commit**

```bash
git add tools/build_notebook_demo.py notebooks/04_demo.ipynb tests/test_notebooks.py
git commit -m "feat(demo): Colab notebook for the Gradio demo"
```

- [ ] **Step 5: Ask the user before pushing**

Pushing publishes all four sub-projects to the public repo (`main`, cloned by the notebooks). Only after the user confirms: `git push -u origin main`.

# Multimodal Food Understanding from Images and Text

Deep Learning final project, Group 12, USTH.

We recognize 101 dishes from food photos and their web text (UPMC Food-101), compare image-only, text-only and
multimodal models, and extract structured information (dish, cuisine, ingredients, cooking method) with a
vision–language model. Because the web text often contains the dish name, every main result uses text in which
all class names are masked (`strict`).

- **Report:** [`docs/report/main.tex`](docs/report/main.tex) (LaTeX source; figures in `docs/report/figures/`)
- **Data:** [`DATA.md`](DATA.md): source, version, split, preprocessing
- **Result tables:** [`docs/results/`](docs/results/)
- **Demo:** Gradio app, see [Demo](#demo); recorded demo video: <https://drive.google.com/file/d/1S3OLhJCYuJWdLxxNX41pHEKuCQX2-KQO/view>

## Main results

Test set: 22,712 images, text masked with `strict`. 95% bootstrap confidence intervals in brackets.

| Model | Modality | Accuracy (%) | Top-5 (%) |
|---|---|---|---|
| TF-IDF + logistic regression | text | 43.69 | 59.96 |
| DistilBERT (fine-tuned) | text | 41.42 | 60.18 |
| ResNet-50 (fine-tuned) | image | 65.56 | 84.44 |
| Early fusion, ResNet-50 + DistilBERT | image + text | 77.65 | 91.35 |
| CLIP ViT-B/16 zero-shot | image | 71.37 [70.76, 71.97] | 87.33 |
| CLIP text head | text | 38.22 [37.64, 38.83] | 57.13 |
| CLIP image head | image | 79.06 [78.56, 79.57] | 92.51 |
| CLIP late fusion | image + text | 83.73 [83.26, 84.18] | 95.18 |
| CLIP gated fusion | image + text | 84.94 [84.51, 85.42] | 95.17 |
| CLIP concat fusion | image + text | 85.08 [84.61, 85.56] | 95.29 |
| **CLIP cross-attention fusion** | image + text | **85.45 [85.01, 85.94]** | **95.49** |

- Fusion beats the image head by **+6.39 points** (paired CI 5.96–6.84, McNemar p < 1e-160).
- With raw text (dish names kept) text-only models reach about 86%, so masking is necessary for a fair
  comparison.
- Fusion relies mostly on the image. Raising modality dropout from 0.1 to 0.3 lifts cross-attention without the
  image from 34.96% to 38.58% (text-only head: 38.22%) at no cost when both inputs are present.
- Qwen3-VL-4B (200 test images): dish accuracy by label 61.0% with the image, 5.5% with masked text only, 60.0% with
  both. Manual grades with the image: dish 0.80–0.85, ingredients 1.65 / 2, cooking method 0.90.

Full tables: [`docs/results/`](docs/results/README.md).

## Repository layout

```
configs/default.yaml      every setting (paths, data, models, training, CLIP, VLM, demo, statistics)
src/foodmm/
  data/                   manifest building, text cleaning and masking, PyTorch datasets
  models/                 Milestone 1 models (ResNet-50, DistilBERT, early fusion)
  engine.py               training loop: AMP, checkpoints, resume, early stopping
  late_fusion.py          late fusion with the weight tuned on validation
  clip/                   Milestone 2: CLIP feature extraction, fusion heads, training, reports
  vlm/                    Qwen3-VL backend, prompts, JSON parsing, extraction, evaluation
  demo/                   Gradio app
  stats.py                bootstrap CIs and McNemar tests
  report_stats.py         extra statistics quoted in the report
scripts/                  command-line entry points (one per step, listed below)
notebooks/                Colab notebooks 01–04 that run the scripts in order
tools/                    generators of the notebooks (python tools/build_notebook*.py)
tests/                    pytest suite (CPU only)
docs/report/              report source and figures
docs/results/             copies of the result tables
DATA.md                   dataset documentation
```

## Setup

The experiments were run on **Google Colab** (one NVIDIA L4 GPU); the notebooks install everything they need.

**Colab (recommended).** Open a notebook from `notebooks/` in Colab with a GPU runtime and run the cells in order.
The first cells mount Google Drive, clone this repository and install `requirements-colab.txt`. Outputs go to
`MyDrive/foodmm/`. The data cell needs a Kaggle token in Colab Secrets (`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` +
`KAGGLE_KEY`).

**Local (Python ≥ 3.10).**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full environment (GPU machine): all experiments, VLM, demo, tests
# or: pip install -r requirements-dev.txt   # lighter CPU environment for the tests only
```

| File | Use |
|---|---|
| `requirements.txt` | everything needed to reproduce the experiments on a CUDA GPU (adds accelerate, bitsandbytes, kaggle) |
| `requirements-dev.txt` | CPU environment for the test suite |
| `requirements-colab.txt` | only the packages Colab does not preinstall (timm, transformers, bitsandbytes, gradio); used by the notebooks |

Training the models needs a CUDA GPU; the tests run on CPU.

## Data

Download UPMC Food-101 from Kaggle and build the manifest (split, cleaned and masked texts). Details in
[`DATA.md`](DATA.md).

```bash
kaggle datasets download -d gianmarco96/upmcfood101 -p <work_dir>/data
unzip <work_dir>/data/upmcfood101.zip -d <data_root>
python scripts/prepare_data.py --set paths.data_root=<data_root> paths.work_dir=<work_dir>
```

## Training and evaluation

Every script reads `configs/default.yaml`; override any key with `--set key=value` (it must be the last option).
Below, `$P` stands for `paths.data_root=<data_root> paths.work_dir=<work_dir>`. Scripts skip finished runs and
resume interrupted ones; add `--force` to redo a run.

**Milestone 1: fine-tuned baselines** (notebook `01_milestone1.ipynb`)

```bash
for m in none strict; do python scripts/train_tfidf.py --set $P data.text_mask=$m; done
python scripts/train.py --modality image --set $P
for m in none strict; do python scripts/train.py --modality text --set $P data.text_mask=$m; done
for m in none strict; do python scripts/late_fusion.py --text_run text_$m --set $P; done
for m in none strict; do python scripts/train.py --modality multimodal --set $P data.text_mask=$m; done
python scripts/summarize.py --set $P                 # -> <work_dir>/results/summary.{csv,md}
```

**Milestone 2: frozen CLIP features and fusion heads** (notebook `02_milestone2.ipynb`)

```bash
python scripts/extract_clip.py --parts image,text_none,text_strict --set $P   # cached features, resumable
python scripts/run_clip_suite.py --set $P            # zero-shot, image, text, concat, gated, xattn, late
python scripts/summarize_clip.py --set $P            # -> <work_dir>/clip/results/{main,significance,missing}.md
```

Single head: `python scripts/train_head.py --head xattn --set $P data.text_mask=strict`. Modality-dropout ablation:
add `head.modality_dropout=0` or `head.modality_dropout=0.3`.

**Information extraction with Qwen3-VL** (notebook `03_vlm.ipynb`, needs a GPU)

```bash
python scripts/vlm_extract.py --mode all --set $P    # image, text, image_text on 200 stratified test images
python scripts/vlm_evaluate.py --set $P              # -> <work_dir>/vlm/qwen3vl4b/eval_summary.md, grading sheet
```

`vlm_evaluate.py` writes a blind grading sheet (`manual_grading.csv`, mode hidden in `manual_grading_key.csv`);
fill the `grade_*` columns and run it again to get `manual_scores.csv`.

**Statistics used in the report**

```bash
python scripts/report_stats.py --set $P              # -> <work_dir>/results/report_stats/
```

Bonferroni-corrected p-values, bootstrap CIs of the Milestone 1 runs, a residual-leakage audit of the masked text,
the paired image vs. image+text VLM test and the manual-grade CIs.

## Demo

A Gradio app shows the top-5 dishes of the image head, the text head and the cross-attention fusion side by side,
and can run the VLM to display the extracted JSON (notebook `04_demo.ipynb`):

```bash
python scripts/build_demo_examples.py --n 8 --set $P
export DEMO_USERNAME=... DEMO_PASSWORD=...           # required for a public link
python scripts/demo.py --share --set $P
```

Without a GPU, `--no_vlm` disables the VLM option. The public `*.gradio.live` link expires when the session ends.

**Permanent demo on Hugging Face Spaces** (CPU, image / text / fusion only; the VLM needs a GPU). Hugging Face now
requires a PRO account to host Gradio Spaces, so this step is optional and was not used for the submission:

```bash
export HF_TOKEN=<write token>
python scripts/deploy_space.py --space <hf-username>/food-recognition --set $P
```

The script bundles `space/app.py`, the `foodmm` package, the config and the three head checkpoints the demo loads
(no dataset images unless `--with_examples`), then creates or updates the Space.

## Tests

```bash
pytest -q -m "not network"     # CPU, no downloads
pytest -q                      # also runs tests that download tiny models from the Hugging Face Hub
```

## Notes

- All results use one training seed; intervals reflect test-set sampling only.
- `strict` masking does not catch accented or misspelled dish names; see `DATA.md` and the report.
- Outputs (runs, features, predictions) stay on Google Drive and are not committed; the tables quoted in the report
  are copied to `docs/results/`.

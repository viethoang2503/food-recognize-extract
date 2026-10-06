# Data

## Source

| | |
|---|---|
| Dataset | **UPMC Food-101**: food photos paired with the text of the web page they come from, 101 dish classes (the Food-101 classes) |
| Original release | X. Wang, D. Kumar, N. Thome, M. Cord, F. Precioso. *Recipe recognition with large multimodal food dataset.* IEEE ICME Workshops, 2015 |
| Official source | Paper introducing the dataset: <https://doi.org/10.1109/ICMEW.2015.7169757> (the authors' original download page is no longer online) |
| Copy used | Kaggle dataset `gianmarco96/upmcfood101`: <https://www.kaggle.com/datasets/gianmarco96/upmcfood101> |
| Version | **Kaggle Version 1** (last updated 2020-10-12, 8.3 GB), downloaded on 2026-09-30. The archive contains `images/{train,test}/<class>/*.jpg` and the texts as CSV files |
| Licence | See the Kaggle page and the original paper; the images and texts were collected from the web and remain the property of their owners. We do not redistribute them |

The download is automatic in the notebooks (Kaggle API, Colab Secret `KAGGLE_API_TOKEN` or `KAGGLE_USERNAME` +
`KAGGLE_KEY`):

```bash
kaggle datasets download -d gianmarco96/upmcfood101 -p <work_dir>/data
unzip <work_dir>/data/upmcfood101.zip -d <data_root>
```

## Split

We keep the **official train/test split** of the archive and hold out a stratified 10% of the official training
set as validation (`data.val_ratio: 0.1`, `seed: 42`, `sklearn.model_selection.train_test_split(stratify=label)`).

| Split | Samples | Origin |
|---|---|---|
| train | 61,169 | official `train/` minus validation |
| val | 6,797 | stratified 10% of official `train/` |
| test | 22,712 | official `test/`, unchanged |
| total | 90,678 | 90,704 images found, 26 dropped because their text is empty |

The counts are in [`docs/results/data_stats.json`](docs/results/data_stats.json). Validation is used for every
choice (early stopping, late-fusion weight, TF-IDF `C`); test is used only for reporting.

## Preprocessing

All done by `scripts/prepare_data.py` (`src/foodmm/data/prepare.py`, `src/foodmm/data/text_utils.py`):

1. **Pairing.** Images are found recursively; the label is the parent folder, the split the `train`/`test` folder.
   Texts are read from the CSV files (image column and text column detected automatically) and joined on
   `(filename, split)`. Samples without an image or with empty text are dropped; the script stops if more than 5%
   would be dropped.
2. **Text cleaning.** HTML entities unescaped, HTML tags and URLs removed, white space collapsed, lower-cased,
   truncated to 5,000 characters.
3. **Masking dish names** (three text columns):
   - `text` (`none`): cleaned text.
   - `text_exact` (`exact`): every class name of all 101 classes, with singular/plural variants, replaced by
     `[MASK]` (longest phrase first, word boundaries).
   - `text_strict` (`strict`, used for all conclusions): `exact` plus every single word of length >= 3 that occurs
     in any class name (stop words excluded), e.g. *chicken*, *rice*, *grilled*.

   Known gap: masking matches ASCII class names, so accented spellings (*phở*, *crème brûlée*), compounds
   (*cheeseburgers*) and misspellings are not masked. `scripts/report_stats.py` (section `leakage_audit`) measures
   how often a class word reappears once diacritics are removed.
4. **Images** are not modified on disk. Training transforms are in `src/foodmm/data/datasets.py` (Milestone 1) and
   the CLIP processor (Milestone 2); the VLM resizes the longer side to at most 768 px.

Outputs, written to `<work_dir>/data/`:

| File | Content |
|---|---|
| `manifest.csv` | `id, image_path, label, label_idx, split, text, text_exact, text_strict` (one row per sample) |
| `classes.json` | the 101 class names in label-index order |
| `stats.json` | images found, samples dropped, split counts |

## Reproducing the data used in the experiments

```bash
pip install -r requirements.txt              # or requirements-colab.txt on Colab
python scripts/prepare_data.py --set paths.data_root=<data_root> paths.work_dir=<work_dir>
```

`prepare_data.py` is deterministic (fixed seed), so rerunning it on the same archive reproduces the same manifest and
split. Notebook `notebooks/01_milestone1.ipynb` runs the same steps on Colab and shows the leakage table for the
three masking modes.

## Derived data

We do not create new raw data. The derived files are the manifest above (the split and the masked texts), the cached
CLIP features (`<work_dir>/clip/features/`) and the VLM outputs (`<work_dir>/vlm/qwen3vl4b/`). All of them are
regenerated deterministically by the scripts from the Kaggle archive (`prepare_data.py` for the manifest, fixed
seed 42), so we do not distribute them separately; the split counts are in
[`docs/results/data_stats.json`](docs/results/data_stats.json).

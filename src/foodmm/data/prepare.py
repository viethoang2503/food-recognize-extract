"""Build the dataset manifest: image path, cleaned + masked text, label, split."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .text_utils import build_mask_pattern, clean_text, mask_text

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MANIFEST_COLUMNS = ["id", "image_path", "label", "label_idx", "split", "text", "text_exact", "text_strict"]


class DataFormatError(RuntimeError):
    """Raised when the dataset layout or its text files cannot be understood."""


def describe_tree(root: Path, max_depth: int = 3, max_entries: int = 40) -> str:
    root = Path(root)
    lines: list[str] = []
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        lines.append("  " * (depth - 1) + p.name + ("/" if p.is_dir() else ""))
        if len(lines) >= max_entries:
            lines.append("  ...")
            break
    return "\n".join(lines) or "(empty)"


def _split_from_dirs(parts: tuple[str, ...] | list[str]) -> str:
    lowered = {p.lower() for p in parts}
    for name in ("train", "test"):
        if name in lowered:
            return name
    return "unknown"


def _split_from_filename(name: str) -> str:
    lowered = name.lower()
    has_train, has_test = "train" in lowered, "test" in lowered
    if has_train and not has_test:
        return "train"
    if has_test and not has_train:
        return "test"
    return "unknown"


def _is_image_name(value: object) -> bool:
    return Path(str(value).strip()).suffix.lower() in IMAGE_EXTS


def scan_images(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = []
    for p in sorted(data_root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            rel = p.relative_to(data_root)
            rows.append({"image_path": rel.as_posix(), "filename": p.name,
                         "label": p.parent.name, "split": _split_from_dirs(rel.parts[:-1])})
    return pd.DataFrame(rows, columns=["image_path", "filename", "label", "split"])


def read_text_csv(path: Path) -> pd.DataFrame:
    """Detect the image-filename column and the (longest) text column of a CSV."""
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, on_bad_lines="skip")
    if df.empty:
        raise DataFormatError(f"{path} is empty")
    if not any(_is_image_name(v) for v in df.iloc[0]):
        df = df.iloc[1:].reset_index(drop=True)  # header row
    if df.empty or df.shape[1] < 2:
        raise DataFormatError(f"{path}: expected at least 2 columns (image filename, text)")
    img_frac = {c: df[c].map(_is_image_name).mean() for c in df.columns}
    img_col = max(img_frac, key=img_frac.get)
    if img_frac[img_col] < 0.9:
        raise DataFormatError(f"{path}: no column of image filenames. First rows:\n{df.head(3).to_string()}")
    rest = [c for c in df.columns if c != img_col]
    text_col = max(rest, key=lambda c: df[c].str.len().mean())
    return pd.DataFrame({"filename": df[img_col].map(lambda v: Path(v.strip()).name), "text": df[text_col]})


def _texts_from_csvs(data_root: Path) -> pd.DataFrame | None:
    frames = []
    for path in sorted(data_root.rglob("*.csv")):
        try:
            texts = read_text_csv(path)
        except DataFormatError as err:
            logger.warning("Skipping %s: %s", path, err)
            continue
        split = _split_from_filename(path.stem)
        if split == "unknown":
            split = _split_from_dirs(path.relative_to(data_root).parts[:-1])
        texts["split"] = split
        frames.append(texts)
    return pd.concat(frames, ignore_index=True) if frames else None


def _texts_from_txts(data_root: Path) -> pd.DataFrame | None:
    rows = [{"stem": p.stem, "split": _split_from_dirs(p.relative_to(data_root).parts[:-1]),
             "text": p.read_text(encoding="utf-8", errors="ignore")}
            for p in sorted(data_root.rglob("*.txt"))]
    return pd.DataFrame(rows) if rows else None


def load_texts(data_root: Path, images: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return (texts[filename, split, text], source) where source is "csv" or "txt"."""
    data_root = Path(data_root)
    texts = _texts_from_csvs(data_root)
    if texts is not None:
        return texts[["filename", "split", "text"]], "csv"
    txts = _texts_from_txts(data_root)
    if txts is not None:
        stems = images.assign(stem=images["filename"].map(lambda f: Path(f).stem))
        merged = stems.merge(txts, on=["stem", "split"], how="inner")
        return merged[["filename", "split", "text"]], "txt"
    raise DataFormatError(f"No text files (.csv or .txt) found under {data_root}.\n{describe_tree(data_root)}")


def build_manifest(data_root: Path, *, val_ratio: float = 0.1, seed: int = 42,
                   subset_classes: int | None = None, max_per_class: int | None = None,
                   max_chars: int = 5000, max_drop_ratio: float = 0.05) -> tuple[pd.DataFrame, list[str], dict]:
    data_root = Path(data_root)
    images = scan_images(data_root)
    if images.empty:
        raise DataFormatError(f"No images found under {data_root}.\n{describe_tree(data_root)}")
    if (images["split"] == "unknown").any():
        bad = images.loc[images["split"] == "unknown", "image_path"].head(3).tolist()
        raise DataFormatError(f"Could not infer train/test split for images such as {bad}")

    texts, source = load_texts(data_root, images)
    keys = ["filename", "split"] if (texts["split"] != "unknown").all() else ["filename"]
    texts = texts.drop_duplicates(subset=keys, keep="first")
    df = images.merge(texts[keys + ["text"]], on=keys, how="left")
    df["text"] = df["text"].fillna("").map(lambda t: clean_text(t, max_chars))

    n_images = len(df)
    keep = df["text"].str.len() > 0
    n_dropped = int((~keep).sum())
    if n_dropped / n_images > max_drop_ratio:
        examples = df.loc[~keep, "image_path"].head(5).tolist()
        raise DataFormatError(f"{n_dropped}/{n_images} images have no matching text "
                              f"(max_drop_ratio={max_drop_ratio}). Examples: {examples}")
    df = df[keep].copy()

    classes = sorted(df["label"].unique().tolist())
    if subset_classes:
        classes = classes[: int(subset_classes)]
        df = df[df["label"].isin(classes)].copy()
    if max_per_class:
        df = df.sample(frac=1.0, random_state=seed).groupby(["split", "label"], sort=False).head(int(max_per_class)).copy()
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)}).astype(int)

    train_index = df.index[df["split"] == "train"]
    if len(train_index) == 0 or not (df["split"] == "test").any():
        raise DataFormatError("Need both train and test images")
    _, val_index = train_test_split(train_index, test_size=val_ratio, random_state=seed,
                                    stratify=df.loc[train_index, "label"])
    df.loc[val_index, "split"] = "val"

    for mode in ("exact", "strict"):
        pattern = build_mask_pattern(classes, mode)
        df[f"text_{mode}"] = df["text"].map(lambda t, p=pattern: mask_text(t, p))
    df["id"] = df["image_path"]
    df = df.sort_values(["split", "label", "image_path"]).reset_index(drop=True)[MANIFEST_COLUMNS]
    stats = {
        "text_source": source,
        "n_images_found": n_images,
        "n_dropped_no_text": n_dropped,
        "n_classes": len(classes),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
    }
    return df, classes, stats


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)  # keep empty strings as "", not NaN
    df["label_idx"] = df["label_idx"].astype(int)
    return df

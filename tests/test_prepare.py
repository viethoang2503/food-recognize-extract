import pandas as pd
import pytest

from foodmm.data.prepare import (
    MANIFEST_COLUMNS, DataFormatError, build_manifest, load_manifest, read_text_csv,
)
from helpers import CLASSES, make_fake_dataset


@pytest.mark.parametrize("fmt", ["csv", "txt"])
def test_build_manifest_detects_format(tmp_path, fmt):
    root = make_fake_dataset(tmp_path / "ds", fmt=fmt)
    df, classes, stats = build_manifest(root, val_ratio=0.3, seed=0)
    assert classes == CLASSES
    assert stats["text_source"] == fmt
    assert list(df.columns) == MANIFEST_COLUMNS
    counts = df["split"].value_counts().to_dict()
    assert counts["test"] == 12
    assert counts["val"] == 9
    assert counts["train"] == 21
    assert set(df.loc[df["split"] == "val", "label"]) == set(CLASSES)
    row = df[df["label"] == "apple_pie"].iloc[0]
    assert "<p>" not in row["text"] and "http" not in row["text"]
    assert "apple pie" in row["text"]
    assert "apple pie" not in row["text_exact"] and "[MASK]" in row["text_exact"]
    assert "apple" not in row["text_strict"]
    assert df["id"].is_unique


def test_csv_with_header_row(tmp_path):
    root = make_fake_dataset(tmp_path / "ds", fmt="csv", csv_header=True)
    df, _, _ = build_manifest(root, val_ratio=0.3, seed=0)
    assert len(df) == 42


def test_read_text_csv_picks_columns(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text('a_1.jpg,"long text, with comma here",a\nb_2.jpg,"another long text",b\n')
    t = read_text_csv(path)
    assert t["filename"].tolist() == ["a_1.jpg", "b_2.jpg"]
    assert t["text"].iloc[0] == "long text, with comma here"


def test_subset_and_max_per_class(tmp_path):
    root = make_fake_dataset(tmp_path / "ds")
    df, classes, _ = build_manifest(root, val_ratio=0.3, seed=0, subset_classes=2, max_per_class=5)
    assert classes == CLASSES[:2]
    assert set(df["label"]) == set(CLASSES[:2])
    counts = df["split"].value_counts().to_dict()
    assert counts["test"] == 8
    assert counts["train"] + counts["val"] == 10
    assert sorted(df["label_idx"].unique().tolist()) == [0, 1]


def test_missing_text_raises(tmp_path):
    root = make_fake_dataset(tmp_path / "ds")
    (root / "texts" / "test_titles.csv").unlink()
    with pytest.raises(DataFormatError, match="no matching text"):
        build_manifest(root, val_ratio=0.3, seed=0)


def test_no_images_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataFormatError, match="No images"):
        build_manifest(tmp_path / "empty")


def test_load_manifest_roundtrip(tmp_path):
    root = make_fake_dataset(tmp_path / "ds")
    df, _, _ = build_manifest(root, val_ratio=0.3, seed=0)
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    back = load_manifest(path)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)

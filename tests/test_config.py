import numpy as np
import pytest

from foodmm.config import DEFAULT_CONFIG, load_config, parse_value, work_paths
from foodmm.utils import load_json, save_json


def test_default_config_loads():
    cfg = load_config()
    assert DEFAULT_CONFIG.name == "default.yaml"
    assert cfg["image"]["backbone"] == "resnet50"
    assert cfg["text"]["model_name"] == "distilbert/distilbert-base-uncased"
    assert cfg["data"]["text_mask"] == "none"
    assert isinstance(cfg["train"]["image"]["lr_backbone"], float)


def test_overrides_parse_types():
    cfg = load_config(DEFAULT_CONFIG, [
        "data.subset_classes=5", "train.image.lr_head=1e-4", "data.text_mask=strict",
        "fusion.text_init=null", "image.pretrained=false", "tfidf.C_grid=[1, 2]", "new.key=abc",
    ])
    assert cfg["data"]["subset_classes"] == 5
    assert cfg["train"]["image"]["lr_head"] == pytest.approx(1e-4)
    assert cfg["data"]["text_mask"] == "strict"
    assert cfg["fusion"]["text_init"] is None
    assert cfg["image"]["pretrained"] is False
    assert cfg["tfidf"]["C_grid"] == [1, 2]
    assert cfg["new"]["key"] == "abc"


def test_parse_value_keeps_plain_strings():
    assert parse_value("resnet50") == "resnet50"
    assert parse_value("/content/data/x") == "/content/data/x"
    assert parse_value("3.0e-5") == pytest.approx(3e-5)


def test_bad_override_raises():
    with pytest.raises(ValueError, match="key=value"):
        load_config(DEFAULT_CONFIG, ["no_equals_sign"])


def test_work_paths(tmp_path):
    cfg = load_config(DEFAULT_CONFIG, [f"paths.work_dir={tmp_path}"])
    p = work_paths(cfg)
    assert p["manifest"] == tmp_path / "data" / "manifest.csv"
    assert p["classes"] == tmp_path / "data" / "classes.json"
    assert p["runs"] == tmp_path / "runs"
    assert p["results"] == tmp_path / "results"


def test_save_json_handles_numpy(tmp_path):
    path = tmp_path / "sub" / "x.json"
    save_json({"a": np.float32(0.5), "b": np.arange(3), "c": np.int64(2)}, path)
    assert load_json(path) == {"a": 0.5, "b": [0, 1, 2], "c": 2}

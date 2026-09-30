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

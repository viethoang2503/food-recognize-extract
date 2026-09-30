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

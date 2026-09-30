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

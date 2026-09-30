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

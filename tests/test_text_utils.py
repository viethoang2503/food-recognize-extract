import pytest

from foodmm.data.text_utils import (
    build_mask_pattern, clean_text, leakage_stats, mask_text, text_column, word_variants,
)

CLASSES = ["apple_pie", "caesar_salad", "french_fries", "spaghetti_carbonara", "fish_and_chips"]


def test_clean_text_strips_html_urls_and_whitespace():
    raw = "<p>Best&amp;Easy   Apple Pie</p>\nVisit http://x.com/a now"
    assert clean_text(raw) == "best&easy apple pie visit now"


def test_clean_text_truncates_and_handles_none():
    assert clean_text(None) == ""
    assert clean_text("abcdef", max_chars=3) == "abc"


def test_text_column():
    assert text_column("none") == "text"
    assert text_column("strict") == "text_strict"
    with pytest.raises(ValueError):
        text_column("bogus")


def test_word_variants():
    assert word_variants("pie") == {"pie", "pies"}
    assert word_variants("fries") == {"fries", "frie"}
    assert "sandwiches" in word_variants("sandwich")


def test_none_mode_has_no_pattern():
    assert build_mask_pattern(CLASSES, "none") is None
    assert mask_text("apple pie", None) == "apple pie"


def test_exact_masks_all_class_phrases_and_plurals():
    p = build_mask_pattern(CLASSES, "exact")
    assert mask_text("two apple pies and a caesar salad", p) == "two [MASK] and a [MASK]"


def test_exact_keeps_partial_names():
    p = build_mask_pattern(CLASSES, "exact")
    assert mask_text("classic carbonara with bacon", p) == "classic carbonara with bacon"


def test_strict_masks_single_words_but_not_stopwords():
    p = build_mask_pattern(CLASSES, "strict")
    assert mask_text("carbonara and chips with fish", p) == "[MASK] and [MASK] with [MASK]"


def test_strict_does_not_mask_inside_words():
    p = build_mask_pattern(CLASSES, "strict")
    assert mask_text("a piece of pie", p) == "a piece of [MASK]"


TEXTS = ["best apple pie ever", "classic carbonara with bacon", "salad next to french fries", "nothing relevant"]
LABELS = ["apple_pie", "spaghetti_carbonara", "caesar_salad", "fish_and_chips"]


def test_leakage_stats_none():
    s = leakage_stats(TEXTS, LABELS, CLASSES, "none")
    assert s["mode"] == "none" and s["n"] == 4
    assert s["own_label_rate"] == pytest.approx(0.25)
    assert s["own_word_rate"] == pytest.approx(0.75)
    assert s["other_label_rate"] == pytest.approx(0.25)
    assert s["mean_tokens"] == pytest.approx(3.75)


def test_leakage_stats_exact_and_strict():
    exact = leakage_stats(TEXTS, LABELS, CLASSES, "exact")
    assert exact["own_label_rate"] == 0.0
    assert exact["own_word_rate"] == pytest.approx(0.5)
    assert exact["other_label_rate"] == 0.0
    strict = leakage_stats(TEXTS, LABELS, CLASSES, "strict")
    assert strict["own_label_rate"] == strict["own_word_rate"] == strict["other_label_rate"] == 0.0


def test_leakage_stats_length_mismatch():
    with pytest.raises(ValueError):
        leakage_stats(["a"], [], CLASSES, "none")

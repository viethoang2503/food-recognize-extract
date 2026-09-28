"""Text cleaning, class-name masking and label-leakage statistics.

Masking runs on already-cleaned (lower-case) text, so the inserted "[MASK]" keeps its
upper case and is recognised as a special token by BERT tokenizers.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence

MASK_TOKEN = "[MASK]"
MASK_MODES = ("none", "exact", "strict")
STOPWORDS = frozenset({"and", "with", "de", "the", "a", "of", "in", "la", "le", "au", "en"})
MIN_WORD_LEN = 3

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_WS = re.compile(r"\s+")


def clean_text(text: object, max_chars: int = 5000) -> str:
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = _HTML_TAG.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    return s[:max_chars].strip()


def text_column(mode: str) -> str:
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown text mask mode {mode!r}; expected one of {MASK_MODES}")
    return "text" if mode == "none" else f"text_{mode}"


def class_to_phrase(name: str) -> str:
    return _WS.sub(" ", name.replace("_", " ")).strip().lower()


def word_variants(word: str) -> set[str]:
    variants = {word}
    if word.endswith("s") and len(word) > MIN_WORD_LEN:
        variants.add(word[:-1])
    else:
        variants.add(word + "s")
        if word.endswith(("ch", "sh", "x", "z")):
            variants.add(word + "es")
    return variants


def phrase_variants(phrase: str) -> set[str]:
    words = phrase.split()
    if not words:
        return set()
    return {" ".join([*words[:-1], v]) for v in word_variants(words[-1])}


def class_words(name: str) -> list[str]:
    return [w for w in class_to_phrase(name).split() if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]


def _compile(alternatives: Iterable[str]) -> re.Pattern[str] | None:
    alts = sorted({a for a in alternatives if a}, key=len, reverse=True)  # longest first
    if not alts:
        return None
    body = "|".join(r"\s+".join(re.escape(w) for w in a.split()) for a in alts)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


def build_mask_pattern(classes: Sequence[str], mode: str) -> re.Pattern[str] | None:
    """exact: every class phrase (+plural/singular); strict: exact + every single class word."""
    text_column(mode)  # validates mode
    if mode == "none":
        return None
    alts: set[str] = set()
    for c in classes:
        alts |= phrase_variants(class_to_phrase(c))
    if mode == "strict":
        for c in classes:
            for w in class_words(c):
                alts |= word_variants(w)
    return _compile(alts)


def mask_text(text: str, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return text
    return _WS.sub(" ", pattern.sub(MASK_TOKEN, text)).strip()


def _variant_index(classes: Sequence[str], words: bool) -> tuple[re.Pattern[str] | None, dict[str, set[str]]]:
    index: dict[str, set[str]] = {}
    for c in classes:
        variants: set[str] = set()
        if words:
            for w in class_words(c):
                variants |= word_variants(w)
        else:
            variants = phrase_variants(class_to_phrase(c))
        for v in variants:
            index.setdefault(v, set()).add(c)
    return _compile(index), index


def _matched_classes(text: str, pattern: re.Pattern[str] | None, index: dict[str, set[str]]) -> set[str]:
    if pattern is None:
        return set()
    found: set[str] = set()
    for m in pattern.finditer(text):
        found |= index.get(_WS.sub(" ", m.group(0)), set())
    return found


def leakage_stats(texts: Sequence[str], labels: Sequence[str], classes: Sequence[str], mode: str) -> dict:
    """Apply `mode` masking, then measure how much label information is left in the text."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    pattern = build_mask_pattern(classes, mode)
    phrase_pat, phrase_idx = _variant_index(classes, words=False)
    word_pat, word_idx = _variant_index(classes, words=True)
    n = len(texts)
    own = own_word = other = tokens = 0
    for text, label in zip(texts, labels):
        masked = mask_text(text, pattern)
        phrases = _matched_classes(masked, phrase_pat, phrase_idx)
        words = _matched_classes(masked, word_pat, word_idx)
        own += label in phrases
        own_word += label in words
        other += bool(phrases - {label})
        tokens += len(masked.split())
    denom = max(n, 1)
    return {
        "mode": mode,
        "n": n,
        "own_label_rate": own / denom,
        "own_word_rate": own_word / denom,
        "other_label_rate": other / denom,
        "mean_tokens": tokens / denom,
    }

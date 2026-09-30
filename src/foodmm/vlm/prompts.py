"""Prompts for the three extraction modes."""
from __future__ import annotations

MODES = ("image", "text", "image_text")
TEXT_MODES = ("text", "image_text")  # modes whose prompt contains the sample text

SYSTEM_PROMPT = (
    "You are a food analysis assistant. You get a food photo, a text about a dish, or both. "
    "Answer with ONLY one JSON object "
    "(no markdown, no explanation) with exactly these keys:\n"
    '{"dish_name": string, "cuisine": string, "main_ingredients": [string, ...], '
    '"cooking_method": string, "confidence": number between 0 and 1}\n'
    "Write every value in English. List at most 10 main ingredients that are shown, mentioned or very likely."
)

RETRY_PROMPT = (
    "Your previous answer was not valid ({error}). Reply again with ONLY the JSON object with the keys "
    "dish_name, cuisine, main_ingredients, cooking_method and confidence."
)

_TEXT_CONTEXT = ("Text from a web page about this dish. It may be noisy, and dish names were "
                 "replaced by [MASK]:\n\"\"\"\n{text}\n\"\"\"")


def build_prompt(mode: str, text: str | None = None, max_chars: int = 1000) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    first = "Identify the dish described in this text and describe it." if mode == "text" else \
        "Identify the dish in this photo and describe it."
    parts = [first]
    if mode in TEXT_MODES:
        snippet = str(text or "").strip()[:max_chars] or "(no text)"
        parts.append(_TEXT_CONTEXT.format(text=snippet))
    parts.append("Answer with the JSON object only.")
    return "\n\n".join(parts)

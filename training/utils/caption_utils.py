"""Normalize captions toward product-only garment photos (no person / model)."""

from __future__ import annotations

import re
from collections import Counter

_PERSON_RE = re.compile(
    r"\b("
    r"a |an |the )?"
    r"(man|woman|girl|boy|person|people|human|humans|model|models|"
    r"guy|lady|gentleman|child|kid|baby|"
    r"he|she|they|"
    r"wearing|worn by|dressed in|poses?|posing|standing|sitting|walking|"
    r"runway|street style|selfie"
    r")\b",
    re.IGNORECASE,
)

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")

PRODUCT_SUFFIX = (
    "product photo of an isolated fashion garment, "
    "ghost mannequin or flat lay, clean pure white background, "
    "no person, no model, no face, no hands, studio e-commerce catalog shot"
)

DEFAULT_PRODUCT_CAPTION = (
    "a fashion garment, " + PRODUCT_SUFFIX
)


def is_generic_product_caption(text: str | None) -> bool:
    """True for fallback/default captions with no garment-specific detail."""
    raw = (text or "").strip()
    if not raw:
        return True
    if raw == DEFAULT_PRODUCT_CAPTION:
        return True
    # Degenerate → normalize yields DEFAULT; also catch pre-normalize filler.
    lower = raw.lower()
    return lower.startswith("a fashion garment,") and "product photo" in lower


def is_degenerate_caption(text: str | None) -> bool:
    """Detect BLIP-style repetition loops / near-empty garbage captions."""
    raw = (text or "").strip()
    if len(raw) < 8:
        return True

    words = _WORD_RE.findall(raw.lower())
    if len(words) < 3:
        return True

    counts = Counter(words)
    top_word, top_n = counts.most_common(1)[0]
    if top_n >= 4 and top_n / len(words) >= 0.35:
        return True
    if top_n >= 6:
        return True

    # Long runs of the same token: "dam dam dam..."
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1

    return False


def normalize_product_caption(text: str | None) -> str:
    """Strip person/model language and append a strong product-photo suffix."""
    raw = (text or "").strip()
    if is_degenerate_caption(raw):
        return DEFAULT_PRODUCT_CAPTION

    cleaned = _PERSON_RE.sub(" ", raw)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip(" ,.;:-")

    if len(cleaned) < 6 or is_degenerate_caption(cleaned):
        return DEFAULT_PRODUCT_CAPTION

    lower = cleaned.lower()
    if "no person" in lower and "white background" in lower and "product" in lower:
        return cleaned

    return f"{cleaned}, {PRODUCT_SUFFIX}"

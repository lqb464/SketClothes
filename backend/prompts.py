"""Build text prompts for garment generation (FashionSD-X style)."""

DEFAULT_STYLE = (
    "isolated fashion garment, product photo, ghost mannequin, "
    "clean pure white background, no person"
)

PROMPT_TEMPLATE = (
    "{style}, fashion garment product photo, flat lay or ghost mannequin, "
    "isolated on pure white background, studio e-commerce catalog, "
    "no person, no model, no face, no hands"
)


def build_prompt(style: str = "", category: str | None = None) -> str:
    """Build prompt from free-text style. `category` is ignored (kept for API compat)."""
    _ = category
    style_text = style.strip() if style.strip() else DEFAULT_STYLE
    return PROMPT_TEMPLATE.format(style=style_text)

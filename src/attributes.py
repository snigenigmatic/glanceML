"""Fashion attribute vocabulary and lightweight NL query parsing.

Deterministic keyword parsing keeps compositionality explicit and avoids
depending on another LLM at query time.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


COLORS = [
    "yellow",
    "red",
    "blue",
    "white",
    "black",
    "green",
    "pink",
    "orange",
    "purple",
    "brown",
    "gray",
    "grey",
    "beige",
    "navy",
    "cream",
    "gold",
    "silver",
]

CLOTHING = [
    "raincoat",
    "coat",
    "jacket",
    "blazer",
    "suit",
    "hoodie",
    "sweater",
    "cardigan",
    "shirt",
    "t-shirt",
    "tee",
    "blouse",
    "dress",
    "skirt",
    "jeans",
    "pants",
    "trousers",
    "shorts",
    "tie",
    "scarf",
    "hat",
    "sneakers",
    "boots",
    "heels",
    "outerwear",
    "button-down",
    "button down",
]

SCENES = [
    "office",
    "park",
    "street",
    "urban",
    "city",
    "home",
    "indoor",
    "outdoor",
    "beach",
    "cafe",
    "restaurant",
    "studio",
    "runway",
    "formal setting",
]

STYLES = [
    "formal",
    "business",
    "professional",
    "casual",
    "streetwear",
    "sporty",
    "elegant",
    "chic",
    "vintage",
    "minimal",
    "weekend",
]

# Synonym / soft expansion used for metadata matching
COLOR_ALIASES = {
    "grey": "gray",
    "navy": "blue",
    "golden": "gold",
}

CLOTHING_ALIASES = {
    "tee": "t-shirt",
    "tshirt": "t-shirt",
    "t shirt": "t-shirt",
    "button down": "shirt",
    "button-down": "shirt",
    "trousers": "pants",
    "outerwear": "coat",
    "raincoat": "coat",
}

SCENE_ALIASES = {
    "urban": "street",
    "city": "street",
    "city walk": "street",
    "modern office": "office",
    "formal setting": "office",
    "park bench": "park",
}

STYLE_ALIASES = {
    "business": "formal",
    "professional": "formal",
    "weekend": "casual",
}


@dataclass
class FashionAttributes:
    colors: list[str] = field(default_factory=list)
    clothing: list[str] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    caption: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def normalized(self) -> "FashionAttributes":
        return FashionAttributes(
            colors=_normalize_list(self.colors, COLOR_ALIASES),
            clothing=_normalize_list(self.clothing, CLOTHING_ALIASES),
            scenes=_normalize_list(self.scenes, SCENE_ALIASES),
            styles=_normalize_list(self.styles, STYLE_ALIASES),
            caption=self.caption,
        )


def _normalize_token(token: str, aliases: dict[str, str]) -> str:
    t = token.strip().lower()
    return aliases.get(t, t)


def _normalize_list(values: list[str], aliases: dict[str, str]) -> list[str]:
    out: list[str] = []
    for v in values:
        n = _normalize_token(v, aliases)
        if n and n not in out:
            out.append(n)
    return out


def _find_terms(text: str, vocabulary: list[str]) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    # Longer phrases first to prefer "park bench" / "t-shirt"
    for term in sorted(vocabulary, key=len, reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lower) and term not in found:
            found.append(term)
    return found


def parse_attributes(text: str) -> FashionAttributes:
    """Extract multi-attribute signals from free text (query or caption)."""
    if not text:
        return FashionAttributes()
    attrs = FashionAttributes(
        colors=_find_terms(text, COLORS),
        clothing=_find_terms(text, CLOTHING),
        scenes=_find_terms(text, SCENES),
        styles=_find_terms(text, STYLES),
        caption=text.strip(),
    )
    # Light style inference from wording
    lower = text.lower()
    if any(w in lower for w in ("business", "professional", "blazer", "suit", "tie")):
        if "formal" not in attrs.styles:
            attrs.styles.append("formal")
    if any(w in lower for w in ("weekend", "hoodie", "casual", "city walk")):
        if "casual" not in attrs.styles:
            attrs.styles.append("casual")
    return attrs.normalized()


def metadata_match_score(query_attrs: FashionAttributes, doc_attrs: FashionAttributes) -> float:
    """Soft Jaccard-style overlap across attribute axes.

    Axes with no query constraint are ignored (do not penalize).
    Compositional queries get credit only when multiple constrained axes match.
    """
    q = query_attrs.normalized()
    d = doc_attrs.normalized()

    axis_scores: list[float] = []
    for q_vals, d_vals in (
        (q.colors, d.colors),
        (q.clothing, d.clothing),
        (q.scenes, d.scenes),
        (q.styles, d.styles),
    ):
        if not q_vals:
            continue
        d_set = set(d_vals)
        hits = sum(1 for v in q_vals if v in d_set)
        axis_scores.append(hits / len(q_vals))

    if not axis_scores:
        return 0.0
    return sum(axis_scores) / len(axis_scores)

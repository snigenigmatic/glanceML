"""Fashion attribute vocabulary, caption parsing, and metadata scoring.

Color–garment pairs are extracted from captions so compositional queries
("red tie and white shirt") can be scored on binding, not bag-of-words.
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
    "maroon",
    "burgundy",
    "lime",
]

CLOTHING = [
    "raincoat",
    "coat",
    "jacket",
    "blazer",
    "suit",
    "hoodie",
    "sweater",
    "sweatshirt",
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
    "bow tie",
    "scarf",
    "hat",
    "sneakers",
    "boots",
    "heels",
    "jumpsuit",
    "parka",
    "trench",
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
    "red carpet",
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

COLOR_ALIASES = {
    "grey": "gray",
    "navy": "blue",
    "golden": "gold",
    "maroon": "red",
    "burgundy": "red",
    "lime": "yellow",
}

CLOTHING_ALIASES = {
    "tee": "t-shirt",
    "tshirt": "t-shirt",
    "t shirt": "t-shirt",
    "button down": "shirt",
    "button-down": "shirt",
    "trousers": "pants",
    "outerwear": "coat",
    "parka": "coat",
    "trench": "coat",
    "sweatshirt": "hoodie",
    "bow tie": "tie",
}

SCENE_ALIASES = {
    "urban": "street",
    "city": "street",
    "city walk": "street",
    "modern office": "office",
    "formal setting": "office",
    "park bench": "park",
    "red carpet": "formal",
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
    # Explicit bindings, e.g. [("red", "tie"), ("white", "shirt")]
    pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = [f"{c}:{g}" for c, g in self.pairs]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FashionAttributes":
        pairs_raw = data.get("pairs") or []
        pairs: list[tuple[str, str]] = []
        for p in pairs_raw:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.append((str(p[0]), str(p[1])))
            elif isinstance(p, str) and ":" in p:
                c, g = p.split(":", 1)
                pairs.append((c, g))
        return cls(
            colors=list(data.get("colors") or []),
            clothing=list(data.get("clothing") or []),
            scenes=list(data.get("scenes") or []),
            styles=list(data.get("styles") or []),
            caption=data.get("caption") or "",
            pairs=pairs,
        )

    def normalized(self) -> "FashionAttributes":
        pairs = []
        for c, g in self.pairs:
            pairs.append((_normalize_token(c, COLOR_ALIASES), _normalize_token(g, CLOTHING_ALIASES)))
        # dedupe pairs
        seen = set()
        uniq_pairs = []
        for p in pairs:
            if p not in seen and p[0] and p[1]:
                seen.add(p)
                uniq_pairs.append(p)
        return FashionAttributes(
            colors=_normalize_list(self.colors, COLOR_ALIASES),
            clothing=_normalize_list(self.clothing, CLOTHING_ALIASES),
            scenes=_normalize_list(self.scenes, SCENE_ALIASES),
            styles=_normalize_list(self.styles, STYLE_ALIASES),
            caption=self.caption,
            pairs=uniq_pairs,
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
    for term in sorted(vocabulary, key=len, reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lower) and term not in found:
            found.append(term)
    return found


def extract_color_garment_pairs(text: str) -> list[tuple[str, str]]:
    """Find '<color> <garment>' bindings in free text."""
    if not text:
        return []
    color_alt = "|".join(sorted((re.escape(c) for c in COLORS), key=len, reverse=True))
    garment_alt = "|".join(sorted((re.escape(g) for g in CLOTHING), key=len, reverse=True))
    pattern = re.compile(
        rf"\b({color_alt})\s+({garment_alt})\b",
        flags=re.IGNORECASE,
    )
    pairs: list[tuple[str, str]] = []
    for m in pattern.finditer(text):
        pairs.append((m.group(1).lower(), m.group(2).lower()))
    # also "garment ... color" light patterns: "tie that is red", "shirt and a maroon tie"
    pattern2 = re.compile(
        rf"\b({garment_alt})\s+(?:that\s+is\s+|in\s+)?({color_alt})\b",
        flags=re.IGNORECASE,
    )
    for m in pattern2.finditer(text):
        pairs.append((m.group(2).lower(), m.group(1).lower()))
    return pairs


def parse_attributes(text: str) -> FashionAttributes:
    """Extract multi-attribute signals from free text (query or caption)."""
    if not text:
        return FashionAttributes()
    pairs = extract_color_garment_pairs(text)
    attrs = FashionAttributes(
        colors=_find_terms(text, COLORS),
        clothing=_find_terms(text, CLOTHING),
        scenes=_find_terms(text, SCENES),
        styles=_find_terms(text, STYLES),
        caption=text.strip(),
        pairs=pairs,
    )
    # Prefer colors/clothing attested in pairs when present (tighter binding)
    if pairs:
        pair_colors = [c for c, _ in pairs]
        pair_clothes = [g for _, g in pairs]
        # keep unpaired extras, but put pair-attested first
        attrs.colors = _normalize_list(pair_colors + attrs.colors, COLOR_ALIASES)
        attrs.clothing = _normalize_list(pair_clothes + attrs.clothing, CLOTHING_ALIASES)
    lower = text.lower()
    if any(w in lower for w in ("business", "professional", "blazer", "suit", "tie", "red carpet")):
        if "formal" not in attrs.styles:
            attrs.styles.append("formal")
    if any(w in lower for w in ("weekend", "hoodie", "casual", "city walk", "sneakers")):
        if "casual" not in attrs.styles:
            attrs.styles.append("casual")
    return attrs.normalized()


def parse_caption_attributes(caption: str, seed_text: str = "") -> FashionAttributes:
    """Strict image metadata: caption-primary, seed categories as soft clothing priors only.

    Does NOT merge open-vocabulary detection dumps (those caused 15+ garment catch-alls).
    """
    attrs = parse_attributes(caption)
    if seed_text:
        seed = parse_attributes(seed_text)
        # Only add seed clothing categories that are real words (not numeric ids)
        for g in seed.clothing:
            if g not in attrs.clothing and not g.isdigit():
                attrs.clothing.append(g)
        for s in seed.styles:
            if s not in attrs.styles:
                attrs.styles.append(s)
    # Cap clothing to keep metadata discriminative (caption order first)
    attrs.clothing = attrs.clothing[:6]
    attrs.colors = attrs.colors[:6]
    attrs.caption = caption.strip()
    return attrs.normalized()


def metadata_match_score(query_attrs: FashionAttributes, doc_attrs: FashionAttributes) -> float:
    """Soft overlap across attribute axes, with extra weight on color–garment pairs."""
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

    if q.pairs:
        # Exact color–garment bindings only. A loose colors×clothing product
        # would score "red shirt + white tie" the same as "red tie + white shirt".
        d_pair_set = set(d.pairs)
        hits = sum(1 for p in q.pairs if p in d_pair_set)
        axis_scores.append(hits / len(q.pairs))

    if not axis_scores:
        return 0.0
    return sum(axis_scores) / len(axis_scores)

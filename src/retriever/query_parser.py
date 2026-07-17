"""Natural-language query parsing for multi-attribute fashion search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "pink",
    "purple",
    "orange",
    "navy",
    "beige",
    "bright yellow",
}

CLOTHING_WORDS = {
    "shirt",
    "tie",
    "blazer",
    "suit",
    "dress",
    "pants",
    "jeans",
    "hoodie",
    "t-shirt",
    "tee",
    "raincoat",
    "coat",
    "jacket",
    "skirt",
    "shorts",
    "sweater",
    "top",
    "dress shirt",
    "button-down",
}

SCENE_WORDS = {
    "office",
    "modern office",
    "park",
    "bench",
    "street",
    "city",
    "urban",
    "home",
    "indoor",
    "outdoor",
    "formal setting",
}

STYLE_WORDS = {
    "casual",
    "formal",
    "professional",
    "business",
    "weekend",
    "city walk",
}


@dataclass
class ParsedQuery:
    raw: str
    colors: list[str] = field(default_factory=list)
    clothing: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)

    @property
    def has_structure(self) -> bool:
        return bool(self.colors or self.clothing or self.scene or self.style)


def _find_matches(text: str, vocabulary: set[str]) -> list[str]:
    lowered = text.lower()
    matches = []
    for token in sorted(vocabulary, key=len, reverse=True):
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            matches.append(token)
    return matches


def parse_query(query: str) -> ParsedQuery:
    lowered = query.lower()
    parsed = ParsedQuery(
        raw=query,
        colors=_find_matches(lowered, COLOR_WORDS),
        clothing=_find_matches(lowered, CLOTHING_WORDS),
        scene=_find_matches(lowered, SCENE_WORDS),
        style=_find_matches(lowered, STYLE_WORDS),
    )

    # Compositional clauses improve reranking beyond a single global embedding.
    split_pattern = r"\b(?:and|with|in|on|for|wearing|sitting)\b"
    parsed.phrases = [
        part.strip(" .")
        for part in re.split(split_pattern, lowered)
        if part.strip(" .")
    ] or [lowered]
    return parsed

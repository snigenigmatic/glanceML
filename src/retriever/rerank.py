"""Lightweight attribute-aware reranker over the top candidate pool.

Avoids a second heavy cross-encoder: boosts candidates that satisfy more
query constraints (color ∩ clothing ∩ scene), which is where vanilla CLIP
typically fails on compositional fashion prompts.
"""

from __future__ import annotations

from src.attributes import COLOR_ALIASES, FashionAttributes
from src.retriever.types import RetrievalResult

# reverse map: canonical color -> surface forms that may appear in captions
_COLOR_SURFACE = {
    "red": ["red", "maroon", "burgundy"],
    "yellow": ["yellow", "lime", "gold"],
    "blue": ["blue", "navy"],
    "gray": ["gray", "grey"],
}
for _src, _dst in COLOR_ALIASES.items():
    _COLOR_SURFACE.setdefault(_dst, [_dst]).append(_src)


def composition_bonus(query_attrs: FashionAttributes, result: RetrievalResult) -> float:
    q = query_attrs.normalized()
    axes_present = 0
    axes_hit = 0

    checks = (
        (q.colors, result.colors),
        (q.clothing, result.clothing),
        (q.scenes, result.scenes),
        (q.styles, result.styles),
    )
    for q_vals, r_vals in checks:
        if not q_vals:
            continue
        axes_present += 1
        r_set = set(r_vals)
        if any(v in r_set for v in q_vals):
            axes_hit += 1

    pair_bonus = 0.0
    if q.pairs:
        # Prefer captions that literally bind the requested color–garment pairs.
        caption = (result.caption or "").lower()
        hits = 0.0
        for color, garment in q.pairs:
            surfaces = _COLOR_SURFACE.get(color, [color])
            bound = any(
                f"{s} {garment}" in caption or f"{s} bow {garment}" in caption
                for s in surfaces
            )
            if bound:
                hits += 1.0
            elif color in result.colors and garment in result.clothing:
                # weaker structured credit (not caption-proven)
                hits += 0.35
        pair_bonus = 0.4 * (hits / len(q.pairs))

    if axes_present == 0:
        return pair_bonus
    coverage = axes_hit / axes_present
    joint = 0.15 if axes_hit >= 2 else 0.0
    return 0.25 * coverage + joint + pair_bonus


def rerank_candidates(
    query: str,
    query_attrs: FashionAttributes,
    candidates: list[RetrievalResult],
) -> list[RetrievalResult]:
    rescored: list[RetrievalResult] = []
    for c in candidates:
        bonus = composition_bonus(query_attrs, c)
        # Soft caption overlap for leftover tokens
        caption_bonus = 0.0
        if c.caption and query:
            q_tokens = {t for t in query.lower().split() if len(t) > 3}
            c_tokens = set(c.caption.lower().split())
            if q_tokens:
                caption_bonus = 0.05 * len(q_tokens & c_tokens) / len(q_tokens)
        new_score = c.score + bonus + caption_bonus
        rescored.append(
            RetrievalResult(
                image_id=c.image_id,
                path=c.path,
                score=new_score,
                dense_score=c.dense_score,
                metadata_score=c.metadata_score,
                caption=c.caption,
                colors=c.colors,
                clothing=c.clothing,
                scenes=c.scenes,
                styles=c.styles,
            )
        )
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored

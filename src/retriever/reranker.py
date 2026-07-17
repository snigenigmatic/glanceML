"""Lightweight compositional reranking for top candidates."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from src.models.embeddings import EmbeddingModel
from src.retriever.query_parser import ParsedQuery


def rerank_candidates(
    model: EmbeddingModel,
    query: ParsedQuery,
    candidates: list[dict[str, Any]],
    images: list[Image.Image],
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    global_embedding = model.encode_texts([query.raw])[0]
    phrase_embeddings = model.encode_texts(query.phrases)

    image_embeddings = model.encode_images(images)
    scores: list[float] = []

    for index in range(len(candidates)):
        semantic = float((image_embeddings[index] @ global_embedding).item())
        phrase_score = float((image_embeddings[index] @ phrase_embeddings.T).mean().item())
        metadata_score = float(candidates[index].get("metadata_score", 0.0))
        final = 0.55 * semantic + 0.25 * phrase_score + 0.20 * metadata_score
        scores.append(final)
        candidates[index]["rerank_score"] = final
        candidates[index]["phrase_score"] = phrase_score

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [candidate for candidate, _ in ranked[:top_k]]

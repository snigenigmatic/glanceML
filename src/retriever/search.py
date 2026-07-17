"""Hybrid dense + metadata retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch
from PIL import Image

from src.config import load_config, resolve_path
from src.data.preprocess import load_metadata
from src.models.embeddings import EmbeddingModel
from src.models.vlm import VLMMetadataExtractor
from src.retriever.query_parser import ParsedQuery, parse_query
from src.retriever.reranker import rerank_candidates


def _metadata_overlap(query_values: list[str], doc_values: list[str] | str) -> float:
    if isinstance(doc_values, str):
        doc_values = [doc_values]
    if not query_values:
        return 0.0
    query_set = {value.lower() for value in query_values}
    doc_set = {value.lower() for value in doc_values}
    if not doc_set:
        return 0.0
    return len(query_set & doc_set) / len(query_set)


def metadata_score(parsed: ParsedQuery, record: dict[str, Any]) -> float:
    parts = [
        _metadata_overlap(parsed.colors, record.get("colors", [])),
        _metadata_overlap(parsed.clothing, record.get("clothing", [])),
        _metadata_overlap(parsed.scene, record.get("scene", [])),
        _metadata_overlap(parsed.style, record.get("style", [])),
    ]
    weights = [0.25, 0.35, 0.25, 0.15]
    return float(sum(weight * score for weight, score in zip(weights, parts)))


def _open_collection(config: dict[str, Any]):
    chroma_dir = resolve_path(config, "chroma_dir")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=config["retrieval"]["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )


class FashionRetriever:
    def __init__(
        self,
        config_path: str | Path | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        model_name = embedding_model_name or self.config["models"]["embedding_model"]
        self.embedding_model = EmbeddingModel.from_pretrained(model_name)
        self.collection = _open_collection(self.config)
        self.metadata = {
            record["id"]: record
            for record in load_metadata(resolve_path(self.config, "metadata_file"))
        }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        top_k = top_k or int(self.config["retrieval"]["top_k"])
        initial_k = int(self.config["retrieval"]["initial_candidates"])
        rerank_k = int(self.config["retrieval"]["rerank_candidates"])
        semantic_weight = float(self.config["retrieval"]["semantic_weight"])
        metadata_weight = float(self.config["retrieval"]["metadata_weight"])

        parsed = parse_query(query)
        query_embedding = self.embedding_model.encode_texts([query])[0].cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(initial_k, self.collection.count()),
            include=["metadatas", "distances", "embeddings"],
        )

        candidates: list[dict[str, Any]] = []
        for index, doc_id in enumerate(results["ids"][0]):
            record = self.metadata.get(doc_id, {})
            chroma_distance = results["distances"][0][index]
            semantic = 1.0 - float(chroma_distance)
            meta = metadata_score(parsed, record)
            fused = semantic_weight * semantic + metadata_weight * meta

            candidates.append(
                {
                    "id": doc_id,
                    "image_path": record.get("image_path"),
                    "semantic_score": semantic,
                    "metadata_score": meta,
                    "score": fused,
                    "matched_attributes": {
                        "colors": sorted(
                            set(parsed.colors) & {value.lower() for value in record.get("colors", [])}
                        ),
                        "clothing": sorted(
                            set(parsed.clothing)
                            & {value.lower() for value in record.get("clothing", [])}
                        ),
                        "scene": sorted(
                            set(parsed.scene)
                            & {
                                value.lower()
                                for value in (
                                    record.get("scene", [])
                                    if isinstance(record.get("scene"), list)
                                    else [record.get("scene", "")]
                                )
                            }
                        ),
                        "style": sorted(
                            set(parsed.style)
                            & {
                                value.lower()
                                for value in (
                                    record.get("style", [])
                                    if isinstance(record.get("style"), list)
                                    else [record.get("style", "")]
                                )
                            }
                        ),
                    },
                    "caption": record.get("caption", ""),
                    "record": record,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        shortlist = candidates[:rerank_k]

        if use_reranker and shortlist:
            images = [
                Image.open(item["image_path"]).convert("RGB")
                for item in shortlist
                if item["image_path"] and Path(item["image_path"]).exists()
            ]
            if len(images) == len(shortlist):
                return rerank_candidates(
                    model=self.embedding_model,
                    query=parsed,
                    candidates=shortlist,
                    images=images,
                    top_k=top_k,
                )

        return shortlist[:top_k]


def search(
    query: str,
    config_path: str | Path | None = None,
    top_k: int | None = None,
    use_reranker: bool = True,
) -> list[dict[str, Any]]:
    retriever = FashionRetriever(config_path=config_path)
    return retriever.search(query=query, top_k=top_k, use_reranker=use_reranker)

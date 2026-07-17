"""Part B — Retriever: dense search + metadata scoring + weighted fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.attributes import FashionAttributes, metadata_match_score, parse_attributes
from src.indexer.vector_store import get_client, get_or_create_collection, query_collection
from src.models.embeddings import EmbeddingModel, load_fashion_siglip
from src.retriever.rerank import rerank_candidates
from src.retriever.types import RetrievalResult
from src.utils import load_config, resolve_path


def _meta_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x for x in str(value).split(",") if x]


def _distance_to_similarity(distance: float) -> float:
    # Chroma cosine distance ≈ 1 - cosine_similarity
    return max(0.0, 1.0 - float(distance))


class FashionRetriever:
    def __init__(
        self,
        config_path: str | None = None,
        data_dir: str | Path | None = None,
        embedder: EmbeddingModel | None = None,
    ):
        self.cfg = load_config(config_path)
        self.data_root = resolve_path(data_dir or self.cfg["dataset"]["output_dir"])
        chroma_dir = resolve_path(self.cfg["index"]["chroma_dir"])
        if data_dir is not None:
            chroma_dir = resolve_path(Path(data_dir) / "chroma")
        client = get_client(chroma_dir)
        self.collection = get_or_create_collection(
            client, self.cfg["index"]["collection_name"]
        )
        device = self.cfg["models"].get("device", "cuda")
        self.embedder = embedder or load_fashion_siglip(
            self.cfg["models"]["embedding"], device=device
        )
        self.dense_weight = float(self.cfg["retrieval"]["dense_weight"])
        self.metadata_weight = float(self.cfg["retrieval"]["metadata_weight"])
        self.candidate_pool = int(self.cfg["retrieval"]["candidate_pool"])
        self.rerank_enabled = bool(self.cfg["retrieval"].get("rerank", True))
        self.rerank_pool = int(self.cfg["retrieval"].get("rerank_pool", 20))

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or int(self.cfg["retrieval"]["top_k"])
        query_attrs = parse_attributes(query)
        q_vec = self.embedder.embed_texts([query])[0].tolist()

        raw = query_collection(
            self.collection,
            query_embeddings=[q_vec],
            n_results=min(self.candidate_pool, max(self.collection.count(), 1)),
        )

        ids = raw["ids"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]
        documents = raw["documents"][0]

        candidates: list[RetrievalResult] = []
        for image_id, dist, md, doc in zip(ids, distances, metadatas, documents):
            doc_attrs = FashionAttributes(
                colors=_meta_list(md.get("colors")),
                clothing=_meta_list(md.get("clothing")),
                scenes=_meta_list(md.get("scenes")),
                styles=_meta_list(md.get("styles")),
                caption=md.get("caption") or doc or "",
            )
            dense = _distance_to_similarity(dist)
            meta_score = metadata_match_score(query_attrs, doc_attrs)
            fused = self.dense_weight * dense + self.metadata_weight * meta_score
            candidates.append(
                RetrievalResult(
                    image_id=image_id,
                    path=md.get("path", ""),
                    score=fused,
                    dense_score=dense,
                    metadata_score=meta_score,
                    caption=doc_attrs.caption,
                    colors=doc_attrs.colors,
                    clothing=doc_attrs.clothing,
                    scenes=doc_attrs.scenes,
                    styles=doc_attrs.styles,
                )
            )

        candidates.sort(key=lambda r: r.score, reverse=True)

        if self.rerank_enabled and candidates:
            pool = candidates[: max(self.rerank_pool, top_k)]
            rest = candidates[len(pool) :]
            pool = rerank_candidates(query, query_attrs, pool)
            candidates = pool + rest

        return candidates[:top_k]


def search_query(
    query: str,
    top_k: int = 5,
    config_path: str | None = None,
    data_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    retriever = FashionRetriever(config_path=config_path, data_dir=data_dir)
    return [r.to_dict() for r in retriever.search(query, top_k=top_k)]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query fashion retrieval index")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    results = search_query(
        args.query, top_k=args.top_k, config_path=args.config, data_dir=args.data_dir
    )
    print(json.dumps(results, indent=2))

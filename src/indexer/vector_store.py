"""ChromaDB vector store helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

# Must be set before chromadb import — broken posthog stubs spam ERROR logs.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import chromadb
from chromadb.config import Settings

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def get_client(persist_dir: str | Path) -> chromadb.ClientAPI:
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(client: chromadb.ClientAPI, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_records(
    collection,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=list(ids[i : i + batch_size]),
            embeddings=list(embeddings[i : i + batch_size]),
            documents=list(documents[i : i + batch_size]),
            metadatas=list(metadatas[i : i + batch_size]),
        )


def query_collection(
    collection,
    query_embeddings: Sequence[Sequence[float]],
    n_results: int = 50,
    where: dict | None = None,
) -> dict:
    kwargs = {
        "query_embeddings": list(query_embeddings),
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)

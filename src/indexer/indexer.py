"""Part A: build the searchable vector index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from PIL import Image
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.data.preprocess import load_metadata
from src.models.embeddings import EmbeddingModel
from src.models.vlm import VLMMetadataExtractor


def _flatten(values: Any) -> str:
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values)


def build_index(
    config_path: str | Path | None = None,
    force: bool = False,
    use_vlm: bool = True,
) -> dict[str, Any]:
    """Extract embeddings + metadata and persist them in ChromaDB."""
    config = load_config(config_path)
    metadata_file = resolve_path(config, "metadata_file")
    chroma_dir = resolve_path(config, "chroma_dir")
    chroma_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(metadata_file)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection_name = config["retrieval"]["collection_name"]

    if force:
        try:
            client.delete_collection(collection_name)
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() > 0 and not force:
        return {
            "status": "exists",
            "count": collection.count(),
            "chroma_dir": str(chroma_dir),
        }

    embedding_model = EmbeddingModel.from_pretrained(config["models"]["embedding_model"])
    vlm = (
        VLMMetadataExtractor.from_pretrained(
            model_name=config["models"]["vlm_model"],
            task=config["models"]["vlm_task"],
        )
        if use_vlm
        else None
    )

    batch_size = 16
    for start in tqdm(range(0, len(records), batch_size), desc="Indexing"):
        batch = records[start : start + batch_size]
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        embeddings = embedding_model.encode_images(images).cpu().numpy().tolist()

        ids: list[str] = []
        metadatas: list[dict[str, str]] = []
        documents: list[str] = []

        for record, image in zip(batch, images):
            enriched = dict(record)
            if vlm is not None:
                vlm_metadata = vlm.extract_metadata(
                    image=image,
                    seed={
                        "clothing": record.get("clothing", []),
                        "colors": record.get("colors", []),
                        "scene": [record.get("scene", "unknown")],
                        "style": [record.get("style", "unknown")],
                    },
                )
                enriched.update(vlm_metadata)

            record["caption"] = enriched.get("caption", record.get("caption_seed", ""))
            record["colors"] = enriched.get("colors", record.get("colors", []))
            record["clothing"] = enriched.get("clothing", record.get("clothing", []))
            record["scene"] = enriched.get("scene", record.get("scene", "unknown"))
            record["style"] = enriched.get("style", record.get("style", "unknown"))

            ids.append(record["id"])
            metadatas.append(
                {
                    "image_path": record["image_path"],
                    "caption": _flatten(record.get("caption", "")),
                    "colors": _flatten(record.get("colors", [])),
                    "clothing": _flatten(record.get("clothing", [])),
                    "scene": _flatten(record.get("scene", "unknown")),
                    "style": _flatten(record.get("style", "unknown")),
                }
            )
            documents.append(record.get("caption") or record.get("caption_seed", ""))

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    # Persist enriched metadata for retrieval-time attribute scoring.
    from src.data.preprocess import save_metadata

    save_metadata(metadata_file, records)

    return {
        "status": "built",
        "count": collection.count(),
        "chroma_dir": str(chroma_dir),
        "metadata_file": str(metadata_file),
    }

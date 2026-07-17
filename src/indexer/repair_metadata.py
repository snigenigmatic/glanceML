"""Re-parse captions into clean metadata without re-running Florence-2 / re-embedding.

Fixes the open-vocab catch-all clothing lists and restores color–garment pairs
into Chroma metadata while keeping existing FashionSigLIP vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.attributes import FashionAttributes, parse_caption_attributes
from src.indexer.vector_store import get_client, get_or_create_collection
from src.utils import load_config, read_json, resolve_path, write_json


def _flatten_metadata(image_id: str, path: str, attrs: FashionAttributes, buckets: dict) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "path": path,
        "caption": (attrs.caption or "")[:1000],
        "colors": ",".join(attrs.colors),
        "clothing": ",".join(attrs.clothing),
        "scenes": ",".join(attrs.scenes),
        "styles": ",".join(attrs.styles),
        "pairs": ",".join(f"{c}:{g}" for c, g in attrs.pairs),
        "bucket_color": buckets.get("color", "unknown"),
        "bucket_clothing": buckets.get("clothing", "unknown"),
        "bucket_scene": buckets.get("scene", "unknown"),
    }


def repair_metadata(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    cfg = load_config(config_path)
    data_root = resolve_path(data_dir or cfg["dataset"]["output_dir"])
    index_meta_path = data_root / "index_meta.json"
    if not index_meta_path.exists():
        raise FileNotFoundError(f"Missing {index_meta_path}; build_index first.")

    index_meta = read_json(index_meta_path)
    items = index_meta["items"]

    chroma_dir = resolve_path(cfg["index"]["chroma_dir"])
    if data_dir is not None:
        chroma_dir = resolve_path(Path(data_dir) / "chroma")

    client = get_client(chroma_dir)
    collection = get_or_create_collection(client, cfg["index"]["collection_name"])

    # Pull existing embeddings so we can upsert cleaned metadata
    existing = collection.get(include=["embeddings", "documents"])
    id_to_emb = {i: e for i, e in zip(existing["ids"], existing["embeddings"])}

    repaired_items = []
    ids, embeddings, documents, metadatas = [], [], [], []
    color_hist: dict[str, int] = {}
    clothing_len_hist: dict[int, int] = {}

    for item in items:
        caption = (item.get("vlm") or {}).get("caption") or item.get("seed_text") or ""
        seed_text = item.get("seed_text") or ""
        attrs = parse_caption_attributes(caption, seed_text=seed_text)
        # Prefer caption scene; else seed bucket
        if not attrs.scenes:
            scene = item.get("buckets", {}).get("scene")
            if scene:
                attrs.scenes = [scene]

        image_id = item["image_id"]
        if image_id not in id_to_emb:
            # Skip orphans; shouldn't happen
            continue

        flat = _flatten_metadata(image_id, item["path"], attrs, item.get("buckets", {}))
        ids.append(image_id)
        embeddings.append(id_to_emb[image_id])
        documents.append(attrs.caption or seed_text)
        metadatas.append(flat)

        enriched = dict(item)
        enriched["vlm"] = attrs.to_dict()
        repaired_items.append(enriched)

        primary = attrs.colors[0] if attrs.colors else "unknown"
        color_hist[primary] = color_hist.get(primary, 0) + 1
        n = len(attrs.clothing)
        clothing_len_hist[n] = clothing_len_hist.get(n, 0) + 1

    # Batch upsert
    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    index_meta["items"] = repaired_items
    index_meta["metadata_repair"] = {
        "method": "caption_reparse_v2",
        "color_primary_histogram": color_hist,
        "clothing_len_histogram": {str(k): v for k, v in sorted(clothing_len_hist.items())},
        "count": len(ids),
    }
    write_json(index_meta_path, index_meta)

    # Refresh dataset_summary-style stats on volume
    summary = {
        "dataset": "fashionpedia-subset",
        "count": len(ids),
        "vlm_color_primary_histogram": color_hist,
        "clothing_len_histogram": {str(k): v for k, v in sorted(clothing_len_hist.items())},
        "note": "Colors from caption reparse (not seed Fashionpedia attribute ids).",
    }
    write_json(data_root / "eval_results" / "dataset_summary.json", summary)
    print("Repaired metadata for", len(ids), "images")
    print("Primary color hist:", color_hist)
    print("Clothing len hist:", clothing_len_hist)
    return index_meta_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    repair_metadata(config_path=args.config, data_dir=args.data_dir)

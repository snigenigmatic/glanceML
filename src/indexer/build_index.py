"""Part A — Indexer: embed images, extract VLM metadata, persist to ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.attributes import FashionAttributes
from src.indexer.vector_store import get_client, get_or_create_collection, upsert_records
from src.models.embeddings import load_fashion_siglip
from src.models.vlm import Florence2Extractor, HeuristicExtractor
from src.utils import ensure_dir, load_config, read_json, resolve_path, write_json


def _flatten_metadata(image_id: str, path: str, attrs: FashionAttributes, buckets: dict) -> dict[str, Any]:
    """Chroma metadata values must be scalars."""
    return {
        "image_id": image_id,
        "path": path,
        "caption": attrs.caption[:1000] if attrs.caption else "",
        "colors": ",".join(attrs.colors),
        "clothing": ",".join(attrs.clothing),
        "scenes": ",".join(attrs.scenes),
        "styles": ",".join(attrs.styles),
        "pairs": ",".join(f"{c}:{g}" for c, g in attrs.pairs),
        "bucket_color": buckets.get("color", "unknown"),
        "bucket_clothing": buckets.get("clothing", "unknown"),
        "bucket_scene": buckets.get("scene", "unknown"),
    }


def build_index(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
    use_vlm: bool = True,
    max_items: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    data_root = resolve_path(data_dir or cfg["dataset"]["output_dir"])
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run dataset prep first "
            "(python -m src.dataset.prepare or modal run modal_app.py::prepare_dataset)."
        )

    meta = read_json(meta_path)
    items = meta["items"]
    if max_items is not None:
        items = items[:max_items]

    chroma_dir = resolve_path(cfg["index"]["chroma_dir"])
    if str(cfg["index"]["chroma_dir"]).startswith("data/") and data_dir is not None:
        chroma_dir = resolve_path(Path(data_dir) / "chroma")
    ensure_dir(chroma_dir)

    device = cfg["models"].get("device", "cuda")
    print(f"Loading FashionSigLIP on {device} …")
    embedder = load_fashion_siglip(cfg["models"]["embedding"], device=device)

    extractor: Florence2Extractor | HeuristicExtractor
    if use_vlm:
        print(f"Loading Florence-2 ({cfg['models']['vlm']}) …")
        try:
            extractor = Florence2Extractor(cfg["models"]["vlm"], device=device)
        except Exception as exc:
            print(f"Florence-2 unavailable ({exc}); falling back to heuristic extractor.")
            extractor = HeuristicExtractor()
    else:
        extractor = HeuristicExtractor()

    client = get_client(chroma_dir)
    # Rebuild cleanly for reproducibility
    try:
        client.delete_collection(cfg["index"]["collection_name"])
    except Exception:
        pass
    collection = get_or_create_collection(client, cfg["index"]["collection_name"])

    batch_size = int(cfg["index"].get("batch_size", 16))
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    enriched_items: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(items), batch_size), desc="Indexing"):
        batch = items[start : start + batch_size]
        images: list[Image.Image] = []
        batch_attrs: list[FashionAttributes] = []
        for item in batch:
            img_path = data_root / item["path"]
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            attrs = extractor.extract(image, seed_text=item.get("seed_text", ""))
            # Prefer VLM scene; else seed bucket scene
            if not attrs.scenes:
                scene = item.get("buckets", {}).get("scene")
                if scene:
                    attrs.scenes = [scene]
            batch_attrs.append(attrs)

        vecs = embedder.embed_images(images, batch_size=batch_size)
        for item, attrs, vec in zip(batch, batch_attrs, vecs):
            ids.append(item["image_id"])
            embeddings.append(vec.tolist())
            documents.append(attrs.caption or item.get("seed_text", ""))
            metadatas.append(
                _flatten_metadata(item["image_id"], item["path"], attrs, item.get("buckets", {}))
            )
            enriched = dict(item)
            enriched["vlm"] = attrs.to_dict()
            enriched_items.append(enriched)

    upsert_records(collection, ids, embeddings, documents, metadatas)
    index_meta = {
        "collection": cfg["index"]["collection_name"],
        "count": len(ids),
        "embedding_model": cfg["models"]["embedding"],
        "vlm_model": cfg["models"]["vlm"] if use_vlm else "heuristic",
        "chroma_dir": str(chroma_dir),
        "items": enriched_items,
    }
    index_meta_path = data_root / "index_meta.json"
    write_json(index_meta_path, index_meta)
    print(f"Indexed {len(ids)} images → {chroma_dir}")
    return index_meta_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build fashion retrieval index")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()
    build_index(
        config_path=args.config,
        data_dir=args.data_dir,
        use_vlm=not args.no_vlm,
        max_items=args.max_items,
    )

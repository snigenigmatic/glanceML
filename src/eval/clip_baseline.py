"""Vanilla CLIP corpus index for baseline comparison (numpy, not Chroma)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.embeddings import load_clip_baseline
from src.utils import load_config, read_json, resolve_path, write_json


def build_clip_index(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
    batch_size: int = 32,
) -> Path:
    cfg = load_config(config_path)
    data_root = resolve_path(data_dir or cfg["dataset"]["output_dir"])
    meta = read_json(data_root / "metadata.json")
    items = meta["items"]

    device = cfg["models"].get("device", "cuda")
    model_id = cfg["models"]["baseline_clip"]
    print(f"Building CLIP baseline ({model_id}) on {device} …")
    clip = load_clip_baseline(model_id, device=device)

    ids: list[str] = []
    paths: list[str] = []
    vectors: list[np.ndarray] = []

    for start in tqdm(range(0, len(items), batch_size), desc="CLIP embed"):
        batch = items[start : start + batch_size]
        images = [Image.open(data_root / it["path"]).convert("RGB") for it in batch]
        vecs = clip.embed_images(images, batch_size=batch_size)
        for it, v in zip(batch, vecs):
            ids.append(it["image_id"])
            paths.append(it["path"])
            vectors.append(v)

    mat = np.stack(vectors, axis=0).astype(np.float32)
    out_dir = data_root / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clip_embeddings.npy", mat)
    write_json(
        out_dir / "clip_ids.json",
        {"model": model_id, "ids": ids, "paths": paths, "count": len(ids)},
    )
    print(f"Wrote CLIP index ({len(ids)}) → {out_dir}")
    return out_dir / "clip_embeddings.npy"


class ClipBaselineRetriever:
    def __init__(self, data_dir: str | Path, config_path: str | None = None):
        cfg = load_config(config_path)
        self.data_root = resolve_path(data_dir)
        base = self.data_root / "baselines"
        meta = read_json(base / "clip_ids.json")
        self.ids = meta["ids"]
        self.paths = meta["paths"]
        self.mat = np.load(base / "clip_embeddings.npy")
        device = cfg["models"].get("device", "cuda")
        self.clip = load_clip_baseline(cfg["models"]["baseline_clip"], device=device)
        # captions from index_meta if available
        self.captions = {}
        index_meta = self.data_root / "index_meta.json"
        if index_meta.exists():
            for it in read_json(index_meta)["items"]:
                self.captions[it["image_id"]] = (it.get("vlm") or {}).get("caption", "")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.clip.embed_texts([query])[0]
        scores = self.mat @ q
        order = np.argsort(-scores)[:top_k]
        out = []
        for rank, i in enumerate(order, 1):
            image_id = self.ids[int(i)]
            out.append(
                {
                    "image_id": image_id,
                    "path": self.paths[int(i)],
                    "score": float(scores[int(i)]),
                    "dense_score": float(scores[int(i)]),
                    "metadata_score": 0.0,
                    "caption": self.captions.get(image_id, ""),
                    "matched_attributes": {
                        "colors": [],
                        "clothing": [],
                        "scenes": [],
                        "styles": [],
                    },
                    "rank": rank,
                }
            )
        return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    build_clip_index(config_path=args.config, data_dir=args.data_dir)

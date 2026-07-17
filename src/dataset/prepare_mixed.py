"""Mixed corpus: CUHK-PEDES (attribute captions) + COCO people-in-context (scenes).

Designed to fix Fashionpedia's runway skew for the assignment axes:
  - clothing / color language → PEDES captions
  - environment (park / street / home / office-ish) → COCO person images
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.attributes import parse_attributes
from src.utils import ensure_dir, resolve_path, write_json

# detection-datasets/coco uses 0-indexed ClassLabel names (person=0).
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
COCO_CATS = {i: name for i, name in enumerate(COCO_NAMES)}
PERSON_ID = 0

SCENE_RULES = [
    ("park", {"bench", "bird", "dog", "frisbee", "kite", "sports ball"}),
    ("street", {"car", "bus", "truck", "motorcycle", "traffic light", "stop sign", "bicycle"}),
    ("home", {"couch", "bed", "dining table", "tv", "refrigerator", "oven", "sink", "microwave"}),
    ("office", {"laptop", "keyboard", "mouse", "book", "cell phone"}),
]


def _save_image(image: Image.Image, path: Path, image_size: int) -> None:
    """Resize with letterbox padding so PEDES tall crops are not crushed."""
    image = image.convert("RGB")
    w, h = image.size
    scale = image_size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    canvas.paste(resized, ((image_size - nw) // 2, (image_size - nh) // 2))
    canvas.save(path, quality=90)


def _infer_scene_from_coco(cat_names: set[str]) -> str:
    for scene, cues in SCENE_RULES:
        if cat_names & cues:
            return scene
    if "umbrella" in cat_names:
        return "street"
    return "outdoor"


def _bucket_from_text(seed_text: str, scene_hint: str | None = None) -> dict[str, str]:
    attrs = parse_attributes(seed_text)
    return {
        "color": attrs.colors[0] if attrs.colors else "unknown",
        "clothing": attrs.clothing[0] if attrs.clothing else "unknown",
        "scene": scene_hint or (attrs.scenes[0] if attrs.scenes else "unknown"),
    }


def sample_cuhk_pedes(
    n: int,
    images_dir: Path,
    seed: int = 42,
    image_size: int = 512,
    max_scan: int = 20000,
) -> list[dict[str, Any]]:
    """Stream CUHK-PEDES; prefer captions with color/clothing keywords."""
    from datasets import load_dataset

    random.seed(seed)
    ds = load_dataset("PeterPanTheGenius/CUHK-PEDES", split="train", streaming=True)
    pool: list[dict[str, Any]] = []
    scanned = 0

    for row in ds:
        scanned += 1
        text = (row.get("text") or "").strip()
        if len(text) < 20:
            continue
        attrs = parse_attributes(text)
        # Prefer attribute-rich captions
        score = len(attrs.colors) + len(attrs.clothing)
        if score == 0 and random.random() > 0.05:
            continue

        image_id = f"pedes_{scanned:06d}"
        fname = f"{image_id}.jpg"
        _save_image(row["image"], images_dir / fname, image_size)
        scene = attrs.scenes[0] if attrs.scenes else "street"
        pool.append(
            {
                "image_id": image_id,
                "path": str(Path("images") / fname),
                "source": "cuhk_pedes",
                "seed_labels": {
                    "categories": attrs.clothing,
                    "attributes": attrs.colors,
                },
                "seed_text": text,
                "buckets": _bucket_from_text(text, scene_hint=scene),
                "_score": score,
            }
        )
        if len(pool) >= n * 3 or scanned >= max_scan:
            break

    pool.sort(key=lambda r: r["_score"], reverse=True)
    selected = pool[:n]
    # diversify a bit
    random.shuffle(selected)
    selected = selected[:n]
    for r in selected:
        r.pop("_score", None)
    print(f"CUHK-PEDES: kept {len(selected)} / scanned {scanned}")
    return selected


def sample_coco_people_scenes(
    n: int,
    images_dir: Path,
    seed: int = 42,
    image_size: int = 512,
    max_scan: int = 15000,
) -> list[dict[str, Any]]:
    """Stream COCO; keep images with a person + scene cue objects."""
    from datasets import load_dataset

    random.seed(seed + 7)
    ds = load_dataset("detection-datasets/coco", split="train", streaming=True)

    # Balance across target scenes
    quotas = {
        "park": n // 4,
        "street": n // 4,
        "home": n // 4,
        "office": n - 3 * (n // 4),
    }
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in quotas}
    scanned = 0

    for row in ds:
        scanned += 1
        objects = row.get("objects") or {}
        cat_ids = objects.get("category") or []
        names: set[str] = set()
        has_person = False
        for c in cat_ids:
            if isinstance(c, str) and not c.isdigit():
                name = c.lower()
            else:
                cid = int(c)
                name = COCO_CATS.get(cid, str(cid))
                if cid == PERSON_ID:
                    has_person = True
            if name == "person":
                has_person = True
            names.add(name)
        if not has_person:
            continue

        scene = _infer_scene_from_coco(names)
        if scene not in quotas:
            continue
        if len(buckets[scene]) >= quotas[scene]:
            if all(len(buckets[s]) >= quotas[s] for s in quotas):
                break
            continue

        # Build seed text from objects (helps attribute parse + Florence merge)
        interesting = sorted(n for n in names if n != "person")
        seed_text = "a person with " + ", ".join(interesting[:12]) if interesting else "a person"
        # Soft clothing priors from COCO accessories
        if "tie" in names:
            seed_text += ", wearing a tie"
        if "umbrella" in names:
            seed_text += ", with an umbrella"
        if "backpack" in names:
            seed_text += ", casual backpack"

        image_id = f"coco_{row.get('image_id', scanned)}"
        fname = f"{image_id}.jpg"
        _save_image(row["image"], images_dir / fname, image_size)
        buckets[scene].append(
            {
                "image_id": str(image_id),
                "path": str(Path("images") / fname),
                "source": "coco",
                "seed_labels": {
                    "categories": interesting[:16],
                    "attributes": [],
                },
                "seed_text": seed_text,
                "buckets": _bucket_from_text(seed_text, scene_hint=scene),
            }
        )
        if scanned >= max_scan:
            break

    selected: list[dict[str, Any]] = []
    for scene, rows in buckets.items():
        selected.extend(rows)
    random.shuffle(selected)
    print(
        "COCO people/scenes:",
        {k: len(v) for k, v in buckets.items()},
        f"scanned={scanned}",
    )
    return selected[:n]


def prepare_mixed_corpus(
    output_dir: str | Path,
    pedes_n: int = 500,
    coco_n: int = 300,
    image_size: int = 512,
    seed: int = 42,
) -> Path:
    """Write mixed PEDES+COCO corpus to data/images + metadata.json."""
    output_dir = resolve_path(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    # Fresh corpus directory contents for this prep
    for p in images_dir.glob("*.jpg"):
        p.unlink()

    pedes = sample_cuhk_pedes(pedes_n, images_dir, seed=seed, image_size=image_size)
    coco = sample_coco_people_scenes(coco_n, images_dir, seed=seed, image_size=image_size)
    items = pedes + coco
    random.seed(seed)
    random.shuffle(items)

    meta = {
        "dataset": "mixed-cuhk_pedes-coco",
        "count": len(items),
        "image_size": image_size,
        "seed": seed,
        "sources": {
            "cuhk_pedes": sum(1 for i in items if i["source"] == "cuhk_pedes"),
            "coco": sum(1 for i in items if i["source"] == "coco"),
        },
        "clothing_histogram": dict(Counter(i["buckets"]["clothing"] for i in items)),
        "color_histogram": dict(Counter(i["buckets"]["color"] for i in items)),
        "scene_histogram": dict(Counter(i["buckets"]["scene"] for i in items)),
        "items": items,
    }
    path = output_dir / "metadata.json"
    write_json(path, meta)
    print(f"Mixed corpus → {path} ({meta['count']} images)")
    print("sources:", meta["sources"])
    print("scenes:", meta["scene_histogram"])
    print("colors:", meta["color_histogram"])
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prepare mixed PEDES+COCO fashion/context corpus")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--pedes-n", type=int, default=500)
    p.add_argument("--coco-n", type=int, default=300)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_mixed_corpus(
        output_dir=args.output_dir,
        pedes_n=args.pedes_n,
        coco_n=args.coco_n,
        image_size=args.image_size,
        seed=args.seed,
    )

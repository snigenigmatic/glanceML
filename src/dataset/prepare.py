"""Dataset preparation: sample Fashionpedia into a balanced local corpus.

Produces:
  data/images/*.jpg
  data/metadata.json
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.attributes import COLORS, parse_attributes
from src.utils import ensure_dir, resolve_path, write_json


# Fashionpedia category id → name (subset of apparel; ids from official ontology)
# Used when HF objects.category is numeric.
FASHIONPEDIA_CATEGORIES = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
    26: "umbrella",
}


def _category_names(objects: dict[str, Any]) -> list[str]:
    cats = objects.get("category") or objects.get("categories") or []
    names: list[str] = []
    for c in cats:
        if isinstance(c, str):
            names.append(c.lower())
        elif isinstance(c, int):
            names.append(FASHIONPEDIA_CATEGORIES.get(c, str(c)).lower())
    return names


def _attribute_names(objects: dict[str, Any]) -> list[str]:
    attrs = objects.get("attribute") or objects.get("attributes") or []
    out: list[str] = []
    for a in attrs:
        if isinstance(a, (list, tuple)):
            out.extend(str(x).lower() for x in a)
        else:
            out.append(str(a).lower())
    return out


def _infer_buckets(seed_text: str) -> dict[str, str]:
    attrs = parse_attributes(seed_text)
    color = attrs.colors[0] if attrs.colors else "unknown"
    clothing = attrs.clothing[0] if attrs.clothing else "unknown"
    # Fashionpedia is mostly outdoor/street/event; assign soft environment priors
    # so the corpus spans the three required axes. Florence-2 later refines scene.
    if clothing in {"blazer", "suit", "tie", "shirt"} or "formal" in attrs.styles:
        scene = "office"
    elif clothing in {"hoodie", "t-shirt", "jeans", "sneakers"}:
        scene = random.choice(["street", "park", "home"])
    elif clothing in {"coat", "raincoat", "jacket"}:
        scene = random.choice(["street", "park", "urban"])
    else:
        scene = random.choice(["street", "park", "home", "office"])
    return {"color": color, "clothing": clothing, "scene": scene}


def prepare_fashionpedia_subset(
    output_dir: str | Path,
    target_size: int = 800,
    image_size: int = 512,
    seed: int = 42,
    hf_dataset: str = "detection-datasets/fashionpedia",
    hf_split: str = "train",
) -> Path:
    """Download / stream Fashionpedia and write a local image + metadata corpus."""
    from datasets import load_dataset

    random.seed(seed)
    output_dir = resolve_path(output_dir)
    images_dir = ensure_dir(output_dir / "images")

    print(f"Loading {hf_dataset} ({hf_split}) …")
    ds = load_dataset(hf_dataset, split=hf_split, streaming=True)

    # Over-sample then balance across coarse clothing buckets
    bucket_quota = max(1, target_size // 8)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    max_scan = max(target_size * 20, 5000)

    for row in ds:
        scanned += 1
        objects = row.get("objects") or {}
        cat_names = _category_names(objects)
        attr_names = _attribute_names(objects)
        seed_text = " ".join(cat_names + attr_names)
        if not seed_text.strip():
            continue

        # Prefer images that mention at least one clothing or color keyword
        parsed = parse_attributes(seed_text)
        if not parsed.clothing and not any(c in seed_text for c in COLORS):
            # still keep some diversity
            if random.random() > 0.15:
                continue

        clothing_key = parsed.clothing[0] if parsed.clothing else "other"
        if len(buckets[clothing_key]) >= bucket_quota and clothing_key != "other":
            if sum(len(v) for v in buckets.values()) >= target_size * 2:
                break
            continue

        image: Image.Image = row["image"].convert("RGB")
        image_id = str(row.get("image_id", scanned))
        fname = f"{image_id}.jpg"
        out_path = images_dir / fname
        image.resize((image_size, image_size), Image.Resampling.LANCZOS).save(
            out_path, quality=90
        )

        buckets_info = _infer_buckets(seed_text)
        record = {
            "image_id": image_id,
            "path": str(Path("images") / fname),
            "seed_labels": {
                "categories": cat_names,
                "attributes": attr_names[:32],
            },
            "seed_text": seed_text,
            "buckets": buckets_info,
        }
        buckets[clothing_key].append(record)

        if sum(len(v) for v in buckets.values()) >= target_size * 2:
            break
        if scanned >= max_scan:
            break

    # Round-robin sample to target_size
    selected: list[dict[str, Any]] = []
    keys = list(buckets.keys())
    random.shuffle(keys)
    while len(selected) < target_size and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(selected) < target_size:
                selected.append(buckets[k].pop())

    meta = {
        "dataset": "fashionpedia-subset",
        "count": len(selected),
        "image_size": image_size,
        "seed": seed,
        "clothing_histogram": dict(Counter(r["buckets"]["clothing"] for r in selected)),
        "color_histogram": dict(Counter(r["buckets"]["color"] for r in selected)),
        "scene_histogram": dict(Counter(r["buckets"]["scene"] for r in selected)),
        "items": selected,
    }
    meta_path = output_dir / "metadata.json"
    write_json(meta_path, meta)
    print(f"Wrote {len(selected)} images → {images_dir}")
    print(f"Metadata → {meta_path}")
    print("Scene histogram:", meta["scene_histogram"])
    print("Clothing histogram:", meta["clothing_histogram"])
    return meta_path


def prepare_demo_corpus(
    output_dir: str | Path,
    target_size: int = 32,
    image_size: int = 256,
    seed: int = 42,
) -> Path:
    """Tiny synthetic corpus for CPU smoke tests (no HF download)."""
    random.seed(seed)
    output_dir = resolve_path(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    scenes = ["office", "park", "street", "home"]
    clothes = ["blazer", "hoodie", "raincoat", "shirt", "dress", "jeans"]
    colors = ["red", "blue", "yellow", "white", "black", "green"]
    items = []
    for i in range(target_size):
        color = colors[i % len(colors)]
        clothing = clothes[i % len(clothes)]
        scene = scenes[i % len(scenes)]
        img = Image.new(
            "RGB",
            (image_size, image_size),
            color=_rgb_for_name(color),
        )
        fname = f"demo_{i:04d}.jpg"
        img.save(images_dir / fname)
        seed_text = f"a person wearing a {color} {clothing} in a {scene}"
        items.append(
            {
                "image_id": f"demo_{i:04d}",
                "path": str(Path("images") / fname),
                "seed_labels": {"categories": [clothing], "attributes": [color]},
                "seed_text": seed_text,
                "buckets": {"color": color, "clothing": clothing, "scene": scene},
            }
        )
    meta = {
        "dataset": "demo-synthetic",
        "count": len(items),
        "image_size": image_size,
        "seed": seed,
        "items": items,
    }
    path = output_dir / "metadata.json"
    write_json(path, meta)
    return path


def _rgb_for_name(name: str) -> tuple[int, int, int]:
    table = {
        "red": (220, 40, 40),
        "blue": (40, 80, 200),
        "yellow": (240, 210, 40),
        "white": (240, 240, 240),
        "black": (20, 20, 20),
        "green": (40, 160, 70),
    }
    return table.get(name, (128, 128, 128))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fashion retrieval dataset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Tiny synthetic corpus")
    args = parser.parse_args()

    if args.demo:
        prepare_demo_corpus(args.output_dir, target_size=min(32, args.target_size))
    else:
        prepare_fashionpedia_subset(
            output_dir=args.output_dir,
            target_size=args.target_size,
            image_size=args.image_size,
            seed=args.seed,
        )

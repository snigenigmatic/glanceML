"""Download Fashionpedia images and build a stratified sample for indexing."""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import requests
from PIL import Image
from tqdm import tqdm

from src.config import load_config, resolve_path


SCENE_KEYWORDS = {
    "office": ["office", "desk", "meeting", "corporate", "workplace", "boardroom"],
    "urban": ["street", "city", "urban", "sidewalk", "downtown", "avenue"],
    "park": ["park", "bench", "garden", "outdoor", "grass", "tree"],
    "home": ["home", "living room", "bedroom", "indoor", "couch", "kitchen"],
}

STYLE_KEYWORDS = {
    "formal": ["suit", "blazer", "tie", "dress shirt", "formal", "business"],
    "casual": ["t-shirt", "hoodie", "jeans", "sneakers", "casual", "weekend"],
    "outerwear": ["coat", "jacket", "raincoat", "parka", "outerwear"],
}

COLOR_WORDS = [
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "pink",
    "purple",
    "orange",
    "navy",
    "beige",
]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.touch()


def _infer_scene_from_url(url: str) -> str:
    lowered = (url or "").lower()
    for scene, keywords in SCENE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return scene
    return "urban"


def _infer_style_from_categories(categories: list[str]) -> str:
    joined = " ".join(categories).lower()
    if any(word in joined for word in STYLE_KEYWORDS["formal"]):
        return "formal"
    if any(word in joined for word in STYLE_KEYWORDS["outerwear"]):
        return "outerwear"
    return "casual"


def _infer_colors_from_attributes(attribute_names: list[str]) -> list[str]:
    colors: list[str] = []
    for attribute in attribute_names:
        lowered = attribute.lower()
        for color in COLOR_WORDS:
            if color in lowered and color not in colors:
                colors.append(color)
    return colors


def _build_image_records(
    annotations: dict[str, Any],
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    category_by_id = {item["id"]: item["name"] for item in annotations["categories"]}
    attribute_by_id = {item["id"]: item["name"] for item in annotations["attributes"]}

    per_image_categories: dict[int, list[str]] = defaultdict(list)
    per_image_attributes: dict[int, list[str]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        category = category_by_id.get(annotation["category_id"])
        if category:
            per_image_categories[image_id].append(category)
        for attribute_id in annotation.get("attribute_ids", []):
            attribute = attribute_by_id.get(attribute_id)
            if attribute:
                per_image_attributes[image_id].append(attribute)

    candidates: list[dict[str, Any]] = []
    for image in annotations["images"]:
        source_path = Path(image["resolved_path"])
        if not source_path.exists():
            continue

        categories = sorted(set(per_image_categories[image["id"]]))
        attributes = sorted(set(per_image_attributes[image["id"]]))
        colors = _infer_colors_from_attributes(attributes)
        scene = _infer_scene_from_url(image.get("original_url", ""))
        style = _infer_style_from_categories(categories)

        candidates.append(
            {
                "image_id": image["id"],
                "source_path": str(source_path),
                "file_name": image["file_name"],
                "categories": categories,
                "attributes": attributes,
                "colors": colors,
                "scene": scene,
                "style": style,
                "original_url": image.get("original_url", ""),
            }
        )

    random.seed(seed)
    random.shuffle(candidates)

    # Stratify across scene buckets to satisfy environment diversity requirement.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        buckets[record["scene"]].append(record)

    selected: list[dict[str, Any]] = []
    scenes = list(SCENE_KEYWORDS.keys())
    while len(selected) < target_count and any(buckets[scene] for scene in scenes):
        for scene in scenes:
            if buckets[scene] and len(selected) < target_count:
                selected.append(buckets[scene].pop())

    if len(selected) < target_count:
        remaining = [record for scene in scenes for record in buckets[scene]]
        random.shuffle(remaining)
        selected.extend(remaining[: target_count - len(selected)])

    return selected[:target_count]


def _materialize_sample(
    records: list[dict[str, Any]],
    images_dir: Path,
    image_size: int,
) -> list[dict[str, Any]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_records: list[dict[str, Any]] = []

    for index, record in enumerate(tqdm(records, desc="Preparing images")):
        image_id = f"img_{index:05d}"
        destination = images_dir / f"{image_id}.jpg"
        with Image.open(record["source_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((image_size * 2, image_size * 2))
            image.save(destination, format="JPEG", quality=92)

        metadata_records.append(
            {
                "id": image_id,
                "file_name": destination.name,
                "image_path": str(destination),
                "fashionpedia_image_id": record["image_id"],
                "categories": record["categories"],
                "attributes": record["attributes"],
                "colors": record["colors"],
                "scene": record["scene"],
                "style": record["style"],
                "clothing": record["categories"],
                "caption_seed": (
                    f"A {record['style']} outfit with "
                    f"{', '.join(record['categories'][:3]) or 'apparel'} "
                    f"in a {record['scene']} setting."
                ),
            }
        )

    return metadata_records


def download_and_sample_dataset(
    config_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download Fashionpedia assets and write a sampled metadata.json file."""
    config = load_config(config_path)
    data_dir = resolve_path(config, "data_dir")
    images_dir = resolve_path(config, "images_dir")
    metadata_file = resolve_path(config, "metadata_file")

    if metadata_file.exists() and not force:
        return metadata_file

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_zip = raw_dir / "train2020.zip"
    val_zip = raw_dir / "val_test2020.zip"
    train_json = raw_dir / "instances_attributes_train2020.json"
    val_json = raw_dir / "instances_attributes_val2020.json"

    _download_file(config["dataset"]["train_images_url"], train_zip)
    _download_file(config["dataset"]["val_test_images_url"], val_zip)
    _download_file(config["dataset"]["train_annotations_url"], train_json)
    _download_file(config["dataset"]["val_annotations_url"], val_json)

    train_extract = raw_dir / "train2020"
    val_extract = raw_dir / "val_test2020"
    _extract_zip(train_zip, train_extract)
    _extract_zip(val_zip, val_extract)

    with train_json.open("r", encoding="utf-8") as handle:
        train_annotations = json.load(handle)
    with val_json.open("r", encoding="utf-8") as handle:
        val_annotations = json.load(handle)

    # Merge annotations for broader coverage.
    merged = {
        "categories": train_annotations["categories"],
        "attributes": train_annotations["attributes"],
        "images": train_annotations["images"] + val_annotations["images"],
        "annotations": train_annotations["annotations"] + val_annotations["annotations"],
    }

  # Map Fashionpedia file names to extracted paths.
    image_path_by_name: dict[str, Path] = {}
    for root in (train_extract, val_extract):
        if not root.exists():
            continue
        for image_path in root.rglob("*.jpg"):
            image_path_by_name.setdefault(image_path.name, image_path)

    resolved_images: list[dict[str, Any]] = []
    for image in merged["images"]:
        resolved = image_path_by_name.get(Path(image["file_name"]).name)
        if resolved is None:
            continue
        image_copy = dict(image)
        image_copy["resolved_path"] = str(resolved)
        resolved_images.append(image_copy)
    merged["images"] = resolved_images

    target_count = int(config["dataset"]["num_images"])
    seed = int(config["dataset"]["seed"])
    image_size = int(config["dataset"]["image_size"])

    records = _build_image_records(
        annotations=merged,
        target_count=target_count,
        seed=seed,
    )

    if len(records) < int(config["dataset"]["min_images"]):
        raise RuntimeError(
            f"Only found {len(records)} valid images; need at least "
            f"{config['dataset']['min_images']}."
        )

    metadata_records = _materialize_sample(records, images_dir, image_size)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata_records, handle, indent=2)

    return metadata_file


def cleanup_raw_downloads(config_path: str | Path | None = None) -> None:
    """Remove large raw archives after sampling to save Modal volume space."""
    config = load_config(config_path)
    raw_dir = resolve_path(config, "data_dir") / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

"""Configuration loader for the fashion retrieval pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("FASHION_CONFIG_PATH", ROOT_DIR / "configs" / "default.yaml")
)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Allow Modal volume overrides via environment variables.
    paths = config.setdefault("paths", {})
    paths["data_dir"] = os.environ.get("FASHION_DATA_DIR", paths.get("data_dir", "data"))
    paths["images_dir"] = os.environ.get(
        "FASHION_IMAGES_DIR", paths.get("images_dir", "data/images")
    )
    paths["metadata_file"] = os.environ.get(
        "FASHION_METADATA_FILE", paths.get("metadata_file", "data/metadata.json")
    )
    paths["index_dir"] = os.environ.get(
        "FASHION_INDEX_DIR", paths.get("index_dir", "data/index")
    )
    paths["chroma_dir"] = os.environ.get(
        "FASHION_CHROMA_DIR", paths.get("chroma_dir", "data/index/chroma")
    )
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    return (ROOT_DIR / config["paths"][key]).resolve()

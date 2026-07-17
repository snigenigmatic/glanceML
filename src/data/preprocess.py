"""Metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metadata(metadata_file: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_file)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(metadata_file: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(metadata_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

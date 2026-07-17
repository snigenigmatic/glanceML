"""Shared retrieval result types (no heavy ML imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    image_id: str
    path: str
    score: float
    dense_score: float
    metadata_score: float
    caption: str
    colors: list[str]
    clothing: list[str]
    scenes: list[str]
    styles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": self.path,
            "score": self.score,
            "dense_score": self.dense_score,
            "metadata_score": self.metadata_score,
            "caption": self.caption,
            "matched_attributes": {
                "colors": self.colors,
                "clothing": self.clothing,
                "scenes": self.scenes,
                "styles": self.styles,
            },
        }

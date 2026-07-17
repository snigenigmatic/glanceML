"""Embedding model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image


@dataclass
class EmbeddingModel:
    name: str
    model: torch.nn.Module
    processor: object
    device: str

    @classmethod
    def from_pretrained(cls, model_name: str, device: str | None = None) -> "EmbeddingModel":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "marqo-fashionSigLIP" in model_name.lower():
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        else:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)

        model.eval()
        model.to(device)
        return cls(name=model_name, model=model, processor=processor, device=device)

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        image_list = list(images)
        if not image_list:
            return torch.empty(0, 0)

        if hasattr(self.model, "get_image_features"):
            processed = self.processor(
                images=image_list,
                return_tensors="pt",
                padding=True,
            )
            pixel_values = processed["pixel_values"].to(self.device)
            features = self.model.get_image_features(pixel_values, normalize=True)
            return features

        processed = self.processor(images=image_list, return_tensors="pt", padding=True)
        pixel_values = processed["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.inference_mode()
    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        text_list = list(texts)
        if not text_list:
            return torch.empty(0, 0)

        if hasattr(self.model, "get_text_features"):
            processed = self.processor(
                text=text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = processed["input_ids"].to(self.device)
            features = self.model.get_text_features(input_ids, normalize=True)
            return features

        processed = self.processor(
            text=text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = processed["input_ids"].to(self.device)
        attention_mask = processed["attention_mask"].to(self.device)
        features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return torch.nn.functional.normalize(features, dim=-1)

    def cosine_similarity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim == 1:
            left = left.unsqueeze(0)
        if right.ndim == 1:
            right = right.unsqueeze(0)
        return left @ right.T

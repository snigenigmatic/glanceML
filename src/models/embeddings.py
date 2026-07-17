"""Embedding models: FashionSigLIP (primary) and CLIP (baseline)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image


class EmbeddingModel:
    """Thin wrapper around HF multimodal embedding models."""

    def __init__(self, model_id: str, device: str | None = None):
        import logging
        import warnings

        from transformers import AutoModel, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        # FashionSigLIP wraps open_clip; HF/open_clip emit harmless load-time warnings.
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*model of type siglip.*")
            warnings.filterwarnings("ignore", message=".*weights_only=False.*")
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = list(images[i : i + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
            feats = self._image_features(inputs)
            vectors.append(feats.cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)

    @torch.inference_mode()
    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
            feats = self._text_features(inputs)
            vectors.append(feats.cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def _image_features(self, inputs: dict) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            pixel_values = inputs["pixel_values"]
            try:
                feats = self.model.get_image_features(pixel_values, normalize=True)
            except TypeError:
                feats = self.model.get_image_features(pixel_values=pixel_values)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats
        outputs = self.model(**inputs)
        feats = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
        return torch.nn.functional.normalize(feats, dim=-1)

    def _text_features(self, inputs: dict) -> torch.Tensor:
        if hasattr(self.model, "get_text_features"):
            try:
                feats = self.model.get_text_features(inputs["input_ids"], normalize=True)
            except TypeError:
                feats = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats
        outputs = self.model(**inputs)
        feats = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0]
        return torch.nn.functional.normalize(feats, dim=-1)


def load_fashion_siglip(model_id: str = "Marqo/marqo-fashionSigLIP", device: str | None = None) -> EmbeddingModel:
    return EmbeddingModel(model_id=model_id, device=device)


def load_clip_baseline(model_id: str = "openai/clip-vit-base-patch32", device: str | None = None) -> EmbeddingModel:
    return EmbeddingModel(model_id=model_id, device=device)

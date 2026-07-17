"""Vision-language metadata extraction with Florence-2."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from PIL import Image

from src.retriever.query_parser import COLOR_WORDS, CLOTHING_WORDS, SCENE_WORDS, STYLE_WORDS


@dataclass
class VLMMetadataExtractor:
    model_name: str
    task: str
    model: object
    processor: object
    device: str

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "microsoft/Florence-2-base",
        task: str = "<MORE_DETAILED_CAPTION>",
        device: str | None = None,
    ) -> "VLMMetadataExtractor":
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model.eval()
        model.to(device)
        return cls(
            model_name=model_name,
            task=task,
            model=model,
            processor=processor,
            device=device,
        )

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(text=self.task, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=(image.width, image.height),
        )
        return parsed[self.task]

    def extract_metadata(self, image: Image.Image, seed: dict | None = None) -> dict:
        caption = self.generate_caption(image)
        metadata = _parse_caption(caption)
        if seed:
            metadata = _merge_seed_metadata(metadata, seed)
        metadata["caption"] = caption
        return metadata


def _token_hits(text: str, vocabulary: set[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for token in vocabulary:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            hits.append(token)
    return sorted(set(hits))


def _parse_caption(caption: str) -> dict:
    return {
        "clothing": _token_hits(caption, CLOTHING_WORDS),
        "colors": _token_hits(caption, COLOR_WORDS),
        "scene": _token_hits(caption, SCENE_WORDS)[:1] or ["unknown"],
        "style": _token_hits(caption, STYLE_WORDS)[:1] or ["unknown"],
    }


def _merge_seed_metadata(metadata: dict, seed: dict) -> dict:
    merged = dict(metadata)
    for field in ("clothing", "colors", "scene", "style"):
        seed_values = seed.get(field)
        if not seed_values:
            continue
        if isinstance(seed_values, str):
            seed_values = [seed_values]
        existing = set(merged.get(field, []))
        merged[field] = sorted(existing.union(seed_values))
    return merged

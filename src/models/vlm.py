"""Florence-2 VLM for structured fashion/scene metadata extraction."""

from __future__ import annotations

import re

import torch
from PIL import Image

from src.attributes import FashionAttributes, parse_caption_attributes


class Florence2Extractor:
    """Uses Florence-2 detailed captions, then parses fashion attributes.

    Caption-only parsing (no open-vocabulary detection dump). Open-vocab
    detection previously tagged ~16 garments on every image and destroyed
    metadata discriminativeness.
    """

    TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def _run_task(self, image: Image.Image, task: str) -> str:
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        if "pixel_values" in inputs and self.device.startswith("cuda"):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            decoded,
            task=task,
            image_size=(image.width, image.height),
        )
        result = parsed.get(task, parsed)
        if isinstance(result, dict):
            return str(result)
        return str(result)

    def extract(self, image: Image.Image, seed_text: str = "") -> FashionAttributes:
        image = image.convert("RGB")
        caption = _clean_caption(self._run_task(image, self.TASK))
        attrs = parse_caption_attributes(caption, seed_text=seed_text)
        attrs.caption = caption
        return attrs


def _clean_caption(text: str) -> str:
    text = re.sub(r"</?s>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HeuristicExtractor:
    """CPU-only fallback when Florence-2 weights are unavailable."""

    def extract(self, image: Image.Image, seed_text: str = "") -> FashionAttributes:
        return parse_caption_attributes(seed_text, seed_text=seed_text)

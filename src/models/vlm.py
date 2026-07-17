"""Florence-2 VLM for structured fashion/scene metadata extraction."""

from __future__ import annotations

import re

import torch
from PIL import Image

from src.attributes import FashionAttributes, parse_attributes


class Florence2Extractor:
    """Uses Florence-2 detailed captions, then parses fashion attributes.

    Florence-2 is task-token based (not free-form chat). We run
    MORE_DETAILED_CAPTION and optionally OPEN_VOCABULARY_DETECTION for
    garment phrases, then map text → structured attributes.
    """

    TASK = "<MORE_DETAILED_CAPTION>"
    DETECT_TASK = "<OPEN_VOCABULARY_DETECTION>"
    DETECT_PROMPTS = "shirt, t-shirt, blouse, dress, jacket, coat, raincoat, blazer, suit, hoodie, jeans, pants, skirt, tie, sneakers, boots"

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
    def _run_task(self, image: Image.Image, task: str, text_input: str | None = None) -> str:
        prompt = task if text_input is None else task + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
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
            # open-vocab detection returns labels/bboxes
            labels = result.get("labels") or result.get("bboxes_labels") or []
            if labels:
                return ", ".join(str(x) for x in labels)
            return str(result)
        return str(result)

    def extract(self, image: Image.Image) -> FashionAttributes:
        image = image.convert("RGB")
        caption = self._run_task(image, self.TASK)
        detect_text = ""
        try:
            detect_text = self._run_task(image, self.DETECT_TASK, self.DETECT_PROMPTS)
        except Exception:
            detect_text = ""

        combined = f"{caption}. {detect_text}".strip()
        attrs = parse_attributes(combined)
        attrs.caption = _clean_caption(caption)
        return attrs


def _clean_caption(text: str) -> str:
    text = re.sub(r"</?s>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class HeuristicExtractor:
    """CPU-only fallback when Florence-2 weights are unavailable."""

    def extract(self, image: Image.Image, seed_text: str = "") -> FashionAttributes:
        return parse_attributes(seed_text)

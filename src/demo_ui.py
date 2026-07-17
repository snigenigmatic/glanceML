"""Gradio demo for fashion retrieval (used locally and on Modal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from src.retriever.hybrid import FashionRetriever


EXAMPLE_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def build_demo(retriever: FashionRetriever, data_root: Path) -> gr.Blocks:
    def run(query: str, top_k: int) -> tuple[list[tuple[Any, str]], str]:
        if not query.strip():
            return [], "Enter a natural-language fashion query."
        results = retriever.search(query, top_k=int(top_k))
        gallery = []
        lines = []
        for i, r in enumerate(results, 1):
            img_path = data_root / r.path
            caption = (
                f"#{i} score={r.score:.3f} | dense={r.dense_score:.3f} | "
                f"meta={r.metadata_score:.3f}\n"
                f"colors={r.colors} clothing={r.clothing} "
                f"scenes={r.scenes} styles={r.styles}\n"
                f"{r.caption[:180]}"
            )
            if img_path.exists():
                gallery.append((Image.open(img_path), caption))
            lines.append(caption)
        return gallery, "\n\n".join(lines)

    with gr.Blocks(title="Glance Fashion Retrieval") as demo:
        gr.Markdown(
            "## Multimodal Fashion & Context Retrieval\n"
            "FashionSigLIP dense retrieval + Florence-2 metadata + hybrid fusion."
        )
        with gr.Row():
            query = gr.Textbox(label="Query", lines=2, placeholder="Describe outfit + place + vibe…")
            top_k = gr.Slider(1, 12, value=5, step=1, label="Top-K")
        btn = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(
            label="Results",
            columns=3,
            height=480,
            object_fit="contain",
            preview=True,
        )
        details = gr.Textbox(label="Scores & matched attributes", lines=16)
        gr.Examples(EXAMPLE_QUERIES, inputs=[query])
        btn.click(run, inputs=[query, top_k], outputs=[gallery, details])
        query.submit(run, inputs=[query, top_k], outputs=[gallery, details])
    return demo

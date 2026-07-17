"""Modal deployment for multimodal fashion retrieval."""

from __future__ import annotations

import json
import os

import modal

APP_NAME = "glance-fashion-retrieval"
DATA_MOUNT = "/data"
CONFIG_PATH = "/root/configs/default.yaml"

DATA_VOLUME = modal.Volume.from_name("fashion-retrieval-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App(APP_NAME, image=image)


def _configure_paths() -> None:
    os.environ["FASHION_CONFIG_PATH"] = CONFIG_PATH
    os.environ["FASHION_DATA_DIR"] = DATA_MOUNT
    os.environ["FASHION_IMAGES_DIR"] = f"{DATA_MOUNT}/images"
    os.environ["FASHION_METADATA_FILE"] = f"{DATA_MOUNT}/metadata.json"
    os.environ["FASHION_INDEX_DIR"] = f"{DATA_MOUNT}/index"
    os.environ["FASHION_CHROMA_DIR"] = f"{DATA_MOUNT}/index/chroma"


@app.function(
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 60,
    memory=8192,
)
def prepare_dataset(force: bool = False) -> dict:
    """Download Fashionpedia and create a 500-1000 image sample."""
    _configure_paths()
    from src.data.download import cleanup_raw_downloads, download_and_sample_dataset

    metadata_file = download_and_sample_dataset(force=force)
    cleanup_raw_downloads()
    DATA_VOLUME.commit()
    return {"metadata_file": str(metadata_file), "status": "ready"}


@app.function(
    gpu="T4",
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 60 * 2,
    memory=16384,
)
def build_search_index(force: bool = False, use_vlm: bool = True) -> dict:
    """Part A: extract embeddings + VLM metadata and persist to ChromaDB."""
    _configure_paths()
    from src.indexer.indexer import build_index

    result = build_index(force=force, use_vlm=use_vlm)
    DATA_VOLUME.commit()
    return result


@app.function(
    gpu="T4",
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 30,
    memory=16384,
)
def query_images(query: str, top_k: int = 5, use_reranker: bool = True) -> list[dict]:
    """Part B: run hybrid retrieval for a natural-language query."""
    _configure_paths()
    from src.retriever.search import search

    results = search(query=query, top_k=top_k, use_reranker=use_reranker)
    serializable = []
    for item in results:
        serializable.append(
            {
                "id": item["id"],
                "image_path": item["image_path"],
                "score": item.get("rerank_score", item.get("score")),
                "semantic_score": item.get("semantic_score"),
                "metadata_score": item.get("metadata_score"),
                "matched_attributes": item.get("matched_attributes", {}),
                "caption": item.get("caption", ""),
            }
        )
    return serializable


@app.function(
    gpu="T4",
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 60 * 3,
    memory=16384,
)
def run_evaluation() -> dict:
    """Evaluate assignment queries across semantic, metadata, hybrid, and CLIP baselines."""
    _configure_paths()
    from src.eval.evaluate import evaluate_retrieval

    report = evaluate_retrieval()
    DATA_VOLUME.commit()
    return report


@app.function(
    gpu="T4",
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 60 * 4,
    memory=16384,
)
def run_full_setup(force: bool = False) -> dict:
    """One-shot setup: dataset -> index -> evaluation."""
    _configure_paths()
    from src.eval.evaluate import run_full_pipeline

    result = run_full_pipeline(force=force)
    DATA_VOLUME.commit()
    return result


@app.function(
    gpu="T4",
    volumes={DATA_MOUNT: DATA_VOLUME},
    timeout=60 * 30,
    memory=16384,
)
@modal.asgi_app()
def web():
    """Gradio demo for interactive fashion retrieval."""
    _configure_paths()
    import gradio as gr
    from fastapi import FastAPI

    from src.retriever.search import search

    fastapi_app = FastAPI()

    def _search(query: str, top_k: int, use_reranker: bool):
        if not query.strip():
            return [], "Enter a query to search."
        results = search(query=query, top_k=int(top_k), use_reranker=use_reranker)
        gallery = []
        lines = []
        for rank, item in enumerate(results, start=1):
            score = item.get("rerank_score", item.get("score"))
            gallery.append((item["image_path"], f"#{rank} score={score:.3f}"))
            lines.append(
                f"#{rank} {item['id']} | score={score:.3f} "
                f"| matched={json.dumps(item.get('matched_attributes', {}))}"
            )
        return gallery, "\n".join(lines)

    demo = gr.Blocks(title="Fashion Retrieval Demo")
    with demo:
        gr.Markdown(
            "# Multimodal Fashion Retrieval\n"
            "FashionSigLIP dense retrieval + Florence-2 metadata + hybrid fusion + compositional reranking."
        )
        with gr.Row():
            query = gr.Textbox(
                label="Query",
                placeholder="A person in a bright yellow raincoat.",
                lines=2,
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            use_reranker = gr.Checkbox(value=True, label="Use compositional reranker")
        search_btn = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Results", columns=3)
        details = gr.Textbox(label="Scores and matched attributes", lines=8)
        search_btn.click(_search, inputs=[query, top_k, use_reranker], outputs=[gallery, details])

        gr.Examples(
            examples=[
                ["A person in a bright yellow raincoat.", 5, True],
                ["Professional business attire inside a modern office.", 5, True],
                ["Someone wearing a blue shirt sitting on a park bench.", 5, True],
                ["Casual weekend outfit for a city walk.", 5, True],
                ["A red tie and a white shirt in a formal setting.", 5, True],
            ],
            inputs=[query, top_k, use_reranker],
        )

    return gr.mount_gradio_app(fastapi_app, demo, path="/")


@app.local_entrypoint()
def main(
    action: str = "setup",
    query: str = "A person in a bright yellow raincoat.",
    force: bool = False,
):
    """
    Modal CLI entrypoint.

    Examples:
      modal run app.py --action setup
      modal run app.py --action query --query "blue shirt in a park"
      modal run app.py --action eval
      modal deploy app.py  # serves Gradio at the `web` endpoint
    """
    if action == "setup":
        print(run_full_setup.remote(force=force))
    elif action == "dataset":
        print(prepare_dataset.remote(force=force))
    elif action == "index":
        print(build_search_index.remote(force=force))
    elif action == "query":
        print(query_images.remote(query=query))
    elif action == "eval":
        print(run_evaluation.remote())
    else:
        raise ValueError(f"Unknown action: {action}")

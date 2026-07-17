"""
Modal entrypoints for the fashion retrieval system.

Setup (once):
  pip install modal
  modal setup

Typical workflow:
  modal run modal_app.py::prepare_dataset
  modal run modal_app.py::build_index
  modal run modal_app.py::evaluate
  modal run modal_app.py::query --query-text "A person in a bright yellow raincoat."
  modal serve modal_app.py   # Gradio demo
  # or: modal deploy modal_app.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "glance-fashion-retrieval"
VOL_NAME = "glance-fashion-data"
DATA_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# GPU image with ML stack. Models download at runtime into HF cache;
# dataset + Chroma artifacts persist on the Modal volume.
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.45.2",
        "accelerate==0.34.2",
        "timm==1.0.9",
        "einops==0.8.0",
        "open_clip_torch==2.26.1",
        "chromadb==0.5.5",
        "datasets==3.0.1",
        "pillow==10.4.0",
        "pyyaml==6.0.2",
        "tqdm==4.66.5",
        "gradio==4.44.0",
        "numpy==1.26.4",
        "sentencepiece==0.2.0",
        "protobuf==4.25.5",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)


def _write_runtime_config(data_root: Path = DATA_ROOT) -> Path:
    """Point chroma/data paths at the Modal volume."""
    import yaml

    cfg_path = Path("/root/configs/default.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["output_dir"] = str(data_root)
    cfg["index"]["chroma_dir"] = str(data_root / "chroma")
    cfg["eval"]["output_dir"] = str(data_root / "eval_results")
    cfg["eval"]["queries_path"] = "/root/configs/eval_queries.json"
    cfg["models"]["device"] = "cuda"
    out = data_root / "runtime_config.yaml"
    data_root.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return out


@app.function(
    image=ml_image,
    volumes={str(DATA_ROOT): volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def prepare_dataset(target_size: int = 800, image_size: int = 512, seed: int = 42):
    """Download Fashionpedia subset onto the Modal volume."""
    import sys

    sys.path.insert(0, "/root")
    from src.dataset.prepare import prepare_fashionpedia_subset

    meta_path = prepare_fashionpedia_subset(
        output_dir=DATA_ROOT,
        target_size=target_size,
        image_size=image_size,
        seed=seed,
    )
    volume.commit()
    return str(meta_path)


@app.function(
    image=ml_image,
    volumes={str(DATA_ROOT): volume},
    gpu="A10G",
    timeout=60 * 60 * 3,
    memory=16384,
)
def build_index(use_vlm: bool = True, max_items: int | None = None):
    """Part A: FashionSigLIP embeddings + Florence-2 metadata → ChromaDB."""
    import sys

    sys.path.insert(0, "/root")
    cfg_path = _write_runtime_config()
    from src.indexer.build_index import build_index as _build

    path = _build(
        config_path=str(cfg_path),
        data_dir=str(DATA_ROOT),
        use_vlm=use_vlm,
        max_items=max_items,
    )
    volume.commit()
    return str(path)


@app.function(
    image=ml_image,
    volumes={str(DATA_ROOT): volume},
    gpu="A10G",
    timeout=60 * 15,
    memory=16384,
)
def query(query_text: str, top_k: int = 5) -> list[dict]:
    """Part B: natural-language retrieval over the indexed corpus."""
    import sys

    sys.path.insert(0, "/root")
    cfg_path = _write_runtime_config()
    from src.retriever.hybrid import search_query

    results = search_query(
        query=query_text,
        top_k=top_k,
        config_path=str(cfg_path),
        data_dir=str(DATA_ROOT),
    )
    return results


@app.function(
    image=ml_image,
    volumes={str(DATA_ROOT): volume},
    gpu="A10G",
    timeout=60 * 30,
    memory=16384,
)
def evaluate(top_k: int = 5) -> dict:
    """Run ablation (dense / hybrid / hybrid+rerank) on assignment prompts."""
    import sys

    sys.path.insert(0, "/root")
    cfg_path = _write_runtime_config()
    from src.eval.evaluate import run_evaluation

    report = run_evaluation(
        config_path=str(cfg_path),
        data_dir=str(DATA_ROOT),
        top_k=top_k,
    )
    volume.commit()
    return report["summary"]


@app.function(
    image=ml_image,
    volumes={str(DATA_ROOT): volume},
    gpu="A10G",
    timeout=60 * 60,
    memory=16384,
)
@modal.asgi_app()
def demo():
    """Gradio ASGI app for interactive retrieval."""
    import sys

    sys.path.insert(0, "/root")
    import gradio as gr
    from fastapi import FastAPI

    cfg_path = _write_runtime_config()
    from src.demo_ui import build_demo
    from src.retriever.hybrid import FashionRetriever

    retriever = FashionRetriever(config_path=str(cfg_path), data_dir=str(DATA_ROOT))
    blocks = build_demo(retriever, DATA_ROOT)
    api = FastAPI()
    return gr.mount_gradio_app(api, blocks, path="/")


@app.local_entrypoint()
def main(
    stage: str = "all",
    query_text: str = "A person in a bright yellow raincoat.",
    target_size: int = 800,
    top_k: int = 5,
):
    """
    One-shot orchestration from your laptop:
      modal run modal_app.py --stage all
      modal run modal_app.py --stage query --query-text "Casual weekend outfit for a city walk."
    """
    if stage in {"prepare", "all"}:
        print("→ prepare_dataset")
        print(prepare_dataset.remote(target_size=target_size))
    if stage in {"index", "all"}:
        print("→ build_index")
        print(build_index.remote())
    if stage in {"eval", "all"}:
        print("→ evaluate")
        print(evaluate.remote(top_k=top_k))
    if stage in {"query", "all"}:
        print("→ query:", query_text)
        print(query.remote(query_text=query_text, top_k=top_k))

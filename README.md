# Glance ML — Multimodal Fashion & Context Retrieval

Intelligent image search for fashion queries that combine **what** someone is wearing, **where** they are, and the **vibe** of the outfit.

Built to run on [Modal](https://modal.com) with:

| Piece | Choice | Why |
| --- | --- | --- |
| Embeddings | `Marqo/marqo-fashionSigLIP` | Fashion-tuned SigLIP; stronger garment/attribute recall than vanilla CLIP |
| VLM metadata | `microsoft/Florence-2-base` | Structured scene/clothing/color signals for compositionality |
| Index | ChromaDB (cosine HNSW) | Fast ANN with persisted metadata; swap for Qdrant/FAISS at 1M+ |
| Retrieval | Hybrid fusion + light rerank | Dense similarity + attribute overlap; beats dense-only on multi-attribute prompts |

## Repository layout

```text
configs/          default.yaml + official eval prompts
src/dataset/      Fashionpedia sampling → data/images + metadata.json
src/models/       FashionSigLIP + Florence-2 wrappers
src/indexer/      Part A — embed, extract, upsert Chroma
src/retriever/    Part B — parse query, dense search, fuse, rerank
src/eval/         Ablation: dense / hybrid / hybrid+rerank
modal_app.py      Modal entrypoints (prepare, index, query, eval, Gradio)
REPORT.md         Approaches, decisions, future work (for PDF export)
```

## Run on Modal

```bash
pip install modal
modal setup                          # browser auth once

# Recommended corpus: CUHK-PEDES (clothing captions) + COCO people/scenes
modal run modal_app.py --stage prepare --source mixed --pedes-n 500 --coco-n 300
modal run modal_app.py --stage index
modal run modal_app.py --stage eval_prep   # repair metadata + CLIP baseline
modal run modal_app.py --stage eval

# Other sources: --source cuhk_pedes | fashionpedia
# Full pipeline:
modal run modal_app.py --stage all --source mixed

modal run modal_app.py::query --query-text "A woman in a black leather jacket on the street." --top-k 5

# Interactive Gradio demo
modal serve modal_app.py
modal deploy modal_app.py
```

Data and the Chroma index live on the Modal volume `glance-fashion-data` (`/data`).

**Why mixed?** Fashionpedia alone is runway-skewed (weak office/park/rain queries). PEDES supplies attribute-rich person captions; COCO supplies people in park/street/home/office-like contexts.

The Gradio demo is pinned to **one container** (`max_containers=1`) with concurrent inputs so Gradio’s in-memory queue sessions stay sticky (avoids `404: Session not found`).

### GPU notes

- `prepare_dataset` is CPU-only.
- `build_index`, `query`, `evaluate`, and `demo` use an **A10G**. Change `gpu="A10G"` in `modal_app.py` if your workspace prefers `T4` / `L4`.

## Local (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD

# Tiny synthetic corpus (no HF download) for logic smoke tests
python -m src.dataset.prepare --demo --output-dir data
python -m src.indexer.build_index --data-dir data --no-vlm --max-items 32
python -m src.retriever.hybrid "a red shirt in an office" --data-dir data

pytest tests/
```

Full Fashionpedia + Florence-2 indexing is intended for Modal GPUs.

## Evaluation

Judge prompts (hand-labeled in `configs/relevance_labels.json`):

1. Attribute: *A person in a bright yellow raincoat.*
2. Contextual: *Professional business attire inside a modern office.*
3. Complex: *Someone wearing a blue shirt sitting on a park bench.*
4. Style: *Casual weekend outfit for a city walk.*
5. Compositional: *A red tie and a white shirt in a formal setting.*

Primary metrics are **Precision@K / Recall@K / AP@K** against those labels, comparing:

- vanilla **CLIP baseline**
- FashionSigLIP dense-only
- hybrid
- hybrid + composition rerank

```bash
modal run modal_app.py::repair_metadata      # clean caption→attribute metadata
modal run modal_app.py::build_clip_baseline  # CLIP corpus embeddings
modal run modal_app.py::evaluate
```

Attribute Coverage@K is kept only as a diagnostic (it can self-grade); do not use it as the main claim.

## Design rationale (short)

Vanilla CLIP collapses compositional queries into a bag-of-concepts embedding. This system keeps a strong fashion dense retriever, then **reintroduces explicit attributes** from a VLM so “red tie + white shirt + formal” cannot silently swap colors or drop the setting. Details and alternatives are in [`REPORT.md`](REPORT.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

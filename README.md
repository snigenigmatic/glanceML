# Multimodal Fashion & Context Retrieval

Intelligent fashion image search for the Glance ML internship assignment. The system retrieves images from a Fashionpedia sample using:

- **FashionSigLIP** for fashion-aware dense embeddings
- **Florence-2** (VLM) for structured metadata extraction (scene, clothing, color, style, caption)
- **ChromaDB** for fast vector indexing
- **Hybrid retrieval** (70% semantic + 30% metadata match)
- **Compositional reranking** for multi-attribute queries

Vanilla CLIP is included as a baseline in evaluation.

## Architecture

```text
Images
   │
   ├── FashionSigLIP → Embeddings
   ├── Florence-2 → Structured metadata
   │
   ▼
ChromaDB
   │
Natural Language Query
   │
Query Parsing
   │
Dense Retrieval
   │
Metadata Scoring
   │
Weighted Fusion
   │
Compositional Reranking
   │
Top-K Images
```

See `ARCHITECTURE.md`, `PLAN.md`, and `REPORT_NOTES.md` for design rationale.

## Repository Layout

```text
app.py                  # Modal deployment entrypoint
configs/default.yaml    # Model + retrieval configuration
src/
  data/                 # Fashionpedia download + sampling
  models/               # FashionSigLIP + Florence-2 wrappers
  indexer/              # Part A: build Chroma index
  retriever/            # Part B: hybrid search + rerank
  eval/                 # Evaluation on assignment queries
scripts/local_run.py    # Local CLI (without Modal)
```

## Run on Modal

Prerequisites:

1. [Modal account](https://modal.com/)
2. `pip install modal`
3. `modal token new`

### One-shot setup (dataset + index + evaluation)

```bash
modal run app.py --action setup
```

### Step-by-step

```bash
# 1) Download Fashionpedia and sample 800 images
modal run app.py --action dataset

# 2) Build embeddings + VLM metadata index (GPU)
modal run app.py --action index

# 3) Run a query
modal run app.py --action query --query "A person in a bright yellow raincoat."

# 4) Evaluate assignment queries vs baselines
modal run app.py --action eval
```

### Deploy interactive Gradio demo

```bash
modal deploy app.py
```

Open the `web` endpoint URL printed by Modal.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline
python scripts/local_run.py setup

# Or individual steps
python scripts/local_run.py dataset
python scripts/local_run.py index
python scripts/local_run.py query --query "Casual weekend outfit for a city walk."
python scripts/local_run.py eval
```

Use `--no-vlm` during indexing for faster CPU smoke tests (metadata falls back to Fashionpedia annotations).

## Assignment Evaluation Queries

Configured in `configs/default.yaml`:

1. Attribute specific: "A person in a bright yellow raincoat."
2. Contextual/place: "Professional business attire inside a modern office."
3. Complex semantic: "Someone wearing a blue shirt sitting on a park bench."
4. Style inference: "Casual weekend outfit for a city walk."
5. Compositional: "A red tie and a white shirt in a formal setting."

Evaluation compares:

- `semantic_only` (FashionSigLIP dense retrieval)
- `metadata_only` (structured attribute matching)
- `hybrid` (fusion + compositional reranker)
- `baseline_clip` (vanilla CLIP)

Results are written to `data/index/evaluation_report.json`.

## Design Decisions

| Choice | Why |
|--------|-----|
| FashionSigLIP over CLIP | Fashion-domain fine-tuning; better color/category/composition handling |
| Florence-2 metadata | VLM captions + attribute parsing improve multi-attribute queries |
| ChromaDB | Simple persistent vector store; easy to swap for FAISS/Qdrant at 1M scale |
| Hybrid fusion | Dense retrieval handles zero-shot semantics; metadata handles explicit attributes |
| Compositional reranker | Splits query into phrases to reduce CLIP-style global embedding confusion |

## Scalability Notes

- Indexing is batched and stateless; scale horizontally on Modal with more GPU workers.
- Retrieval reranks only top-20 candidates, keeping latency stable as the corpus grows.
- For ~1M images: migrate ChromaDB → Qdrant/FAISS HNSW, precompute embeddings offline, keep the same fusion logic.

## Future Work

- Add geolocation and weather metadata channels
- Region-level garment embeddings for compositionality ("red tie + white shirt")
- Cross-encoder reranker (e.g. BLIP-ITM) for higher precision
- Hard-negative mining on fashion attribute confusion pairs

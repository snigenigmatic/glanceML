# System Overview

## Design Goals

1. Beat vanilla CLIP on compositional fashion queries.
2. Keep logic modular (models ≠ index ≠ retrieval).
3. Scale retrieval to larger corpora via ANN-first design.
4. Run end-to-end on Modal (GPU index + Gradio demo).

## Pipeline

```text
Images (Fashionpedia subset)
   │
   ├── FashionSigLIP → Embeddings
   ├── Florence-2 → Caption / detections → structured metadata
   │
   ▼
ChromaDB (cosine HNSW + metadata)
   │
Natural Language Query
   │
Deterministic attribute parse
   │
Dense Retrieval (top-50)
   │
Metadata Scoring
   │
Weighted Fusion (0.7 dense + 0.3 metadata)
   │
Composition Rerank (top-20)
   │
Top-K Images
```

## Components

### FashionSigLIP
Primary embedding model for semantic fashion retrieval (`Marqo/marqo-fashionSigLIP`).

### Florence-2
Offline VLM metadata: scene, clothing, color, style, caption.

### ChromaDB
Stores embeddings and scalar metadata. Convenient for this scale; replaceable.

### Retrieval score

```text
final = 0.7 * dense_similarity + 0.3 * metadata_overlap + composition_bonus
```

### Modal surface (`modal_app.py`)

| Function | Role |
| --- | --- |
| `prepare_dataset` | Sample Fashionpedia → `/data` volume |
| `build_index` | Part A indexer (GPU) |
| `query` | Part B retriever |
| `evaluate` | Ablations on official prompts |
| `demo` | Gradio ASGI app |

### Scalability

For 1M images:
- Replace Chroma with Qdrant or FAISS IVF/HNSW.
- Batch / shard embedding + VLM jobs.
- Keep rerank only on the top candidate pool.

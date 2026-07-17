# ARCHITECTURE.md

# System Overview

## Design Goals

1. Beat vanilla CLIP.
2. Support compositional fashion queries.
3. Keep the architecture modular.
4. Scale to larger datasets.

## Pipeline

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
Optional Reranking
   │
Top-K Images
```

## Components

### FashionSigLIP
Primary embedding model for semantic retrieval.

### Florence-2
Extracts:
- Scene
- Clothing
- Color
- Style
- Caption

### ChromaDB
Stores embeddings and metadata.

### Retrieval

Final Score =
- 70% semantic similarity
- 30% metadata match

### Scalability

For 1M images:
- Replace Chroma with Qdrant or FAISS IVF/HNSW.
- Batch embedding generation.
- Rerank only top candidates.

## Future Improvements

- Personalization
- Weather-aware retrieval
- Hard-negative mining
- Multi-person reasoning

# Literature / decision scratchpad

## Why vanilla CLIP fails

- Global embeddings mix attributes without binding
- Weak fine-grained fashion color/garment discrimination
- Scene + outfit composition often ignored

## Approaches considered

1. Vanilla CLIP / SigLIP
2. FashionCLIP
3. FashionSigLIP (chosen dense backbone)
4. Caption-only text retrieval
5. Hybrid dense + structured metadata (chosen system)
6. Heavy VLM reranking (deferred to future work)

## Chosen stack

FashionSigLIP + Florence-2 + Chroma + hybrid fusion + composition rerank

See `REPORT.md` for the full write-up suitable for PDF export.

## Evaluation

- Official prompts in `configs/eval_queries.json`
- Proxy metric: attribute Coverage@K
- Ablations: dense_only / hybrid / hybrid_rerank via `modal run modal_app.py::evaluate`

## High Priority

- [x] Download dataset (Fashionpedia on Modal volume)
- [x] Clean / sample dataset (balanced clothing buckets)
- [x] Build indexer (FashionSigLIP + Florence-2 + Chroma)
- [x] Generate embeddings
- [x] Extract metadata
- [x] Create ChromaDB

## Retrieval

- [x] Dense search
- [x] Metadata scoring
- [x] Weighted fusion
- [x] Optional reranker (composition bonus)

## Evaluation

- [x] Coverage@1 / Coverage@5 proxy metrics
- [x] Baseline comparison (dense vs hybrid vs hybrid+rerank)
- [x] Ablation study hook in `src/eval/evaluate.py`

## Demo

- [x] Gradio UI (Modal `demo` ASGI app)
- [x] Example queries
- [x] README

## Final Submission

- [x] REPORT.md (export to PDF for submission)
- [x] GitHub modular layout
- [ ] Screenshots from Modal Gradio after first deploy
- [x] Reproducibility via `modal run modal_app.py --stage all`

# PLAN.md

# Multimodal Fashion Retrieval - Execution Plan

## Objective
Build a fashion retrieval system that outperforms vanilla CLIP by combining:
- FashionSigLIP embeddings
- Florence-2 structured metadata extraction
- Hybrid retrieval
- Lightweight reranking

## Repository Structure

```text
fashion-retrieval/
├── data/
├── src/
├── notebooks/
├── configs/
├── eval/
├── app.py
├── PLAN.md
├── ARCHITECTURE.md
├── REPORT_NOTES.md
└── TODO.md
```

## Milestone 1 - Dataset
Owner: Data Agent

Tasks
- Download Fashionpedia
- Sample 500-1000 images
- Verify office, park, street, home coverage
- Resize images
- Create metadata.json

Definition of Done
- Dataset validated
- Metadata generated

## Milestone 2 - Indexer
Owner: Indexing Agent

Tasks
- Generate FashionSigLIP embeddings
- Store in ChromaDB
- Persist metadata

Definition of Done
- Searchable vector index exists

## Milestone 3 - Metadata Extraction
Owner: Vision Agent

Use Florence-2 to extract:
- Clothing type
- Clothing color
- Scene
- Style
- Caption

## Milestone 4 - Retriever
Owner: Retrieval Agent

Implement:
- Dense retrieval
- Metadata scoring
- Weighted fusion

## Milestone 5 - Reranker
Owner: Ranking Agent

Input:
Top-20 candidates

Output:
Final ranked list

## Milestone 6 - Evaluation

Compare:
- Vanilla CLIP
- FashionSigLIP
- Hybrid
- Hybrid + Reranker

Metrics:
- Precision@5
- Recall@5

## Milestone 7 - Demo

Deliver a Gradio application with:
- Query textbox
- Top-K results
- Similarity score
- Matched attributes

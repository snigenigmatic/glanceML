# Multimodal Fashion & Context Retrieval — Technical Report

## 1. Approaches considered

| Approach | Idea | Strengths | Weaknesses | When it fits |
| --- | --- | --- | --- | --- |
| A. Vanilla CLIP / SigLIP | Single global image–text embedding, ANN search | Simple, zero-shot, scalable | Weak compositionality; confuses attribute binding (“red shirt, blue pants”) | Broad web search, coarse filters |
| B. Fashion-tuned CLIP (FashionCLIP / FashionSigLIP) | Same interface, domain-adapted contrastive training | Better garment/color/material recall | Still one vector; scene + multi-garment binding limited | Product catalog search |
| C. Caption-then-BM25/text | VLM captions indexed as text | Interpretable | Loses visual nuance; brittle phrasing | Small corpora, explainability focus |
| D. Hybrid: fashion embeddings + structured metadata | Dense recall + attribute filters/scores | Fixes compositionality without training a new model | Needs a metadata extractor; fusion weights matter | **Fashion + context queries (this assignment)** |
| E. Heavy VLM rerank of top-k | Ask a large VLM to score each candidate | Strong reasoning | Expensive at query time; overkill for internship scope | High-precision production rerank stage |

**Tradeoff summary:** pure dense models scale and are zero-shot, but fail the assignment’s compositional/fashion cases. Full generative reranking is accurate but slow/costly. Hybrid retrieval is the best precision–cost point: keep ANN for scale, add cheap structured scores for binding.

## 2. Chosen architecture

```text
Images → FashionSigLIP embeddings
      → Florence-2 captions / detections → {color, clothing, scene, style}
      → ChromaDB (vectors + metadata)

Query → attribute parse + FashionSigLIP text embed
     → ANN top-50
     → metadata overlap score
     → score = 0.7·dense + 0.3·metadata
     → lightweight composition rerank on top-20
     → top-k
```

### Why these models

1. **FashionSigLIP (`Marqo/marqo-fashionSigLIP`)** — fine-tuned from SigLIP with generalized contrastive learning on fashion attributes (category, color, material, keywords). Empirically stronger than FashionCLIP / generic CLIP on fashion retrieval benchmarks, while remaining a drop-in embedding model.
2. **Florence-2-base** — lightweight VLM with reliable detailed captioning and open-vocabulary detection. We do **not** treat it as a chatbot; we use task tokens (`<MORE_DETAILED_CAPTION>`, `<OPEN_VOCABULARY_DETECTION>`) and parse attributes deterministically. That keeps indexing reproducible and avoids an extra LLM.
3. **ChromaDB** — easiest persistent vector DB with metadata. Sufficient for 10³–10⁵ images; HNSW cosine search. At ~1M images, swap the store for Qdrant/FAISS IVF-PQ without changing retrieval math.
4. **Hybrid fusion + composition bonus** — dense score finds semantic neighbors; metadata score enforces multi-attribute constraints; reranker boosts candidates that jointly satisfy ≥2 constrained axes (the failure mode of bag-of-embedding CLIP).

### How fashion queries are handled

- **Attribute-specific** (“bright yellow raincoat”) — color + garment terms in metadata; FashionSigLIP already biased toward fashion lexicon.
- **Contextual/place** (“business attire in a modern office”) — Florence-2/scene tags + style inference (`business` → `formal`).
- **Compositional** (“red tie and white shirt”) — both colors and both garments must appear for a high metadata/rerank score; dense-only cannot guarantee binding.
- **Style inference** (“casual weekend city walk”) — style aliases + scene priors (`weekend`→casual, `city`→street).

### Shortcomings (acknowledged)

- Florence-2 captions can miss rare garments; we merge Fashionpedia seed labels at index time as backup.
- Fusion weights (0.7/0.3) are heuristic; should be tuned on a labeled relevance set.
- Attribute parser is keyword-based (explicit, debuggable) but not open-vocabulary for novel slang.
- Dataset scene labels from Fashionpedia are weak; VLM scene extraction is doing the heavy lifting for environment.

## 3. Dataset

- Source: **Fashionpedia** via Hugging Face `detection-datasets/fashionpedia`.
- Sampled **~800 images**, resized to 512px, balanced across coarse clothing buckets.
- Axes covered: clothing types (annotations + caption parse), colors (**from Florence captions + color–garment pairs**), environments (VLM + soft priors).
- Open-vocab detection dumps were removed — they tagged ~16 garments on every image and made metadata scoring meaningless.
- Artifacts: `metadata.json` (seed labels), `index_meta.json` (VLM-enriched), Chroma persistence.

## 3b. Evaluation (what is graded)

Primary metrics: **Precision@K / Recall@K / AP@K** against hand labels in `configs/relevance_labels.json`.

Compared systems:
1. Vanilla CLIP baseline (`openai/clip-vit-base-patch32`)
2. FashionSigLIP dense-only
3. Hybrid (dense + metadata)
4. Hybrid + composition / pair rerank

Attribute Coverage@K is a **diagnostic only** (self-grading against system metadata). Do not use it as the main claim.

## 4. Scalability to 1M images

| Stage | 800 imgs | 1M imgs |
| --- | --- | --- |
| Embedding | batch GPU jobs | sharded Modal/map jobs |
| Metadata | Florence-2 once offline | same, async workers |
| Store | Chroma HNSW | Qdrant / FAISS IVF-HNSW + metadata index |
| Query | top-50 then rerank-20 | unchanged compute profile (ANN + tiny rerank) |

Retrieval logic is already ANN-first; only the backend index class needs to change.

## 5. Zero-shot behavior

No task-specific training on the eval prompts. FashionSigLIP + Florence-2 generalize from pretraining; attribute parsing is lexicon-based, not overfit to the five graded queries.

## 6. Future work

### a. Locations (cities/places) and weather

- Add geo/place tags (EXIF, filename, or a places CNN / geocell classifier) as metadata fields.
- Condition retrieval with weather tokens (`rain` → raincoat/umbrella boost) via a small side model or OpenWeather lookup at query time.
- Expand query parser with place names and weather lexicon; fuse as a fifth axis with a low weight so it doesn’t dominate garment match.

### b. Precision improvements

- Learn fusion weights on a small judged set (logistic regression on `[dense, meta, bonus]`).
- Region-level embeddings (crop garments with Florence/YOLO, embed parts) for true color–garment binding.
- Hard-negative mining / fashion-specific reranker (cross-encoder) on top-20 only.
- Human feedback loop in the Gradio demo to collect relevance for continual tuning.

## 7. Reproducibility

See `README.md` for Modal commands. Core modules: `src/indexer/build_index.py` (Part A), `src/retriever/hybrid.py` (Part B).

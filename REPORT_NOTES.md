# REPORT_NOTES.md

# Literature Review

## Why Vanilla CLIP Fails

Common issues:
- Weak compositional reasoning
- Global embeddings
- Fine-grained attribute confusion

## Approaches Considered

1. Vanilla CLIP
2. SigLIP2
3. FashionSigLIP
4. Hybrid retrieval
5. VLM reranking

## Chosen Approach

FashionSigLIP
+
Florence-2
+
Hybrid Retrieval
+
Lightweight Reranker

## Evaluation

Metrics:
- Precision@1
- Precision@5
- Recall@5

Baselines:
- CLIP
- FashionSigLIP
- Hybrid
- Hybrid + Reranking

## Report Figures

- System architecture
- Retrieval pipeline
- Ablation table
- Precision comparison
- Qualitative retrieval grids

## Future Work

- Larger datasets
- Better rerankers
- Region-level embeddings
- Online feedback learning

## References

Keep all paper citations and benchmark values here instead of PLAN.md.

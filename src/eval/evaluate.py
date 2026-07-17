"""Evaluation utilities and baseline comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import load_config, resolve_path
from src.data.preprocess import load_metadata
from src.indexer.indexer import build_index
from src.models.embeddings import EmbeddingModel
from src.retriever.search import FashionRetriever
from src.retriever.query_parser import parse_query


def _clip_baseline_search(
    query: str,
    config: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Vanilla CLIP retrieval over the sampled metadata corpus."""
    records = load_metadata(resolve_path(config, "metadata_file"))
    valid_records = [
        record for record in records if Path(record["image_path"]).exists()
    ]
    model = EmbeddingModel.from_pretrained(config["models"]["baseline_model"])
    images = [Image.open(record["image_path"]).convert("RGB") for record in valid_records]
    image_embeddings = model.encode_images(images)
    text_embedding = model.encode_texts([query])[0]
    similarities = (image_embeddings @ text_embedding).squeeze()

    ranked_indices = torch_top_indices(similarities, top_k)
    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        record = valid_records[int(index)]
        score = float(similarities[int(index)].item())
        results.append(
            {
                "id": record["id"],
                "image_path": record["image_path"],
                "score": score,
                "semantic_score": score,
                "metadata_score": 0.0,
                "matched_attributes": {},
                "record": record,
            }
        )
    return results


def torch_top_indices(similarities, top_k: int):
    import torch

    if similarities.ndim == 0:
        return [0]
    values, indices = torch.topk(similarities, k=min(top_k, similarities.shape[0]))
    return indices.tolist()


def _attribute_recall(parsed_attributes: dict[str, list[str]], results: list[dict[str, Any]]) -> float:
    expected = {
        key: {value.lower() for value in values}
        for key, values in parsed_attributes.items()
        if values
    }
    if not expected:
        return 0.0

    hits = 0
    total = len(expected)
    top_result = results[0] if results else {}
    matched = top_result.get("matched_attributes", {})

    for field, values in expected.items():
        if values & set(matched.get(field, [])):
            hits += 1
        else:
            record = top_result.get("record", {})
            record_values = record.get(field, [])
            if isinstance(record_values, str):
                record_values = [record_values]
            if values & {value.lower() for value in record_values}:
                hits += 1

    return hits / total


def evaluate_retrieval(
    config_path: str | Path | None = None,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    modes = modes or ["hybrid", "semantic_only", "metadata_only", "baseline_clip"]

    report: dict[str, Any] = {"modes": {}, "queries": config["evaluation"]["queries"]}

    for mode in modes:
        if mode == "baseline_clip":
            use_reranker = False
        else:
            retriever = FashionRetriever(config_path=config_path)
            use_reranker = mode == "hybrid"

        mode_scores: list[dict[str, Any]] = []
        for item in config["evaluation"]["queries"]:
            parsed = parse_query(item["query"])
            if mode == "baseline_clip":
                results = _clip_baseline_search(item["query"], config, top_k=5)
            elif mode == "semantic_only":
                results = retriever.search(item["query"], use_reranker=False)
                for result in results:
                    result["score"] = result["semantic_score"]
                results.sort(key=lambda row: row["score"], reverse=True)
            elif mode == "metadata_only":
                results = retriever.search(item["query"], use_reranker=False)
                for result in results:
                    result["score"] = result["metadata_score"]
                results.sort(key=lambda row: row["score"], reverse=True)
            else:
                results = retriever.search(item["query"], use_reranker=use_reranker)

            recall = _attribute_recall(item["expected_attributes"], results[:5])
            mode_scores.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "attribute_recall@5": recall,
                    "top_result": {
                        "id": results[0]["id"] if results else None,
                        "score": results[0].get("score") if results else None,
                        "matched_attributes": results[0].get("matched_attributes") if results else {},
                    },
                }
            )

        report["modes"][mode] = {
            "mean_attribute_recall@5": sum(row["attribute_recall@5"] for row in mode_scores)
            / max(len(mode_scores), 1),
            "per_query": mode_scores,
        }

    output_file = resolve_path(config, "index_dir") / "evaluation_report.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    report["report_path"] = str(output_file)
    return report


def run_full_pipeline(config_path: str | Path | None = None, force: bool = False) -> dict[str, Any]:
    from src.data.download import download_and_sample_dataset

    metadata_file = download_and_sample_dataset(config_path=config_path, force=force)
    index_info = build_index(config_path=config_path, force=force)
    evaluation = evaluate_retrieval(config_path=config_path)
    return {
        "metadata_file": str(metadata_file),
        "index": index_info,
        "evaluation": evaluation,
    }

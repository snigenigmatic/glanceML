"""Evaluation helpers: attribute proxy metrics + optional dense-only ablations.

True Precision/Recall need human relevance labels. For this assignment we report:
1) Attribute Coverage@K — fraction of constrained query attributes present in top-k
2) Dense-only vs Hybrid vs Hybrid+Rerank score tables on the official prompts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.attributes import FashionAttributes, parse_attributes
from src.retriever.hybrid import FashionRetriever
from src.utils import load_config, read_json, resolve_path, write_json


def attribute_coverage(query_expected: dict[str, list[str]], result_attrs: FashionAttributes) -> float:
    scores: list[float] = []
    mapping = {
        "colors": set(result_attrs.normalized().colors),
        "clothing": set(result_attrs.normalized().clothing),
        "scenes": set(result_attrs.normalized().scenes),
        "styles": set(result_attrs.normalized().styles),
    }
    for key, expected in query_expected.items():
        if not expected:
            continue
        exp = parse_attributes(" ".join(expected))
        exp_vals = getattr(exp, key)
        if not exp_vals:
            continue
        got = mapping.get(key, set())
        hits = sum(1 for v in exp_vals if v in got)
        scores.append(hits / len(exp_vals))
    return sum(scores) / len(scores) if scores else 0.0


def coverage_at_k(results: list[dict[str, Any]], expected: dict[str, list[str]], k: int) -> float:
    if not results:
        return 0.0
    top = results[:k]
    vals = []
    for r in top:
        attrs = FashionAttributes(
            colors=r["matched_attributes"]["colors"],
            clothing=r["matched_attributes"]["clothing"],
            scenes=r["matched_attributes"]["scenes"],
            styles=r["matched_attributes"]["styles"],
            caption=r.get("caption", ""),
        )
        vals.append(attribute_coverage(expected, attrs))
    return sum(vals) / len(vals)


def run_evaluation(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    queries = read_json(resolve_path(cfg["eval"]["queries_path"]))
    retriever = FashionRetriever(config_path=config_path, data_dir=data_dir)

    # Ablations by temporarily toggling fusion / rerank
    modes = {
        "dense_only": {"dense_weight": 1.0, "metadata_weight": 0.0, "rerank": False},
        "hybrid": {
            "dense_weight": float(cfg["retrieval"]["dense_weight"]),
            "metadata_weight": float(cfg["retrieval"]["metadata_weight"]),
            "rerank": False,
        },
        "hybrid_rerank": {
            "dense_weight": float(cfg["retrieval"]["dense_weight"]),
            "metadata_weight": float(cfg["retrieval"]["metadata_weight"]),
            "rerank": True,
        },
    }

    report: dict[str, Any] = {"queries": [], "summary": {}}

    for mode_name, settings in modes.items():
        retriever.dense_weight = settings["dense_weight"]
        retriever.metadata_weight = settings["metadata_weight"]
        retriever.rerank_enabled = settings["rerank"]
        mode_rows = []
        cov1, cov5 = [], []
        for q in queries:
            hits = [r.to_dict() for r in retriever.search(q["query"], top_k=top_k)]
            c1 = coverage_at_k(hits, q["expected"], k=1)
            c5 = coverage_at_k(hits, q["expected"], k=5)
            cov1.append(c1)
            cov5.append(c5)
            mode_rows.append(
                {
                    "id": q["id"],
                    "query": q["query"],
                    "coverage@1": c1,
                    "coverage@5": c5,
                    "top_results": hits,
                }
            )
        report[mode_name] = {
            "per_query": mode_rows,
            "mean_coverage@1": sum(cov1) / len(cov1),
            "mean_coverage@5": sum(cov5) / len(cov5),
        }

    report["summary"] = {
        mode: {
            "mean_coverage@1": report[mode]["mean_coverage@1"],
            "mean_coverage@5": report[mode]["mean_coverage@5"],
        }
        for mode in modes
    }

    out_dir = resolve_path(cfg["eval"]["output_dir"])
    if data_dir is not None:
        out_dir = resolve_path(Path(data_dir) / "eval_results")
    out_path = out_dir / "eval_report.json"
    write_json(out_path, report)
    print("Summary:", report["summary"])
    print(f"Wrote {out_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run_evaluation(config_path=args.config, data_dir=args.data_dir, top_k=args.top_k)

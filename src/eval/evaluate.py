"""Honest evaluation: Precision/Recall against hand labels + CLIP baseline.

Coverage@K (self-grading attribute overlap) is retained only as a diagnostic,
not as the primary claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.attributes import FashionAttributes, parse_attributes
from src.eval.clip_baseline import ClipBaselineRetriever
from src.retriever.hybrid import FashionRetriever
from src.utils import load_config, read_json, resolve_path, write_json


def precision_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for i in top if i in relevant) / len(top)


def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant) / len(relevant)


def average_precision(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, image_id in enumerate(retrieved_ids[:k], 1):
        if image_id in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def attribute_coverage(query_expected: dict[str, list[str]], result_attrs: FashionAttributes) -> float:
    """Diagnostic only — grades against system metadata, not human relevance."""
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


def _hard_negative_analysis(
    retrieved_ids: list[str],
    relevant: set[str],
    hard_negatives: set[str],
    k: int = 10,
) -> dict[str, Any]:
    top = retrieved_ids[:k]
    first_rel = next((i for i, x in enumerate(top) if x in relevant), None)
    first_hn = next((i for i, x in enumerate(top) if x in hard_negatives), None)
    return {
        "first_relevant_rank": first_rel + 1 if first_rel is not None else None,
        "first_hard_negative_rank": first_hn + 1 if first_hn is not None else None,
        "relevant_before_hard_neg": (
            first_rel is not None and (first_hn is None or first_rel < first_hn)
        ),
        "top_k_ids": top,
    }


def run_evaluation(
    config_path: str | None = None,
    data_dir: str | Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    data_root = resolve_path(data_dir or cfg["dataset"]["output_dir"])
    queries_path = resolve_path(cfg["eval"]["queries_path"])
    # Prefer packaged relevance labels next to eval queries
    labels_path = resolve_path("configs/relevance_labels.json")
    if data_dir is not None:
        # Modal image mounts configs at /root/configs
        alt = Path("/root/configs/relevance_labels.json")
        if alt.exists():
            labels_path = alt

    queries = read_json(queries_path)
    labels_doc = read_json(labels_path)
    labels_by_id = {q["id"]: q for q in labels_doc["queries"]}

    retriever = FashionRetriever(config_path=config_path, data_dir=data_dir)

    modes = {
        "clip_baseline": None,  # separate retriever
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

    clip_retriever = None
    clip_index = data_root / "baselines" / "clip_embeddings.npy"
    if clip_index.exists():
        clip_retriever = ClipBaselineRetriever(data_root, config_path=config_path)
    else:
        print("WARNING: CLIP baseline index missing; run build_clip_index first.")

    report: dict[str, Any] = {
        "metric_notes": {
            "primary": "Precision@K / Recall@K / AP@K against hand-labeled relevance_labels.json",
            "diagnostic": "coverage@K is self-grading (system metadata vs query parse) — not used for claims",
            "labels": str(labels_path),
        },
        "summary": {},
    }

    expected_by_id = {q["id"]: q.get("expected", {}) for q in queries}

    for mode_name, settings in modes.items():
        if mode_name == "clip_baseline" and clip_retriever is None:
            report[mode_name] = {"skipped": True, "reason": "CLIP index not found"}
            continue

        if mode_name != "clip_baseline":
            retriever.dense_weight = settings["dense_weight"]
            retriever.metadata_weight = settings["metadata_weight"]
            retriever.rerank_enabled = settings["rerank"]

        rows = []
        p1, p5, r5, ap5 = [], [], [], []
        composition_checks = []

        for q in queries:
            qid = q["id"]
            label = labels_by_id[qid]
            relevant = set(label["relevant"])
            hard_neg = set(label.get("hard_negatives") or [])

            if mode_name == "clip_baseline":
                hits = clip_retriever.search(q["query"], top_k=max(top_k, 10))
            else:
                hits = [r.to_dict() for r in retriever.search(q["query"], top_k=max(top_k, 10))]

            retrieved_ids = [h["image_id"] for h in hits]
            metrics = {
                "precision@1": precision_at_k(retrieved_ids, relevant, 1),
                "precision@5": precision_at_k(retrieved_ids, relevant, 5),
                "recall@5": recall_at_k(retrieved_ids, relevant, 5),
                "ap@5": average_precision(retrieved_ids, relevant, 5),
            }
            # diagnostic coverage on top-5
            cov_vals = []
            for h in hits[:5]:
                attrs = FashionAttributes(
                    colors=h["matched_attributes"]["colors"],
                    clothing=h["matched_attributes"]["clothing"],
                    scenes=h["matched_attributes"]["scenes"],
                    styles=h["matched_attributes"]["styles"],
                    caption=h.get("caption", ""),
                )
                cov_vals.append(attribute_coverage(expected_by_id.get(qid, {}), attrs))
            metrics["diagnostic_coverage@5"] = sum(cov_vals) / len(cov_vals) if cov_vals else 0.0

            hn = _hard_negative_analysis(retrieved_ids, relevant, hard_neg, k=10)
            if qid == "q5_compositional":
                composition_checks.append(hn)

            p1.append(metrics["precision@1"])
            p5.append(metrics["precision@5"])
            r5.append(metrics["recall@5"])
            ap5.append(metrics["ap@5"])

            rows.append(
                {
                    "id": qid,
                    "query": q["query"],
                    "relevant": sorted(relevant),
                    "hard_negatives": sorted(hard_neg),
                    **metrics,
                    "hard_negative_analysis": hn,
                    "top_results": hits[:top_k],
                }
            )

        summary = {
            "mean_precision@1": sum(p1) / len(p1),
            "mean_precision@5": sum(p5) / len(p5),
            "mean_recall@5": sum(r5) / len(r5),
            "mean_ap@5": sum(ap5) / len(ap5),
        }
        if composition_checks:
            summary["q5_relevant_before_hard_neg"] = all(
                c["relevant_before_hard_neg"] for c in composition_checks
            )

        report[mode_name] = {"per_query": rows, **summary}
        report["summary"][mode_name] = summary

    out_dir = resolve_path(cfg["eval"]["output_dir"])
    if data_dir is not None:
        out_dir = resolve_path(Path(data_dir) / "eval_results")
    out_path = out_dir / "eval_report.json"
    write_json(out_path, report)

    print("=== Primary metrics (hand-labeled P/R) ===")
    for mode, s in report["summary"].items():
        print(mode, s)
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

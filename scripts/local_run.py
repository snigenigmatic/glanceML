"""Run the pipeline locally without Modal (CPU-friendly smoke tests)."""

from __future__ import annotations

import argparse
import json

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fashion retrieval pipeline")
    parser.add_argument(
        "action",
        choices=["dataset", "index", "query", "eval", "setup"],
        help="Pipeline step to run",
    )
    parser.add_argument("--query", default="A person in a bright yellow raincoat.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-vlm", action="store_true", help="Skip Florence-2 during indexing")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config()

    if args.action == "dataset":
        from src.data.download import cleanup_raw_downloads, download_and_sample_dataset

        path = download_and_sample_dataset(force=args.force)
        cleanup_raw_downloads()
        print(json.dumps({"metadata_file": str(path)}, indent=2))
        return

    if args.action == "index":
        from src.indexer.indexer import build_index

        print(json.dumps(build_index(force=args.force, use_vlm=not args.no_vlm), indent=2))
        return

    if args.action == "query":
        from src.retriever.search import search

        results = search(query=args.query, top_k=args.top_k)
        print(json.dumps(results, indent=2, default=str))
        return

    if args.action == "eval":
        from src.eval.evaluate import evaluate_retrieval

        print(json.dumps(evaluate_retrieval(), indent=2))
        return

    if args.action == "setup":
        from src.eval.evaluate import run_full_pipeline

        print(json.dumps(run_full_pipeline(force=args.force), indent=2, default=str))


if __name__ == "__main__":
    main()

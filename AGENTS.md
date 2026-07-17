# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python (>=3.10) product: **Glance ML — multimodal fashion retrieval**
(FashionSigLIP embeddings + Florence-2 VLM metadata + ChromaDB hybrid retrieval).
There are no standalone service processes (ChromaDB is an in-process library). The
intended full pipeline runs on **Modal GPUs**; see `README.md` / `ARCHITECTURE.md`
for the standard `modal run` / `modal serve` commands.

Dependencies are installed into the system Python by the startup update script
(`pip install --break-system-packages -r requirements.txt`). Run everything with
`python3` from the repo root (`/workspace`); `python -m ...` puts the repo root on
`sys.path`, so no `PYTHONPATH` export is needed for module runs. For plain scripts
outside the repo, set `PYTHONPATH=/workspace`.

### Tests / lint / build
- Tests: `python3 -m pytest tests/` — pure CPU logic (attribute parsing + scoring),
  no GPU, no network, no model downloads. Fast and deterministic.
- Lint: none configured (no ruff/flake8/black/mypy config in the repo).
- Build: `python3 -m build` (setuptools); not needed for dev.

### Running the app in this cloud VM (no GPU)
The cloud VM has **no GPU** and Modal requires external auth (`modal setup`) + a GPU,
so the real Fashionpedia + Florence-2 pipeline cannot run here. Use the local CPU
smoke path to exercise core retrieval end to end:

1. `configs/default.yaml` sets `models.device: cuda`. For local CPU runs you MUST
   override it, e.g. `sed 's/device: cuda/device: cpu/' configs/default.yaml > /tmp/cpu_config.yaml`
   and pass `--config /tmp/cpu_config.yaml` (do not edit the committed config).
2. `python3 -m src.dataset.prepare --demo --output-dir data` — tiny synthetic corpus,
   no network. NOTE: demo images are solid color blocks by design (not photographs).
3. `python3 -m src.indexer.build_index --config /tmp/cpu_config.yaml --data-dir data --no-vlm --max-items 32`
   — downloads `Marqo/marqo-fashionSigLIP` from HuggingFace (needs network) and runs
   embeddings on CPU. `--no-vlm` skips Florence-2 (uses the heuristic extractor).
4. `python3 -m src.retriever.hybrid "a red shirt in an office" --config /tmp/cpu_config.yaml --data-dir data`
5. Gradio demo locally: build a `FashionRetriever` with the CPU config + `data/`,
   pass it to `src.demo_ui.build_demo`, and `blocks.launch()`. The `demo` entrypoint
   in `modal_app.py` is Modal-only and cannot be imported/run directly here.

### Dependency note (important)
`requirements.txt` is pinned to match the validated Modal image in `modal_app.py`.
Do NOT relax these to floating `>=` latest: newer `transformers` 5.x / `open_clip` 3.x
break the FashionSigLIP loader with a `Cannot copy out of meta tensor` error.

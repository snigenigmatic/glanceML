"""Evaluation helpers."""

__all__ = ["evaluate_retrieval", "run_full_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from . import evaluate as evaluate_module

        return getattr(evaluate_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

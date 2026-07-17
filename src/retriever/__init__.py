"""Retrieval pipeline."""

from .query_parser import ParsedQuery, parse_query

__all__ = ["ParsedQuery", "parse_query", "FashionRetriever", "search"]


def __getattr__(name: str):
    if name == "FashionRetriever":
        from .search import FashionRetriever

        return FashionRetriever
    if name == "search":
        from .search import search

        return search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

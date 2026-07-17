"""Unit tests for attribute parsing and hybrid scoring logic (no GPU)."""

from src.attributes import metadata_match_score, parse_attributes
from src.retriever.rerank import composition_bonus
from src.retriever.types import RetrievalResult


def test_compositional_query_parses_both_colors_and_garments():
    attrs = parse_attributes("A red tie and a white shirt in a formal setting.")
    assert "red" in attrs.colors
    assert "white" in attrs.colors
    assert "tie" in attrs.clothing
    assert "shirt" in attrs.clothing
    assert "formal" in attrs.styles or "office" in attrs.scenes


def test_metadata_score_rewards_joint_matches():
    query = parse_attributes("Someone wearing a blue shirt sitting on a park bench.")
    good = parse_attributes("blue shirt in a park")
    bad = parse_attributes("red dress in an office")
    assert metadata_match_score(query, good) > metadata_match_score(query, bad)


def test_rerank_bonus_prefers_multi_axis_hits():
    query = parse_attributes("yellow raincoat on the street")
    multi = RetrievalResult(
        image_id="1",
        path="x",
        score=0.5,
        dense_score=0.5,
        metadata_score=0.5,
        caption="yellow raincoat street",
        colors=["yellow"],
        clothing=["coat"],
        scenes=["street"],
        styles=["casual"],
    )
    single = RetrievalResult(
        image_id="2",
        path="y",
        score=0.5,
        dense_score=0.5,
        metadata_score=0.2,
        caption="yellow wall",
        colors=["yellow"],
        clothing=[],
        scenes=[],
        styles=[],
    )
    assert composition_bonus(query, multi) > composition_bonus(query, single)

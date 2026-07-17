"""Unit tests for attribute parsing and hybrid scoring logic (no GPU)."""

from src.attributes import (
    extract_color_garment_pairs,
    metadata_match_score,
    parse_attributes,
    parse_caption_attributes,
)
from src.retriever.rerank import composition_bonus
from src.retriever.types import RetrievalResult


def test_compositional_query_parses_both_colors_and_garments():
    attrs = parse_attributes("A red tie and a white shirt in a formal setting.")
    assert "red" in attrs.colors
    assert "white" in attrs.colors
    assert "tie" in attrs.clothing
    assert "shirt" in attrs.clothing
    assert ("red", "tie") in attrs.pairs
    assert ("white", "shirt") in attrs.pairs
    assert "formal" in attrs.styles or "office" in attrs.scenes


def test_caption_parser_does_not_inflate_clothing_from_prompt_dump():
    caption = (
        "The man is wearing a navy blue suit with a white shirt and a maroon tie."
    )
    # Simulate the old bug: open-vocab dump appended to text
    polluted = caption + ". shirt, t-shirt, blouse, dress, jacket, coat, raincoat, blazer, suit, hoodie, jeans, pants, skirt, tie, sneakers, boots"
    clean = parse_caption_attributes(caption)
    dirty = parse_attributes(polluted)
    assert len(clean.clothing) <= 6
    assert len(dirty.clothing) > len(clean.clothing)
    assert "sneakers" not in clean.clothing
    assert "hoodie" not in clean.clothing


def test_color_garment_pairs_from_caption():
    pairs = extract_color_garment_pairs("She wears a yellow raincoat and black boots.")
    assert ("yellow", "raincoat") in pairs
    assert ("black", "boots") in pairs


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


def test_compositional_pair_scoring_prefers_correct_binding():
    query = parse_attributes("A red tie and a white shirt in a formal setting.")
    correct = parse_caption_attributes(
        "He is wearing a navy blue suit with a white shirt and a maroon tie."
    )
    swapped = parse_caption_attributes(
        "He is wearing a black suit with a red shirt and a white tie."
    )
    assert metadata_match_score(query, correct) > metadata_match_score(query, swapped)

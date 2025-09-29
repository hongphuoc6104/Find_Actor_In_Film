import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from tasks.character_task import (
    DEFAULT_HIGHLIGHT_DET_SCORE,
    DEFAULT_HIGHLIGHT_GAP_SECONDS,
    DEFAULT_HIGHLIGHT_SIMILARITY,
    HIGHLIGHT_MIN_DURATION,
    HIGHLIGHT_MIN_SCORE,
    _build_highlights,
    _limit_highlights_per_scene,
    _make_highlight_matcher,

)



def test_build_highlights_requires_matching_target():
    entries = [
        {
            "timestamp": 1.0,

            "det_score": 0.92,
            "cluster_id": "cluster-A",

            "actor_similarity": DEFAULT_HIGHLIGHT_SIMILARITY + 0.1,
        }
    ]

    matcher = _make_highlight_matcher("final-1", {"cluster-B"}, DEFAULT_HIGHLIGHT_SIMILARITY)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,
        max_gap=DEFAULT_HIGHLIGHT_GAP_SECONDS,
        match_fn=matcher,
        sim_threshold=DEFAULT_HIGHLIGHT_SIMILARITY,
    )

    assert highlights == []


def test_highlight_expands_short_segment_to_min_duration():
    similarity = DEFAULT_HIGHLIGHT_SIMILARITY + 0.15
    entries = [
        {
            "timestamp": 5.0,

            "det_score": 0.95,
            "cluster_id": "cluster-A",
            "actor_similarity": similarity,
        }
    ]


    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, DEFAULT_HIGHLIGHT_SIMILARITY)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,

        max_gap=DEFAULT_HIGHLIGHT_GAP_SECONDS,
        match_fn=matcher,
        sim_threshold=DEFAULT_HIGHLIGHT_SIMILARITY,
    )

    assert len(highlights) == 1
    highlight = highlights[0]

    assert highlight["has_target"] is True
    assert highlight["duration"] >= HIGHLIGHT_MIN_DURATION - 1e-6
    assert highlight["score"] == pytest.approx(similarity, rel=1e-6)


def test_highlight_merges_hits_within_gap():
    entries = [
        {

            "timestamp": 10.0,
            "det_score": 0.9,
            "cluster_id": "cluster-A",
            "actor_similarity": DEFAULT_HIGHLIGHT_SIMILARITY + 0.2,
        },
        {

            "timestamp": 10.0 + DEFAULT_HIGHLIGHT_GAP_SECONDS / 2.0,
            "det_score": 0.7,
            "final_character_id": "final-1",
            "actor_similarity": DEFAULT_HIGHLIGHT_SIMILARITY + 0.05,
        },
    ]


    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, DEFAULT_HIGHLIGHT_SIMILARITY)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,

        max_gap=DEFAULT_HIGHLIGHT_GAP_SECONDS,
        match_fn=matcher,
        sim_threshold=DEFAULT_HIGHLIGHT_SIMILARITY,
    )


    assert len(highlights) == 1
    highlight = highlights[0]
    assert highlight["match_count"] == 2
    assert highlight["matched_cluster_ids"] == ["cluster-A"]
    assert highlight["matched_final_character_ids"] == ["final-1"]
    expected_score = (
        (DEFAULT_HIGHLIGHT_SIMILARITY + 0.2) * 0.9
        + (DEFAULT_HIGHLIGHT_SIMILARITY + 0.05) * 0.7
    ) / (0.9 + 0.7)
    assert highlight["score"] == pytest.approx(expected_score, rel=1e-6)



def test_highlight_filters_low_scores_and_limits():
    high_sim = DEFAULT_HIGHLIGHT_SIMILARITY + 0.25
    mid_sim = HIGHLIGHT_MIN_SCORE + 0.01
    low_sim = HIGHLIGHT_MIN_SCORE - 0.1
    entries = [
        {
            "timestamp": 1.0,
            "det_score": 0.95,
            "cluster_id": "cluster-A",
            "actor_similarity": high_sim,
        },
        {
            "timestamp": 10.0,
            "det_score": 0.9,
            "cluster_id": "cluster-A",

            "actor_similarity": mid_sim,
        },
        {

            "timestamp": 25.0,
            "det_score": 0.92,
            "cluster_id": "cluster-A",
            "actor_similarity": low_sim,
        },
    ]


    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, DEFAULT_HIGHLIGHT_SIMILARITY)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,
        max_gap=DEFAULT_HIGHLIGHT_GAP_SECONDS,
        match_fn=matcher,
        sim_threshold=DEFAULT_HIGHLIGHT_SIMILARITY,
    )


    assert all(h["score"] >= HIGHLIGHT_MIN_SCORE for h in highlights)
    assert len(highlights) == 2

    limited = _limit_highlights_per_scene(highlights, limit=1)
    assert len(limited) == 1
    assert limited[0]["score"] == pytest.approx(high_sim, rel=1e-6)


def test_build_highlights_skips_entries_below_similarity_threshold():
    entries = [
        {

            "timestamp": 2.0,
            "det_score": 0.95,
            "cluster_id": "cluster-A",
            "actor_similarity": DEFAULT_HIGHLIGHT_SIMILARITY - 0.1,
        }
    ]


    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,
        max_gap=DEFAULT_HIGHLIGHT_GAP_SECONDS,
        match_fn=lambda _: True,
        sim_threshold=DEFAULT_HIGHLIGHT_SIMILARITY,
    )


    assert highlights == []
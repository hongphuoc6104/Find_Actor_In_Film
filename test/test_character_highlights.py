import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from tasks.character_task import (
    DEFAULT_HIGHLIGHT_DET_SCORE,
    _build_highlights,
    _make_highlight_matcher,
    _summarise_highlight_support,
)


def test_build_highlights_filters_unmatched_entries():
    entries = [
        {
            "timestamp": 1.0,
            "det_score": 0.9,
            "cluster_id": "cluster-A",
            "actor_similarity": 0.82,
            "frame": "frame_0001.jpg",
        },
        {
            "timestamp": 1.6,
            "det_score": 0.88,
            "final_character_id": "final-1",
            "actor_similarity": 0.78,
            "frame": "frame_0002.jpg",
        },
        {
            "timestamp": 5.0,
            "det_score": 0.93,
            "cluster_id": "cluster-B",
            "final_character_id": "other",
            "actor_similarity": 0.25,
            "frame": "frame_0003.jpg",
        },
    ]

    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, 0.7)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,
        max_gap=2.0,
        match_fn=matcher,
    )

    assert len(highlights) == 1
    highlight = highlights[0]
    assert highlight["match_count"] == 2
    assert highlight["matched_cluster_ids"] == ["cluster-A"]
    assert highlight["matched_final_character_ids"] == ["final-1"]
    assert highlight["max_det_score"] == pytest.approx(0.9, rel=1e-6)
    assert highlight["min_det_score"] == pytest.approx(0.88, rel=1e-6)
    assert highlight["avg_similarity"] == pytest.approx((0.82 + 0.78) / 2.0, rel=1e-6)
    assert len(highlight.get("supporting_detections", [])) == 2


def test_build_highlights_rejects_entries_without_actor_confirmation():
    entries = [
        {
            "timestamp": 1.0,
            "det_score": 0.95,
            "cluster_id": "cluster-X",
            "actor_similarity": 0.2,
        },
        {
            "timestamp": 1.2,
            "det_score": 0.91,
            "final_character_id": "someone-else",
            "actor_similarity": 0.3,
        },
    ]

    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, 0.6)
    highlights = _build_highlights(
        entries,
        det_th=DEFAULT_HIGHLIGHT_DET_SCORE,
        max_gap=1.0,
        match_fn=matcher,
    )

    assert highlights == []


def test_highlight_support_summary_tracks_scores():
    entries = [
        {
            "timestamp": 1.0,
            "det_score": 0.9,
            "cluster_id": "cluster-A",
            "actor_similarity": 0.85,
        },
        {
            "timestamp": 1.9,
            "det_score": 0.87,
            "final_character_id": "final-1",
            "actor_similarity": 0.8,
        },
    ]

    matcher = _make_highlight_matcher("final-1", {"cluster-A"}, 0.7)
    highlights = _build_highlights(entries, match_fn=matcher)
    summary = _summarise_highlight_support(
        highlights,
        det_threshold=DEFAULT_HIGHLIGHT_DET_SCORE,
        similarity_threshold=0.7,
    )

    assert summary["highlight_count"] == 1
    assert summary["match_count"] == 2
    assert summary["matched_cluster_ids"] == ["cluster-A"]
    assert summary["matched_final_character_ids"] == ["final-1"]
    assert summary["max_det_score"] == pytest.approx(0.9, rel=1e-6)
    assert summary["min_det_score"] == pytest.approx(0.87, rel=1e-6)
    assert summary["avg_similarity"] == pytest.approx((0.85 + 0.8) / 2.0, rel=1e-6)

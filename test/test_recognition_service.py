from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict

import numpy as np

_stored_images: Dict[str, np.ndarray] = {}


def _cv2_imwrite(path: str, img: Any) -> bool:
    _stored_images[path] = np.asarray(img)
    return True


def _cv2_imread(path: str) -> np.ndarray | None:
    data = _stored_images.get(path)
    if data is None:
        return None
    return data.copy()


_dummy_cv2 = ModuleType("cv2")
_dummy_cv2.imread = _cv2_imread
_dummy_cv2.imwrite = _cv2_imwrite
sys.modules.setdefault("cv2", _dummy_cv2)

_dummy_insightface = ModuleType("insightface")
_dummy_app = ModuleType("insightface.app")


class _StubFaceAnalysis:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def prepare(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        pass

    def get(self, img: Any) -> list[Any]:  # pragma: no cover - stub
        return []


_dummy_app.FaceAnalysis = _StubFaceAnalysis
sys.modules.setdefault("insightface", _dummy_insightface)
sys.modules.setdefault("insightface.app", _dummy_app)

_pd_stub = ModuleType("pandas")
_pd_stub.__version__ = "0.0"
sys.modules.setdefault("pandas", _pd_stub)


import pytest

from services import recognition

sys.modules.pop("pandas", None)
sys.modules.pop("insightface", None)
sys.modules.pop("insightface.app", None)


def _build_candidate(
    score: float,
    movie: str = "Movie",
    character_id: str = "1",
    **overrides: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "movie": movie,
        "character_id": character_id,
        "distance": score,
        "count": 1,
        "track_count": 1,
        "previews": [],
        "preview_paths": [],
        "raw_cluster_ids": [],
        "movies": [],
        "scenes": [],
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def configured_thresholds(monkeypatch):
    cfg = {
        "search": {
            "present_threshold": 0.5,
            "near_match_threshold": 0.3,
            "min_score": 0.3,
            "max_results": 10,
        }
    }
    monkeypatch.setattr(recognition, "load_config", lambda: cfg)
    return cfg


def test_recognize_prefers_present_bucket(monkeypatch, configured_thresholds):
    def fake_search(image_path: str, k: int, score_floor: float, max_results: int):
        assert score_floor == pytest.approx(0.3)
        assert max_results == 10
        return {
            "1": [
                _build_candidate(0.62, movie="Movie A", character_id="A"),
                _build_candidate(0.48, movie="Movie A", character_id="B"),
            ],
            "2": [
                _build_candidate(0.44, movie="Movie B", character_id="C"),
            ],
        }

    monkeypatch.setattr(recognition, "search_actor", fake_search)

    result = recognition.recognize("/tmp/image.jpg")

    assert result["is_unknown"] is False
    assert result["best_score"] == pytest.approx(0.62)
    assert len(result["movies"]) == 1

    movie = result["movies"][0]
    assert movie["movie"] == "Movie A"
    assert movie["match_status"] == "present"
    assert movie["match_label"] == recognition.PRESENT_LABEL_VI
    assert all(char["match_status"] == "present" for char in movie["characters"])
    assert {char["character_id"] for char in movie["characters"]} == {"A"}


def test_recognize_uses_near_match_when_no_present(monkeypatch, configured_thresholds):
    def fake_search(image_path: str, k: int, score_floor: float, max_results: int):
        return {
            "1": [
                _build_candidate(0.41, movie="Movie A", character_id="A"),
                _build_candidate(0.36, movie="Movie A", character_id="B"),
            ],
            "2": [
                _build_candidate(0.31, movie="Movie B", character_id="C"),
            ],
            "3": [
                _build_candidate(0.18, movie="Movie C", character_id="D"),
            ],
        }

    monkeypatch.setattr(recognition, "search_actor", fake_search)

    result = recognition.recognize("/tmp/image.jpg")

    assert result["is_unknown"] is True
    assert result["best_score"] == pytest.approx(0.41)
    # Only near-match movies with >=0.3 scores should be returned.
    returned_ids = {movie["movie"] for movie in result["movies"]}
    assert returned_ids == {"Movie A", "Movie B"}
    assert all(movie["match_status"] == "near_match" for movie in result["movies"])
    assert all(
        char["match_label"] == recognition.NEAR_MATCH_LABEL_VI
        for movie in result["movies"]
        for char in movie["characters"]
    )
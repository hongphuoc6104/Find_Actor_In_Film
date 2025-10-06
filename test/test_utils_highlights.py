import logging

from utils.highlights import normalise_highlights


def test_normalise_highlights_merges_and_sorts_segments():
    raw_highlights = [
        {
            "start": 10.0,
            "end": 12.0,
            "match_count": 1,
            "det_score": 0.9,
            "score": 0.5,
        },
        {
            "start": 13.0,
            "end": 14.0,
            "match_count": 2,
            "det_score": 0.8,
            "score": 0.6,
        },
        {
            "start": 20.0,
            "end": 18.0,
            "match_count": 1,
            "det_score": 0.7,
            "score": 0.4,
        },
        {
            "start": None,
            "end": 30.0,
            "match_count": 1,
            "det_score": 0.5,
            "score": 0.3,
        },
        {
            "start": 40.0,
            "end": 46.0,
            "match_count": 3,
            "det_score": 0.95,
            "score": 0.9,
            "duration": 6.0,
        },
    ]

    highlights = normalise_highlights(raw_highlights, merge_gap=1.0)

    assert len(highlights) == 3

    first, second, third = highlights

    assert first["start"] == 10.0
    assert first["end"] == 14.0
    assert first["match_count"] == 3
    assert len(first.get("sources", [])) == 2

    assert second["start"] == 18.0
    assert second["end"] == 20.0
    assert second["match_count"] == 1

    assert third["start"] == 40.0
    assert third["end"] == 46.0
    assert third["match_count"] == 3
    assert third["duration"] == 6.0



def test_normalise_highlights_logs_context_when_debug_fails(monkeypatch, caplog):
    from utils import highlights as hl

    def _raise_debug(*args, **kwargs):
        raise RuntimeError("debug failure")

    monkeypatch.setattr(hl.LOGGER, "debug", _raise_debug)

    raw_highlights = [
        {"start": 1.0, "end": 2.0, "match_count": 1, "score": 0.9},
    ]

    with caplog.at_level(logging.WARNING, logger=hl.LOGGER.name):
        segments = normalise_highlights(
            raw_highlights,
            logger=hl.LOGGER,
            scene_identifier={
                "movie": "movie-1",
                "scene": 5,
                "track": "track-1",
            },
        )

    assert segments, "expected normalised segments even when debug logging fails"
    warning_records = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warning_records, "expected a warning log when highlight debug logging fails"
    message = warning_records[0].message
    assert "movie-1" in message
    assert "'scene': 5" in message
    assert "'track': 'track-1'" in message
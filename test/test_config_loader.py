import pytest

from utils import config_loader


def test_highlight_settings_defaults(monkeypatch):
    monkeypatch.setattr(config_loader, "load_config", lambda: {})

    settings = config_loader.get_highlight_settings()

    assert settings["MIN_HL_DURATION_SEC"] == pytest.approx(4.0)
    assert settings["MERGE_GAP_SEC"] == pytest.approx(2.0)
    assert settings["MIN_SCORE"] == pytest.approx(0.8)
    assert settings["TOP_K_HL_PER_SCENE"] is None


def test_highlight_settings_overrides():
    cfg = {
        "highlight": {
            "MIN_HL_DURATION_SEC": "6.5",
            "MERGE_GAP_SEC": "1.2",
            "MIN_SCORE": "0.75",
            "TOP_K_HL_PER_SCENE": "5",
        }
    }

    settings = config_loader.get_highlight_settings(cfg)

    assert settings["MIN_HL_DURATION_SEC"] == pytest.approx(6.5)
    assert settings["MERGE_GAP_SEC"] == pytest.approx(1.2)
    assert settings["MIN_SCORE"] == pytest.approx(0.75)
    assert settings["TOP_K_HL_PER_SCENE"] == 5


def test_recognition_threshold_prefers_specific_section():
    cfg = {
        "search": {"near_match_threshold": 0.45},
        "recognition": {"SIM_THRESHOLD": 0.5},
    }

    settings = config_loader.get_recognition_settings(cfg)

    assert settings["SIM_THRESHOLD"] == pytest.approx(0.5)


def test_recognition_threshold_falls_back_to_search():
    cfg = {"search": {"near_match_threshold": "0.41"}}

    settings = config_loader.get_recognition_settings(cfg)

    assert settings["SIM_THRESHOLD"] == pytest.approx(0.41)


def test_frontend_settings_defaults(monkeypatch):
    monkeypatch.setattr(config_loader, "load_config", lambda: {})

    settings = config_loader.get_frontend_settings()

    assert settings["SEEK_PAD_SEC"] == pytest.approx(0.0)
    assert settings["PAUSE_TOLERANCE_SEC"] == pytest.approx(0.2)
    assert settings["MIN_VIEWABLE_SEC"] == pytest.approx(0.35)
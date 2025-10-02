import importlib
import io
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


def _stub_insightface(monkeypatch):
    app_module = types.ModuleType("insightface.app")

    class _DummyFaceAnalysis:  # pragma: no cover - simple import stub
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return []

    app_module.FaceAnalysis = _DummyFaceAnalysis
    insightface_module = types.ModuleType("insightface")
    insightface_module.app = app_module

    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)

    class _DummyDataFrame:
        empty = True

    pandas_module = types.ModuleType("pandas")
    pandas_module.read_parquet = lambda *args, **kwargs: _DummyDataFrame()
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)


def _prepare_api(monkeypatch):
    monkeypatch.setattr("utils.config_loader.load_config", lambda: {})
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    return importlib.import_module("api.main")


def test_upload_endpoint_accepts_video_and_metadata(tmp_path, monkeypatch):
    main = _prepare_api(monkeypatch)

    captured_payloads = []
    monkeypatch.setattr(main, "_run_pipeline_background", lambda payload=None: captured_payloads.append(payload))

    cache_calls = {"count": 0}

    def fake_clear():
        cache_calls["count"] += 1

    monkeypatch.setattr(main, "_clear_character_cache", fake_clear)

    video_content = b"video-bytes"
    with TestClient(main.app) as client:
        response = client.post(
            "/upload",
            data={"movie_id": "mv42", "source": "BluRay", "refresh": "true"},
            files={"video": ("sample.mp4", io.BytesIO(video_content), "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "scheduled"

    assert cache_calls["count"] == 1
    assert len(captured_payloads) == 1

    scheduled = captured_payloads[0]
    assert scheduled["movie_id"] == "mv42"
    assert scheduled["source"] == "BluRay"
    assert scheduled["refresh"] is True
    assert scheduled["filename"] == "sample.mp4"

    temp_path = scheduled["path"]
    assert temp_path
    assert Path(temp_path).exists()
    assert Path(temp_path).read_bytes() == video_content

    # Cleanup temp file created during the test.
    Path(temp_path).unlink(missing_ok=True)
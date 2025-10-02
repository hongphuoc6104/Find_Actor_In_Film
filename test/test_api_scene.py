import importlib
import io
import json
import os
import sys
import types
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




def test_scene_endpoint_serves_frame(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    clips_root = tmp_path / "clips"

    videos_root = tmp_path / "videos"
    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"

    frame_file = frames_root / movie_folder / frame_name
    frame_file.parent.mkdir(parents=True, exist_ok=True)
    frame_file.write_bytes(b"frame")
    second_frame = frames_root / movie_folder / "frame_0002.jpg"
    second_frame.write_bytes(b"frame2")

    clip_rel = os.path.join(movie_folder, "char1", "track1.mp4")
    clip_file = clips_root / clip_rel
    clip_file.parent.mkdir(parents=True, exist_ok=True)
    clip_file.write_bytes(b"clip-bytes")

    video_rel = os.path.join(movie_folder, "movie.mp4")
    video_basename = os.path.basename(video_rel)
    video_file = videos_root / video_basename
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"video-bytes")


    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "timestamp": 12.5,
                        "bbox": [10, 20, 110, 200],
                        "highlights": [
                            {"start": 12.5, "end": 17.5, "max_score": 0.93},
                        ],
                        "timeline": [
                            {
                                "frame": frame_name,
                                "frame_index": 5,
                                "bbox": [10, 20, 110, 200],
                                "timestamp": 12.5,
                            },
                            {
                                "frame": "frame_0002.jpg",
                                "frame_index": 6,
                                "bbox": [12, 22, 112, 202],
                                "timestamp": 13.0,
                                "duration": 0.5,
                            },
                        ],
                        "video_source": video_rel.replace(os.sep, "/"),
                        "video_start_timestamp": 12.5,
                        "video_end_timestamp": 17.5,
                        "start_time": 12.5,
                        "end_time": 17.5,
                        "duration": 5.0,
                        "width": 640,
                        "height": 360,
                    }
                ],
            }
        }
    }
    characters_path.write_text(json.dumps(characters), encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "video_root": str(videos_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 0,
                },
            )
            assert response.status_code == 200
            payload = response.json()

            scene = payload["scene"]
            assert scene["frame"].startswith(main.FRAMES_ROUTE)
            assert scene.get("frame_url", "").startswith(main.FRAMES_ROUTE)
            assert scene.get("frame_name") == frame_name
            expected_video_url = f"{main.VIDEOS_ROUTE}/{video_basename}"
            assert scene.get("video_url") == expected_video_url
            assert scene.get("video_source") == expected_video_url
            assert scene.get("start_time") == 12.5
            assert scene.get("duration") == 5.0
            assert scene.get("width") == 640
            assert scene.get("height") == 360
            assert scene.get("highlight_index") == 0
            assert scene.get("highlight_total") == 1
            assert scene.get("scene_index") == 0

            timeline = scene.get("timeline")
            assert isinstance(timeline, list) and len(timeline) == 2
            assert timeline[0]["bbox"] == [10, 20, 110, 200]
            assert timeline[0]["frame"].startswith(main.FRAMES_ROUTE)
            assert timeline[1]["timestamp"] == 13.0

            video_response = client.get(scene["video_url"])
            assert video_response.status_code == 200

            image_response = client.get(scene["frame"])
            assert image_response.status_code == 200
            assert image_response.content
    finally:
        main._clear_character_cache()


def test_scene_endpoint_normalises_windows_video_source(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    videos_root = tmp_path / "videos"

    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"
    windows_video_path = "nested\\folder\\movie.mp4"
    video_basename = os.path.basename(windows_video_path.replace("\\", "/"))

    (frames_root / movie_folder).mkdir(parents=True, exist_ok=True)
    (frames_root / movie_folder / frame_name).write_bytes(b"frame")

    video_file = videos_root / video_basename
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"video-bytes")

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "video_source": windows_video_path,
                        "timestamp": 5.0,
                        "highlights": [
                            {"start": 5.0, "end": 6.0, "max_score": 0.9},
                        ],
                    }
                ],
            }
        }
    }
    characters_path.write_text(json.dumps(characters), encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "video_root": str(videos_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 0,
                },
            )
            assert response.status_code == 200
            payload = response.json()

            scene = payload["scene"]
            expected_video_url = f"{main.VIDEOS_ROUTE}/{video_basename}"
            assert scene.get("video_url") == expected_video_url
            assert scene.get("video_source") == expected_video_url

            good_response = client.get(scene["video_url"])
            assert good_response.status_code == 200

            nested_path = windows_video_path.replace("\\", "/")
            bad_response = client.get(f"{main.VIDEOS_ROUTE}/{nested_path}")
            assert bad_response.status_code == 404
    finally:
        main._clear_character_cache()


def test_scene_endpoint_flattens_highlights(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    clips_root = tmp_path / "clips"

    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"
    second_frame = "frame_0002.jpg"

    (frames_root / movie_folder).mkdir(parents=True, exist_ok=True)
    (frames_root / movie_folder / frame_name).write_bytes(b"frame")
    (frames_root / movie_folder / second_frame).write_bytes(b"frame2")

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "scene_index": 4,
                        "bbox": [0, 0, 20, 40],
                        "highlights": [
                            {"start": 10, "end": 12.5, "max_score": 0.95},
                            {"start": 20, "end": 24, "max_score": 0.9},
                        ],
                    },
                    {
                        "frame": second_frame,
                        "bbox": [10, 10, 30, 50],
                        "highlights": [],
                    },
                ],
            }
        }
    }

    characters_path.write_text(json.dumps(characters), encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "scene_clips_root": str(clips_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)

    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    try:
        with TestClient(main.app) as client:
            first = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 0,
                },
            )
            assert first.status_code == 200
            payload = first.json()
            assert payload["scene_index"] == 0
            assert payload["total_scenes"] == 2
            assert payload["next_cursor"] == 1
            assert payload["has_more"] is True

            scene = payload["scene"]
            assert scene["scene_index"] == 0
            assert scene["highlight_index"] == 0
            assert scene["highlight_total"] == 2
            assert scene["source_scene_index"] == 4
            assert scene["scene_index"] == payload["scene_index"]
            assert len(scene["highlights"]) == 2
            first_highlight = scene["highlights"][0]
            assert first_highlight["start"] == 10.0
            assert first_highlight["end"] == 12.5
            assert first_highlight["duration"] == 2.5
            assert "similarity_percent" in first_highlight

            second = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 1,
                },
            )
            assert second.status_code == 200
            payload_two = second.json()
            assert payload_two["scene_index"] == 1
            assert payload_two["next_cursor"] is None
            assert payload_two["has_more"] is False

            scene_two = payload_two["scene"]
            assert scene_two["scene_index"] == 1
            assert scene_two["highlight_index"] == 1
            assert scene_two["highlight_total"] == 2
            assert scene_two["source_scene_index"] == 4
            assert scene_two["scene_index"] == payload_two["scene_index"]
            assert isinstance(scene_two["highlights"], list)
            assert len(scene_two["highlights"]) == 2
            second_highlight = scene_two["highlights"][1]
            assert second_highlight["start"] == 20.0
            assert second_highlight["end"] == 24.0
            assert second_highlight["duration"] == 4.0
            assert scene_two["start_time"] == second_highlight["start"]
            assert scene_two["duration"] > 0
    finally:
        main._clear_character_cache()

def test_scene_endpoint_handles_highlights_without_timeline(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"

    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"

    (frames_root / movie_folder).mkdir(parents=True, exist_ok=True)
    (frames_root / movie_folder / frame_name).write_bytes(b"frame")

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "scene_index": 2,
                        "video_start_timestamp": 10.0,
                        "video_end_timestamp": 13.0,
                        "highlights": [
                            {"start": 10.0, "end": 11.0, "max_score": 0.91},
                            {"start": 12.0, "end": 13.0, "max_score": 0.89},
                        ],
                    }
                ],
            }
        }
    }

    characters_path.write_text(json.dumps(characters), encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 0,
                },
            )
            assert response.status_code == 200
            payload = response.json()

            scene = payload["scene"]
            highlights = scene.get("highlights")
            assert isinstance(highlights, list)
            assert len(highlights) == 1

            merged = highlights[0]
            assert merged["start"] == 10.0
            assert merged["end"] == 13.0
            assert scene.get("highlight_total") == 1
            assert scene.get("highlight_index") == 0
    finally:
        main._clear_character_cache()


def test_scene_endpoint_falls_back_when_no_highlights(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"

    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"

    (frames_root / movie_folder).mkdir(parents=True, exist_ok=True)
    (frames_root / movie_folder / frame_name).write_bytes(b"frame")

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scene": {"frame": frame_name, "bbox": [0, 0, 20, 40]},
                "scenes": [
                    {"frame": frame_name, "bbox": [0, 0, 20, 40]},
                ],
            }
        }
    }

    characters_path.write_text(json.dumps(characters), encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/scene",
                json={
                    "movie_id": movie_key,
                    "character_id": character_key,
                    "cursor": 0,
                },
            )

            assert response.status_code == 200
            payload = response.json()

            assert payload["scene_index"] == 0
            assert payload["next_cursor"] is None
            assert payload["has_more"] is False
            assert payload["total_scenes"] == 1

            scene = payload["scene"]
            assert scene["scene_index"] == 0
            assert scene.get("highlight_total") == 0
            assert scene.get("highlights") == []
            assert isinstance(scene.get("highlights"), list)
            assert scene.get("source_scene_index") == 0
            assert scene["frame"].startswith(main.FRAMES_ROUTE)
            assert scene["frame_url"].startswith(main.FRAMES_ROUTE)
    finally:
        main._clear_character_cache()



def test_recognize_endpoint_includes_frame_urls(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    clips_root = tmp_path / "clips"
    videos_root = tmp_path / "videos"

    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0002.jpg"
    preview_rel = "clusters/example.jpg"
    video_rel = os.path.join(movie_folder, "movie.mp4")

    frame_file = frames_root / movie_folder / frame_name
    frame_file.parent.mkdir(parents=True, exist_ok=True)
    frame_file.write_bytes(b"frame-bytes")

    preview_file = previews_root / preview_rel
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    preview_file.write_bytes(b"preview")

    characters_path.write_text("{}", encoding="utf-8")

    config = {
        "storage": {
            "frames_root": str(frames_root),
            "cluster_previews_root": str(previews_root),
            "scene_clips_root": str(clips_root),
            "video_root": str(videos_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
    _stub_insightface(monkeypatch)
    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    def fake_recognize(image_path, top_k=None):
        assert image_path
        return {
            "is_unknown": False,
            "movies": [
                {
                    "movie_id": "movie1",
                    "movie": movie_folder,
                    "score": 0.9,
                    "characters": [
                        {
                            "movie_id": "movie1",
                            "movie": movie_folder,
                            "character_id": "char1",
                            "rep_image": {"movie": movie_folder, "frame": frame_name},
                            "scene": {
                                "frame": frame_name,
                                "bbox": [0, 0, 50, 60],
                                "clip_path": "MOVIE_FOLDER/char1/track1.mp4",
                                "clip_fps": 8.0,
                                "video_source": video_rel.replace(os.sep, "/"),
                                "start_time": 1.5,
                                "duration": 5.0,
                                "video_start_timestamp": 1.5,
                                "video_end_timestamp": 6.5,
                                "duration": 5.0,
                                "width": 640,
                                "height": 360,
                                "timeline": [
                                    {
                                        "frame": frame_name,
                                        "frame_index": 5,
                                        "bbox": [0, 0, 50, 60],
                                        "timestamp": 1.5,
                                    }
                                ],
                            },
                            "scenes": [
                                {
                                    "frame": frame_name,
                                    "bbox": [0, 0, 50, 60],
                                    "video_source": video_rel.replace(os.sep, "/"),
                                    "start_time": 1.5,
                                    "duration": 5.0,
                                    "width": 640,
                                    "height": 360,
                                    "timeline": [
                                        {
                                            "frame": frame_name,
                                            "frame_index": 5,
                                            "bbox": [0, 0, 50, 60],
                                            "timestamp": 1.5,
                                        }
                                    ],
                                }
                            ],
                            "previews": [
                                {
                                    "movie": movie_folder,
                                    "frame": frame_name,
                                    "preview_image": preview_rel,
                                    "annotated_image": preview_rel,
                                }
                            ],
                            "preview_paths": [preview_rel],
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(main, "recognize", fake_recognize)

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/recognize",
                files={"image": ("sample.jpg", io.BytesIO(b"data"), "image/jpeg")},
            )
            assert response.status_code == 200
            payload = response.json()

            assert payload["movies"]
            movie = payload["movies"][0]
            assert movie["characters"]
            character = movie["characters"][0]

            rep = character["rep_image"]
            assert rep["frame"].startswith(main.FRAMES_ROUTE)
            assert rep["frame_url"] == rep["frame"]
            assert rep["frame_name"] == frame_name
            assert rep["highlights"] == []
            assert rep["highlight_total"] == 0

            scene = character["scene"]
            assert scene["frame"].startswith(main.FRAMES_ROUTE)
            assert scene["frame_url"] == scene["frame"]
            assert scene["frame_name"] == frame_name
            assert scene["video_url"].startswith(main.VIDEOS_ROUTE)
            assert scene["start_time"] == 1.5
            assert scene["timeline"][0]["bbox"] == [0, 0, 50, 60]
            assert scene["highlights"] == []
            assert scene["highlight_total"] == 0

            scenes = character.get("scenes")
            assert scenes and scenes[0]["frame"].startswith(main.FRAMES_ROUTE)
            assert scenes[0]["frame_name"] == frame_name
            assert scenes[0]["video_url"].startswith(main.VIDEOS_ROUTE)
            assert scenes[0]["highlights"] == []
            assert scenes[0]["highlight_total"] == 0

            preview_entry = character["previews"][0]
            assert preview_entry["frame"].startswith(main.FRAMES_ROUTE)
            assert preview_entry["frame_url"] == preview_entry["frame"]
            assert preview_entry["frame_name"] == frame_name
            assert preview_entry["preview_image"].startswith(main.PREVIEWS_ROUTE)

            assert character["preview_paths"][0].startswith(main.PREVIEWS_ROUTE)
    finally:
        main._clear_character_cache()
import importlib
import io
import json
import os
import sys

from fastapi.testclient import TestClient


def test_scene_endpoint_serves_frame(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    clips_root = tmp_path / "clips"

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

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "timestamp": 12.5,
                        "bbox": [10, 20, 110, 200],
                        "timeline": [
                            {
                                "frame": frame_name,
                                "frame_index": 5,
                                "bbox": [10, 20, 110, 200],
                                "clip_offset": 0.0,
                            },
                            {
                                "frame": "frame_0002.jpg",
                                "frame_index": 6,
                                "bbox": [12, 22, 112, 202],
                                "clip_offset": 0.5,
                                "duration": 0.5,
                            },
                        ],
                        "clip_path": clip_rel.replace(os.sep, "/"),
                        "clip_fps": 8.0,
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
            "scene_clips_root": str(clips_root),
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
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
            assert scene.get("clip").startswith(main.CLIPS_ROUTE)
            assert scene.get("clip_url") == scene.get("clip")
            assert scene.get("clip_path") == clip_rel.replace(os.sep, "/")
            assert scene.get("clip_fps") == 8.0
            assert scene.get("duration") == 5.0
            assert scene.get("width") == 640
            assert scene.get("height") == 360

            timeline = scene.get("timeline")
            assert isinstance(timeline, list) and len(timeline) == 2
            assert timeline[0]["bbox"] == [10, 20, 110, 200]
            assert timeline[0]["frame"].startswith(main.FRAMES_ROUTE)
            assert timeline[1]["clip_offset"] == 0.5

            clip_response = client.get(scene["clip"])
            assert clip_response.status_code == 200

            image_response = client.get(scene["frame"])
            assert image_response.status_code == 200
            assert image_response.content
    finally:
        main._clear_character_cache()


def test_recognize_endpoint_includes_frame_urls(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"
    clips_root = tmp_path / "clips"

    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0002.jpg"
    preview_rel = "clusters/example.jpg"

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
            "characters_json": str(characters_path),
        }
    }

    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)
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
                                "duration": 5.0,
                                "width": 640,
                                "height": 360,
                                "timeline": [
                                    {
                                        "frame": frame_name,
                                        "frame_index": 5,
                                        "bbox": [0, 0, 50, 60],
                                        "clip_offset": 0.0,
                                    }
                                ],
                            },
                            "scenes": [
                                {
                                    "frame": frame_name,
                                    "bbox": [0, 0, 50, 60],
                                    "clip_path": "MOVIE_FOLDER/char1/track1.mp4",
                                    "clip_fps": 8.0,
                                    "duration": 5.0,
                                    "width": 640,
                                    "height": 360,
                                    "timeline": [
                                        {
                                            "frame": frame_name,
                                            "frame_index": 5,
                                            "bbox": [0, 0, 50, 60],
                                            "clip_offset": 0.0,
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

            scene = character["scene"]
            assert scene["frame"].startswith(main.FRAMES_ROUTE)
            assert scene["frame_url"] == scene["frame"]
            assert scene["frame_name"] == frame_name
            assert scene["clip"].startswith(main.CLIPS_ROUTE)
            assert scene["clip_url"] == scene["clip"]
            assert scene["timeline"][0]["bbox"] == [0, 0, 50, 60]

            scenes = character.get("scenes")
            assert scenes and scenes[0]["frame"].startswith(main.FRAMES_ROUTE)
            assert scenes[0]["frame_name"] == frame_name
            assert scenes[0]["clip"].startswith(main.CLIPS_ROUTE)

            preview_entry = character["previews"][0]
            assert preview_entry["frame"].startswith(main.FRAMES_ROUTE)
            assert preview_entry["frame_url"] == preview_entry["frame"]
            assert preview_entry["frame_name"] == frame_name
            assert preview_entry["preview_image"].startswith(main.PREVIEWS_ROUTE)

            assert character["preview_paths"][0].startswith(main.PREVIEWS_ROUTE)
    finally:
        main._clear_character_cache()
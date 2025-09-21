import importlib
import json
import sys

from fastapi.testclient import TestClient


def test_scene_endpoint_serves_frame(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    previews_root = tmp_path / "previews"
    characters_path = tmp_path / "characters.json"

    movie_key = "movie1"
    character_key = "char1"
    movie_folder = "MOVIE_FOLDER"
    frame_name = "frame_0001.jpg"

    frame_file = frames_root / movie_folder / frame_name
    frame_file.parent.mkdir(parents=True, exist_ok=True)
    frame_file.write_bytes(b"frame")

    characters = {
        movie_key: {
            character_key: {
                "movie": movie_folder,
                "scenes": [
                    {
                        "frame": frame_name,
                        "timestamp": 12.5,
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

            image_response = client.get(scene["frame"])
            assert image_response.status_code == 200
            assert image_response.content
    finally:
        main._clear_character_cache()

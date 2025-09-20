import sys
import types

import pytest

import cli.find_actor as fa


class DummyArray(list):
    def reshape(self, a, b):
        return self


def test_run_recognized(monkeypatch, tmp_path):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: object())
    monkeypatch.setattr(fa, "cv2", stub_cv2)

    dummy_np = types.SimpleNamespace(array=lambda x, dtype=None: DummyArray(x), float32=float)
    monkeypatch.setattr(fa, "np", dummy_np)

    img_path = tmp_path / "img.jpg"
    img_path.write_text("dummy")

    def dummy_search_actor(image_path, k, min_count, return_emb):
        assert k == 20
        assert min_count == 0
        assert return_emb is True

        def _search_func(emb, top_k, min_count):
            assert top_k == 20
            assert min_count == 0
            return {
                "0": [
                    {"character_id": "1", "movie": "m1", "distance": 0.9},
                    {"character_id": "2", "movie": "m1", "distance": 0.8},
                ]
            }
        return {"embedding": [0.0, 0.0], "search_func": _search_func}

    monkeypatch.setattr(fa, "search_actor", dummy_search_actor)
    res = fa.run(str(img_path), 0.5, 1.1, 0.05)
    assert res["recognized"] is True
    assert "0" in res["per_movie"]
    matches = res["per_movie"]["0"]["matches"]
    assert [m["character_id"] for m in matches] == ["1", "2"]


def test_run_unknown(monkeypatch, tmp_path):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: object())
    monkeypatch.setattr(fa, "cv2", stub_cv2)

    dummy_np = types.SimpleNamespace(
        array=lambda x, dtype=None: DummyArray(x), float32=float
    )
    monkeypatch.setattr(fa, "np", dummy_np)

    img_path = tmp_path / "img.jpg"
    img_path.write_text("dummy")

    def dummy_search_actor(image_path, k, min_count, return_emb):
        assert k == 20
        assert min_count == 0
        assert return_emb is True

        def _search_func(emb, top_k, min_count):
            assert top_k == 20
            assert min_count == 0
            return {
                "0": [
                    {"character_id": "1", "movie": "m1", "distance": 0.4},
                    {"character_id": "2", "movie": "m1", "distance": 0.39},
                ]
            }
        return {"embedding": [0.0, 0.0], "search_func": _search_func}

    monkeypatch.setattr(fa, "search_actor", dummy_search_actor)
    res = fa.run(str(img_path), 0.5, 1.1, 0.05)
    assert res["recognized"] is False
    assert "0" in res["per_movie"]
    assert len(res["per_movie"]["0"]["matches"]) == 2


def test_run_missing_image(monkeypatch):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: None)
    monkeypatch.setattr(fa, "cv2", stub_cv2)
    res = fa.run("missing.jpg", 0.5, 1.1, 0.05)
    assert "error" in res


def test_main_merges_movies(monkeypatch, capsys):
    dummy_config_module = types.ModuleType("config_loader")
    dummy_config_module.load_config = lambda: {"search": {}, "storage": {}}
    monkeypatch.setitem(sys.modules, "utils.config_loader", dummy_config_module)

    args = types.SimpleNamespace(
        image="image.jpg",
        sim_threshold=0.5,
        ratio_threshold=1.1,
        margin_threshold=0.05,
        top_k=20,
        min_count=0,
    )
    monkeypatch.setattr(
        fa.argparse.ArgumentParser, "parse_args", lambda self: args
    )


    monkeypatch.setattr(
        fa,
        "run",
        lambda *call_args, **call_kwargs: {
            "recognized": True,
            "per_movie": {
                "0": {
                    "movie": "Film 1",
                    "recognized": True,
                    "matches": [
                        {"character_id": "A", "rep_image": {}, "preview_paths": []}
                    ],
                },
                "1": {
                    "movie": "Film 2",
                    "recognized": True,
                    "matches": [
                        {"character_id": "A", "rep_image": {}, "preview_paths": []}
                    ],
                },
            },
        },
    )

    fa.main()
    output = capsys.readouterr().out
    assert "✅ Recognized for Film 1" in output
    assert "✅ Recognized for Film 2" in output


def test_main_outputs_suggestions(monkeypatch, capsys):
    dummy_config_module = types.ModuleType("config_loader")
    dummy_config_module.load_config = lambda: {"search": {}, "storage": {}}
    monkeypatch.setitem(sys.modules, "utils.config_loader", dummy_config_module)

    args = types.SimpleNamespace(
        image="image.jpg",
        sim_threshold=0.5,
        ratio_threshold=1.1,
        margin_threshold=0.05,
        top_k=20,
        min_count=0,
    )
    monkeypatch.setattr(
        fa.argparse.ArgumentParser, "parse_args", lambda self: args
    )

    monkeypatch.setattr(
        fa,
        "run",
        lambda *call_args, **call_kwargs: {
            "recognized": False,
            "per_movie": {
                "0": {
                    "movie": "Film X",
                    "recognized": False,
                    "matches": [
                        {
                            "character_id": "1",
                            "distance": 0.4,
                            "preview_paths": ["preview1.jpg"],
                        },
                        {
                            "character_id": "2",
                            "distance": 0.35,
                            "preview_paths": ["preview2.jpg"],
                        },
                    ],
                }
            },
        },
    )

    fa.main()
    output = capsys.readouterr().out
    assert "⚠️ Suggestions for Film X" in output
    assert "Actor 1 - score: 0.4000" in output
    assert "Actor 2 - score: 0.3500" in output
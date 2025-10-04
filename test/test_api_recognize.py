import asyncio
import importlib
import io
import logging
import os
import sys
import types
from typing import List

from fastapi import UploadFile


def test_recognize_endpoint_warns_when_temp_cleanup_fails(monkeypatch, caplog):
    config = {"storage": {}}
    monkeypatch.setattr("utils.config_loader.load_config", lambda: config)

    recorded_paths: List[str] = []

    recognition_stub = types.ModuleType("services.recognition")

    def _recognize(image_path: str):
        recorded_paths.append(image_path)
        return {"movies": []}

    recognition_stub.recognize = _recognize
    monkeypatch.setitem(sys.modules, "services.recognition", recognition_stub)

    sys.modules.pop("api.main", None)
    main = importlib.import_module("api.main")

    deleted_paths: List[str] = []
    real_unlink = os.unlink

    def _raise_unlink(path: str) -> None:
        deleted_paths.append(path)
        raise OSError("cannot remove file")

    monkeypatch.setattr(main.os, "unlink", _raise_unlink)

    upload = UploadFile(filename="face.jpg", file=io.BytesIO(b"fake-bytes"))

    async def _invoke() -> dict:
        return await main.recognize_endpoint(upload)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(_invoke())

    assert result == {"movies": []}
    assert recorded_paths, "recognize should have been invoked"
    assert deleted_paths, "cleanup should attempt to remove the temporary file"

    warning_messages = [record.getMessage() for record in caplog.records]
    assert any(deleted_paths[0] in message for message in warning_messages)
    assert any("cannot remove file" in message for message in warning_messages)

    for path in deleted_paths:
        real_unlink(path)
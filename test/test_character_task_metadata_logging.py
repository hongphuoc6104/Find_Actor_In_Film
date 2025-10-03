import logging
import sys
import types

# Provide lightweight stubs for heavy optional dependencies used in tasks.character_task
pandas_stub = types.ModuleType("pandas")
pandas_stub.read_parquet = lambda *args, **kwargs: None
pandas_stub.concat = lambda *args, **kwargs: None
pandas_stub.DataFrame = type("DataFrame", (), {})
pandas_stub.Series = type("Series", (), {})
sys.modules.setdefault("pandas", pandas_stub)

prefect_stub = types.ModuleType("prefect")

def _task(func=None, **_kwargs):
    if func is None:
        return lambda f: f
    return func

prefect_stub.task = _task
sys.modules.setdefault("prefect", prefect_stub)

sklearn_stub = types.ModuleType("sklearn")
sklearn_cluster_stub = types.ModuleType("sklearn.cluster")
class _AgglomerativeClustering:  # pragma: no cover - stub class
    pass

sklearn_cluster_stub.AgglomerativeClustering = _AgglomerativeClustering
sklearn_stub.cluster = sklearn_cluster_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.cluster", sklearn_cluster_stub)

cv2_stub = types.ModuleType("cv2")
sys.modules.setdefault("cv2", cv2_stub)

from tasks.character_task import _load_cluster_metadata


def test_load_cluster_metadata_logs_error(tmp_path, caplog):
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text("{invalid}")

    with caplog.at_level(logging.WARNING, logger="tasks.character_task"):
        result = _load_cluster_metadata(str(meta_file), movie_id=123, cluster_id="456")

    assert result == []
    assert any(
        "movie_id=123" in record.getMessage()
        and "cluster_id=456" in record.getMessage()
        and str(meta_file) in record.getMessage()
        for record in caplog.records
    )
# tasks/build_warehouse_task.py
from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List

import pandas as pd
from prefect import task

from utils.config_loader import load_config


# -------------------------- IO helpers -------------------------- #

def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_to_parquet(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


def _movie_from_path(p: str) -> str:
    """Suy ra tên phim từ đường dẫn parquet per-movie."""
    base = os.path.basename(p)
    name = os.path.splitext(base)[0]
    return name


# -------------------- Source discovery (single-movie aware) -------------------- #

def _list_embedding_files(cfg: dict) -> List[str]:
    """
    Trả về danh sách parquet per-movie.
    - Ưu tiên metadata.json nếu có
    - Fallback glob Data/embeddings/
    - Nếu ENV FS_ACTIVE_MOVIE được set -> chỉ trả về file của phim đó
    """
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json")
    root = storage.get("embeddings_folder_per_movie") or "Data/embeddings"

    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    files: List[str] = []

    # 1) metadata.json
    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        for movie_name, info in meta.items():
            if active_movie and movie_name != active_movie:
                continue
            p = (info or {}).get("embedding_file_path")
            if isinstance(p, str) and os.path.exists(p):
                files.append(p)

    # 2) fallback glob
    if not files:
        candidates = sorted(
            glob(os.path.join(root, "*.parquet"))
            + glob(os.path.join(root, "**/*.parquet"), recursive=True)
        )
        if active_movie:
            files = [p for p in candidates if _movie_from_path(p) == active_movie]
        else:
            files = candidates

    return files


# -------------------------- Mapping & cleanup -------------------------- #

def _stable_movie_id_mapping(files: List[str], dfs: List[pd.DataFrame], prior_map: Dict[str, int] | None) -> Dict[str, int]:
    """
    Trả về map tên_phim → movie_id ổn định.
    Ưu tiên:
      - prior_map (map cũ trong metadata) nếu có
      - movie_id có sẵn & nhất quán trong parquet
      - nếu vẫn thiếu -> gán id mới theo thứ tự tên
    """
    movie_to_id: Dict[str, int] = dict(prior_map or {})

    # Thu thập tên phim từ danh sách file
    names = {(_movie_from_path(p)) for p in files}

    # 1) nếu DF có movie_id nhất quán, dùng lại
    for p, df in zip(files, dfs):
        name = _movie_from_path(p)
        if name in movie_to_id:
            continue
        if "movie_id" in df.columns:
            vals = df["movie_id"].dropna().unique().tolist()
            if len(vals) == 1:
                movie_to_id[name] = int(vals[0])

    # 2) gán mới cho tên chưa có id
    if names - set(movie_to_id.keys()):
        # không phá id cũ; chỉ cộng tiếp
        next_id = 0
        if movie_to_id:
            next_id = max(movie_to_id.values()) + 1
        for name in sorted(names):
            if name not in movie_to_id:
                movie_to_id[name] = next_id
                next_id += 1

    return movie_to_id


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Khử trùng lặp bản ghi theo khoá ưu tiên."""
    if "global_id" in df.columns:
        return df.drop_duplicates(subset=["global_id"], keep="first")
    keys: List[str] = []
    for k in ("movie", "track_id", "frame"):
        if k in df.columns:
            keys.append(k)
    if keys:
        return df.drop_duplicates(subset=keys, keep="first")
    return df.drop_duplicates(keep="first")


# ------------------------------- Task -------------------------------- #

@task(name="Build Warehouse Task")
def build_warehouse_task() -> str:
    """
    Gom parquet embedding per-movie → warehouse/parquet/embeddings.parquet
    - **Tôn trọng ENV FS_ACTIVE_MOVIE**: nếu set, chỉ gom của phim đó.
    - Bổ sung 'movie' (str) & 'movie_id' (ổn định).
    - Khử trùng lặp.
    - Ghi atomic.
    - Cập nhật _generated.movie_id_map trong metadata.json (merge, không xoá id cũ).
    """
    print("\n--- Building Warehouse Embeddings ---")
    cfg = load_config()
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    out_path = storage.get("warehouse_embeddings") or "warehouse/parquet/embeddings.parquet"
    meta_path = storage.get("metadata_json")

    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Warehouse] Single-movie mode → '{active_movie}'")

    files = _list_embedding_files(cfg)
    if not files:
        if active_movie:
            print(f"[Warehouse] No parquet found for movie='{active_movie}'. Skipping.")
        else:
            print("[Warehouse] No per-movie parquet files found. Skipping.")
        return out_path

    # Đọc trước prior movie_id_map (nếu có) để giữ ổn định id
    prior_map: Dict[str, int] = {}
    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        prior_map = ((meta.get("_generated") or {}).get("movie_id_map")) or {}

    dfs: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df is None or df.empty:
                continue
            # đảm bảo có 'movie'
            if "movie" not in df.columns:
                df = df.assign(movie=_movie_from_path(p))
            else:
                df["movie"] = df["movie"].astype(str)
            # nếu single-movie nhưng file chứa nhiều movie (không mong đợi) → lọc
            if active_movie:
                df = df[df["movie"] == active_movie]
                if df.empty:
                    continue
            dfs.append(df)
        except Exception as e:
            print(f"[Warehouse] Skip file (read error): {p} ({e})")

    if not dfs:
        print("[Warehouse] All sources empty/unreadable after filtering. Skipping.")
        return out_path

    # Lập mapping movie_id ổn định (giữ id cũ nếu có)
    movie_to_id = _stable_movie_id_mapping(files, dfs, prior_map)

    # Gán movie_id theo mapping
    patched: List[pd.DataFrame] = []
    for p, df in zip(files, dfs):
        if df.empty:
            continue
        movie_name = str(df["movie"].iloc[0]) if "movie" in df.columns else _movie_from_path(p)
        mid = movie_to_id.get(movie_name)
        if "movie_id" not in df.columns:
            df = df.assign(movie_id=int(mid))
        else:
            df["movie_id"] = int(mid)
        patched.append(df)

    # Hợp nhất & khử trùng lặp
    whole = pd.concat(patched, axis=0, ignore_index=True, sort=True)
    before = len(whole)
    whole = _dedup(whole)
    after = len(whole)
    if after < before:
        print(f"[Warehouse] De-duplicated: {before} -> {after}")

    # Ghi atomic
    _atomic_to_parquet(whole, out_path)
    print(f"[Warehouse] Saved {len(whole)} rows to {out_path}")

    # Cập nhật movie_id_map trong metadata (merge, không xoá id cũ)
    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        meta.setdefault("_generated", {})
        merged_map = dict(((meta["_generated"].get("movie_id_map")) or {}))
        merged_map.update(movie_to_id)  # merge non-destructive
        meta["_generated"]["movie_id_map"] = merged_map
        tmp = f"{meta_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, meta_path)
            print(f"[Warehouse] Updated movie_id_map in {meta_path}")
        except Exception as e:
            print(f"[Warehouse] Could not update movie_id_map: {e}")

    return out_path

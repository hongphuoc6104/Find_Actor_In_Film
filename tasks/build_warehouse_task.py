# tasks/build_warehouse_task.py
from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Tuple  # <-- Đã thêm Tuple

import pandas as pd
from prefect import task

from utils.config_loader import load_config


# -------------------------- IO helpers (Không thay đổi) -------------------------- #
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
    base = os.path.basename(p)
    name = os.path.splitext(base)[0]
    return name


# -------------------- Source discovery (Không thay đổi) -------------------- #
def _list_embedding_files(cfg: dict) -> List[str]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json")
    root = storage.get("embeddings_folder_per_movie") or "Data/embeddings"
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    files: List[str] = []

    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        movie_info = meta.get(active_movie, {})
        if active_movie and movie_info.get("embedding_file_path"):
            p = movie_info["embedding_file_path"]
            if os.path.exists(p):
                files.append(p)

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


# -------------------------- Mapping & cleanup (Không thay đổi) -------------------------- #
def _stable_movie_id_mapping(files: List[str], dfs: List[pd.DataFrame], prior_map: Dict[str, int] | None) -> Dict[
    str, int]:
    movie_to_id: Dict[str, int] = dict(prior_map or {})
    names = {(_movie_from_path(p)) for p in files}
    for p, df in zip(files, dfs):
        name = _movie_from_path(p)
        if name in movie_to_id: continue
        if "movie_id" in df.columns:
            vals = df["movie_id"].dropna().unique().tolist()
            if len(vals) == 1:
                movie_to_id[name] = int(vals[0])
    if names - set(movie_to_id.keys()):
        next_id = 0
        if movie_to_id: next_id = max(movie_to_id.values()) + 1
        for name in sorted(names):
            if name not in movie_to_id:
                movie_to_id[name] = next_id
                next_id += 1
    return movie_to_id


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if "global_id" in df.columns:
        return df.drop_duplicates(subset=["global_id"], keep="first")
    keys: List[str] = []
    for k in ("movie", "track_id", "frame"):
        if k in df.columns:
            keys.append(k)
    if keys:
        return df.drop_duplicates(subset=keys, keep="first")
    return df.drop_duplicates(keep="first")


# ------------------------------- Task (Đã cập nhật) -------------------------------- #

@task(name="Build Warehouse Task")
def build_warehouse_task() -> Tuple[str, int]:  # <-- CẬP NHẬT 1: Thay đổi kiểu trả về
    """
    Gom parquet embedding per-movie → warehouse/parquet/embeddings.parquet
    Trả về (đường dẫn file, số dòng đã xử lý cho phim đang hoạt động).
    """
    print("\n--- Building Warehouse Embeddings ---")
    cfg = load_config()
    storage = cfg.get("storage", {})
    out_path = storage.get("warehouse_embeddings", "warehouse/parquet/embeddings.parquet")
    meta_path = storage.get("metadata_json")

    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Warehouse] Single-movie mode → '{active_movie}'")

    files = _list_embedding_files(cfg)
    if not files:
        print(f"[Warehouse] No parquet found for movie='{active_movie}'. Skipping.")
        # CẬP NHẬT 2: Trả về tuple đúng định dạng khi không có file
        return out_path, 0

    prior_map: Dict[str, int] = {}
    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        prior_map = meta.get("_generated", {}).get("movie_id_map", {})

    dfs: List[pd.DataFrame] = []
    # CẬP NHẬT 3: Biến đếm số dòng cho phim đang hoạt động
    processed_rows_count = 0
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df is None or df.empty: continue

            if "movie" not in df.columns:
                df = df.assign(movie=_movie_from_path(p))
            else:
                df["movie"] = df["movie"].astype(str)

            # Chỉ xử lý dữ liệu của phim đang hoạt động và cập nhật biến đếm
            df_movie = df[df["movie"] == active_movie]
            if not df_movie.empty:
                dfs.append(df_movie)
                processed_rows_count += len(df_movie)
        except Exception as e:
            print(f"[Warehouse] Skip file (read error): {p} ({e})")

    if not dfs:
        print("[Warehouse] No data found for the active movie after filtering. Skipping.")
        # CẬP NHẬT 4: Trả về tuple đúng định dạng khi không có dữ liệu
        return out_path, 0

    movie_to_id = _stable_movie_id_mapping(files, dfs, prior_map)

    patched: List[pd.DataFrame] = []
    for df in dfs:
        movie_name = df["movie"].iloc[0]
        mid = movie_to_id.get(movie_name)
        if mid is not None:
            df = df.copy()  # Tránh SettingWithCopyWarning
            if "movie_id" not in df.columns:
                df['movie_id'] = int(mid)
            else:
                df["movie_id"] = int(mid)
            patched.append(df)

    # Nếu không có dữ liệu sau khi gán id, tạo dataframe rỗng
    whole = pd.concat(patched, axis=0, ignore_index=True) if patched else pd.DataFrame()
    before = len(whole)
    whole = _dedup(whole)
    after = len(whole)
    if after < before:
        print(f"[Warehouse] De-duplicated: {before} -> {after}")

    _atomic_to_parquet(whole, out_path)
    print(f"[Warehouse] Saved {len(whole)} rows to {out_path}")

    # Cập nhật metadata
    if meta_path and os.path.exists(meta_path):
        meta = _read_json(meta_path) or {}
        meta.setdefault("_generated", {})
        merged_map = meta["_generated"].get("movie_id_map", {})
        merged_map.update(movie_to_id)
        meta["_generated"]["movie_id_map"] = merged_map
        tmp = f"{meta_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, meta_path)
            print(f"[Warehouse] Updated movie_id_map in {meta_path}")
        except Exception as e:
            print(f"[Warehouse] Could not update movie_id_map: {e}")

    # CẬP NHẬT 5: Trả về tuple đúng định dạng
    return out_path, processed_rows_count
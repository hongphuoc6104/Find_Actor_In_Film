#!/usr/bin/env python3
"""
Xóa hoàn toàn dữ liệu của 1 video từ tất cả storage.
Usage: python delete_movie.py MOVIE_NAME
"""

import os
import sys
import shutil
import json
from pathlib import Path


def delete_movie_data(movie_name: str, dry_run: bool = False) -> dict:
    """
    Xóa tất cả dữ liệu liên quan đến 1 video.
    
    Args:
        movie_name: Tên video (không có extension)
        dry_run: Nếu True, chỉ liệt kê mà không xóa
        
    Returns:
        dict với danh sách files/folders đã xóa
    """
    deleted = {"files": [], "folders": [], "errors": []}
    
    # 1. Video file
    video_dir = Path("Data/video")
    for ext in [".mp4", ".avi", ".mkv", ".mov"]:
        video_file = video_dir / f"{movie_name}{ext}"
        if video_file.exists():
            if not dry_run:
                video_file.unlink()
            deleted["files"].append(str(video_file))
    
    # 2. Frames folder
    frames_dir = Path("Data/frames") / movie_name
    if frames_dir.exists():
        if not dry_run:
            shutil.rmtree(frames_dir)
        deleted["folders"].append(str(frames_dir))
    
    # 3. Face crops folder
    crops_dir = Path("Data/face_crops") / movie_name
    if crops_dir.exists():
        if not dry_run:
            shutil.rmtree(crops_dir)
        deleted["folders"].append(str(crops_dir))
    
    # 4. Embeddings parquet
    emb_file = Path("Data/embeddings") / f"{movie_name}.parquet"
    if emb_file.exists():
        if not dry_run:
            emb_file.unlink()
        deleted["files"].append(str(emb_file))
    
    # 5. Cluster parquet
    cluster_file = Path("warehouse/parquet") / f"{movie_name}_clusters.parquet"
    if cluster_file.exists():
        if not dry_run:
            cluster_file.unlink()
        deleted["files"].append(str(cluster_file))
    
    # 6. Cluster previews folder
    preview_dir = Path("warehouse/cluster_previews") / movie_name
    if preview_dir.exists():
        if not dry_run:
            shutil.rmtree(preview_dir)
        deleted["folders"].append(str(preview_dir))
    
    # 7. Characters.json - remove movie entry
    chars_file = Path("warehouse/characters.json")
    if chars_file.exists():
        try:
            with open(chars_file, "r", encoding="utf-8") as f:
                chars = json.load(f)
            if movie_name in chars:
                if not dry_run:
                    del chars[movie_name]
                    with open(chars_file, "w", encoding="utf-8") as f:
                        json.dump(chars, f, ensure_ascii=False, indent=2)
                deleted["files"].append(f"{chars_file} (entry: {movie_name})")
        except Exception as e:
            deleted["errors"].append(f"Error updating characters.json: {e}")
    
    # 8. Metadata.json - remove movie entry
    meta_file = Path("Data/metadata.json")
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if movie_name in meta:
                if not dry_run:
                    del meta[movie_name]
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                deleted["files"].append(f"{meta_file} (entry: {movie_name})")
        except Exception as e:
            deleted["errors"].append(f"Error updating metadata.json: {e}")
    
    # Clear search cache
    if not dry_run:
        try:
            from utils.search_actor import _load_characters_json
            _load_characters_json.cache_clear()
        except:
            pass
    
    return deleted


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_movie.py MOVIE_NAME [--dry-run]")
        sys.exit(1)
    
    movie_name = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Deleting all data for: {movie_name}")
    print("=" * 50)
    
    result = delete_movie_data(movie_name, dry_run=dry_run)
    
    if result["files"]:
        print("\n📄 Files:")
        for f in result["files"]:
            print(f"  - {f}")
    
    if result["folders"]:
        print("\n📁 Folders:")
        for f in result["folders"]:
            print(f"  - {f}")
    
    if result["errors"]:
        print("\n❌ Errors:")
        for e in result["errors"]:
            print(f"  - {e}")
    
    if not result["files"] and not result["folders"]:
        print("⚠️  No data found for this movie.")
    else:
        action = "Would delete" if dry_run else "Deleted"
        print(f"\n✅ {action}: {len(result['files'])} files, {len(result['folders'])} folders")

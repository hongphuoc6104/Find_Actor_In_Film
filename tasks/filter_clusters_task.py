from __future__ import annotations

import json
from typing import Dict

import pandas as pd
from prefect import task


@task(name="Filter Clusters Task")
def filter_clusters_task(
    clusters: pd.DataFrame,
    characters_path: str,
    cfg: dict,
):
    """Remove low-quality clusters on a per-movie basis."""

    if clusters.empty or "final_character_id" not in clusters.columns:
        return characters_path

    filter_cfg = cfg.get("filter_clusters", {})
    min_size = int(filter_cfg.get("min_size", 3))
    min_det = float(filter_cfg.get("min_det", 0.6))
    min_frames = int(filter_cfg.get("min_frames", 5))

    group_cols = ["movie_id", "final_character_id"]
    available_cols = set(clusters.columns)
    if "movie_id" not in available_cols:
        clusters = clusters.assign(movie_id=0)

    stats = (
        clusters.groupby(group_cols)
        .agg(
            size=("final_character_id", "size"),
            mean_det=("det_score", "mean"),
            frames=("frame", "nunique") if "frame" in clusters.columns else ("final_character_id", "size"),
        )
        .reset_index()
    )

    valid_pairs = {
        (str(int(row.movie_id)), str(row.final_character_id))
        for row in stats.itertuples()
        if row.size >= min_size and row.mean_det >= min_det and row.frames >= min_frames
    }

    with open(characters_path, "r", encoding="utf-8") as f:
        characters: Dict[str, Dict[str, dict]] = json.load(f)

    cleaned: Dict[str, Dict[str, dict]] = {}
    for movie_id, char_map in characters.items():
        retained = {
            char_id: data
            for char_id, data in char_map.items()
            if (movie_id, char_id) in valid_pairs
        }
        if retained:
            cleaned[movie_id] = retained

    with open(characters_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    return characters_path
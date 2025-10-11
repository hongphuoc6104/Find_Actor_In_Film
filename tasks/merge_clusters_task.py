# tasks/merge_clusters_task.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from prefect import task


@task(name="Merge Clusters Task")
def merge_clusters_task(
    sim_threshold: float = 0.75,
    clusters_parquet: Optional[str] = None,
    output_parquet: Optional[str] = None,
) -> str:
    """
    Gộp cụm trong phạm vi 1 phim (within-movie merge).
    Hiện tại triển khai theo hướng an toàn/no-op (chưa đổi nhãn), đảm bảo pipeline không vỡ:
      - Đọc clusters từ warehouse/parquet/clusters.parquet
      - Nếu FS_ACTIVE_MOVIE có giá trị -> chỉ xử lý phim đó
      - (Placeholder) Có thể thêm tiêu chí gộp sau: centroid cosine sim >= sim_threshold
      - Ghi ra warehouse/parquet/clusters_merged.parquet
    Trả về đường dẫn file parquet đã ghi.
    """
    # Đường dẫn mặc định
    wh_dir = Path("warehouse") / "parquet"
    wh_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(clusters_parquet) if clusters_parquet else (wh_dir / "clusters.parquet")
    dst_path = Path(output_parquet) if output_parquet else (wh_dir / "clusters_merged.parquet")

    if not src_path.exists():
        raise FileNotFoundError(f"[Merge] Không tìm thấy file cụm: {src_path}")

    print("\n--- Starting Merge Clusters Task (within-movie only) ---")
    print(f"[Merge] Input : {src_path}")
    print(f"[Merge] Output: {dst_path}")
    print(f"[Merge] sim_threshold: {sim_threshold}")

    # Đọc dữ liệu cụm
    df = pd.read_parquet(src_path)

    # Xác định chế độ đơn-phim
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Merge] Single-movie mode → '{active_movie}'")
        # Nếu bảng có cột movie (title) thì lọc theo title;
        # nếu chỉ có movie_id, pipeline trước đã map id trong metadata — ở đây không ép buộc.
        if "movie" in df.columns:
            df = df[df["movie"].astype(str) == active_movie]
        elif "movie_id" in df.columns:
            # Giữ nguyên (không lọc) nếu chưa có map id->title; bước Build Warehouse đã ghi map.
            pass

    # ---------------------------
    # NO-OP MERGE (an toàn)
    # ---------------------------
    # Ở bản này mình chưa đổi/ghép nhãn cụm, chỉ pass-through.
    # Khi cần merge thật sự (ví dụ dùng cosine sim giữa centroid cụm trong cùng movie),
    # có thể bổ sung thuật toán ở đây và cập nhật cột cluster_id hợp nhất.
    merged = df.copy()

    # Ghi kết quả
    merged.to_parquet(dst_path, index=False)
    print(f"[Merge] Đã lưu {len(merged)} records -> {dst_path}")

    return str(dst_path)

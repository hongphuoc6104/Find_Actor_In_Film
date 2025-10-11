# tasks/pca_task.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from prefect import task, get_run_logger
from sklearn.decomposition import PCA

from utils.config_loader import load_config


def _safe_makedirs(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


@task(name="PCA Task")
def pca_task(
    embeddings_parquet: str = "warehouse/parquet/embeddings.parquet",
    out_parquet: str = "warehouse/parquet/embeddings_pca.parquet",
    model_path: str = "models/pca_model.joblib",
    requested_components: Optional[int] = None,
) -> None:
    """
    Đọc embeddings tổng (parquet) → fit PCA an toàn → lưu model + embeddings đã giảm chiều.

    * Tự động co n_components = min(n_samples, n_features, requested_components).
    * Nếu số mẫu < 2 (hoặc < n_components tối thiểu): BỎ QUA PCA
      - Copy nguyên embeddings sang out_parquet để những bước sau không bị lỗi file not found.
    * Nếu không có file embeddings_parquet → ghi log cảnh báo và bỏ qua PCA.

    Cấu hình đọc từ config.yaml:
      pca:
        n_components: 256
        whiten: false
        svd_solver: "full"  # (tuỳ chọn)
    """

    logger = get_run_logger()

    cfg = load_config()
    pca_cfg = (cfg or {}).get("pca", {}) if isinstance(cfg, dict) else {}
    default_components = int(pca_cfg.get("n_components", 256))
    whiten = bool(pca_cfg.get("whiten", False))
    svd_solver = str(pca_cfg.get("svd_solver", "full"))

    if requested_components is not None:
        default_components = int(requested_components)

    emb_path = Path(embeddings_parquet)
    if not emb_path.exists():
        logger.warning(f"[PCA] Not found: {embeddings_parquet}. Skip PCA.")
        return

    try:
        df = pd.read_parquet(embeddings_parquet)
    except Exception as e:
        logger.error(f"[PCA] Failed to read {embeddings_parquet}: {e}")
        return

    # Tìm cột embeddings
    emb_col = None
    for c in ["embedding", "embeddings", "vec", "feat"]:
        if c in df.columns:
            emb_col = c
            break

    if emb_col is None:
        # Có thể embeddings đang là nhiều cột số -> gom lại
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not numeric_cols:
            logger.error("[PCA] No embedding column found.")
            return
        X = df[numeric_cols].to_numpy(dtype=np.float32)
    else:
        # Cột embedding có thể là list/np.ndarray hoặc bytes
        def _to_vec(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.asarray(x, dtype=np.float32)
            # nếu string dạng "[...]" thì eval an toàn kiểu numpy
            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                try:
                    return np.asarray(eval(x), dtype=np.float32)  # noqa: S307 (chỉ dùng cho nội bộ)
                except Exception:
                    return None
            return None

        vecs = df[emb_col].apply(_to_vec)
        ok_mask = vecs.apply(lambda v: isinstance(v, np.ndarray))
        if not ok_mask.any():
            logger.error(f"[PCA] Column '{emb_col}' cannot be parsed to vectors.")
            return
        X = np.vstack(vecs[ok_mask].values)

        # Giữ lại các dòng hợp lệ
        df = df.loc[ok_mask].reset_index(drop=True)

    n_samples, n_features = X.shape
    logger.info(f"[PCA] Loaded {n_samples} embeddings with dimension {n_features}.")

    # Điều chỉnh n_components an toàn
    max_components = max(1, min(n_samples, n_features))
    n_components = min(default_components, max_components)

    if n_samples < 2 or n_components < 1:
        logger.warning(
            f"[PCA] Too few samples (n={n_samples}) or components (k={n_components}). "
            f"Skipping PCA and copying embeddings to {out_parquet}."
        )
        # Ghi ra file out_parquet để bước sau không lỗi.
        _safe_makedirs(Path(out_parquet))
        df.to_parquet(out_parquet, index=False)
        return

    logger.info(f"[PCA] Fitting PCA with n_components={n_components}, whiten={whiten}, svd_solver='{svd_solver}' ...")
    try:
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
        X_pca = pca.fit_transform(X)

        # Lưu model
        _safe_makedirs(Path(model_path))
        joblib.dump(pca, model_path)
        logger.info(f"[PCA] Saved PCA model → {model_path}")

        # Gắn lại vào DataFrame và lưu parquet
        df_pca = df.copy()
        # Lưu dưới dạng list để parquet đọc an toàn
        df_pca["embedding_pca"] = [row.astype(np.float32).tolist() for row in X_pca]

        _safe_makedirs(Path(out_parquet))
        df_pca.to_parquet(out_parquet, index=False)
        logger.info(f"[PCA] Saved PCA embeddings → {out_parquet}")

    except Exception as e:
        logger.error(f"[PCA] Exception during PCA: {e}")
        # Fallback: ghi bản gốc để không chặn pipeline
        _safe_makedirs(Path(out_parquet))
        df.to_parquet(out_parquet, index=False)
        logger.warning(f"[PCA] Wrote original embeddings to {out_parquet} as fallback.")

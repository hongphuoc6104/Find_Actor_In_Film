"""Utility helpers for working with numeric vectors."""

from __future__ import annotations


import numpy as np


def _mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
    """Return the mean vector from a list of vectors."""
    return np.mean(np.stack(vectors), axis=0)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector or collection of vectors using the L2 norm."""

    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 0:
        return arr.copy()

    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.copy()
        return arr / norm

    axes = tuple(range(1, arr.ndim))
    norms = np.linalg.norm(arr, axis=axes, keepdims=True)
    # Avoid division by zero by replacing empty norms with 1.0.
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return arr / safe_norms


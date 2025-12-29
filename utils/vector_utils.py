# utils/vector_utils.py
import numpy as np

def l2_normalize(v: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Chuẩn hóa một vector theo norm L2.
    """
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return v
    return v / norm

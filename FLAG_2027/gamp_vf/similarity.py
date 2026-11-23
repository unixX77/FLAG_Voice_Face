"""Stage D: cross-modal similarity matrices and neighbourhood statistics.

All values returned here are **raw cosine similarities** of unit vectors
(``score_type = "raw_cosine"``). Calibrated logits/probabilities and assignment
weights are produced elsewhere and stored under different names.
"""

from __future__ import annotations

import numpy as np
import torch


def cosine_matrix(voices: np.ndarray | torch.Tensor, faces: np.ndarray | torch.Tensor,
                  chunk: int = 4096, device: str = "auto") -> np.ndarray:
    """S[i, j] = cos(voice_i, face_j), computed in row chunks (GPU if available)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    v = torch.as_tensor(np.asarray(voices, dtype=np.float32))
    f = torch.as_tensor(np.asarray(faces, dtype=np.float32)).to(device)
    if v.ndim != 2 or f.ndim != 2 or v.shape[1] != f.shape[1]:
        raise ValueError("Expected 2-D voice and face matrices of equal width")
    f = torch.nn.functional.normalize(f, dim=1)
    out = np.empty((len(v), len(f)), dtype=np.float32)
    for start in range(0, len(v), chunk):
        block = torch.nn.functional.normalize(v[start:start + chunk].to(device), dim=1)
        out[start:start + chunk] = (block @ f.T).cpu().numpy()
    return out


def top_k(scores: np.ndarray, k: int, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k largest entries along ``axis`` (descending)."""
    s = scores if axis == 1 else scores.T
    k = min(k, s.shape[1])
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(s, idx, axis=1)


def reciprocal_top1(scores: np.ndarray) -> np.ndarray:
    """Boolean (rows,) : row i's best column j has row i as its best row."""
    best_col = scores.argmax(1)
    best_row = scores.argmax(0)
    return best_row[best_col] == np.arange(len(scores))


def rank_margin(scores: np.ndarray, axis: int = 1) -> np.ndarray:
    """Best minus second-best score per row (axis=1) or column (axis=0)."""
    s = scores if axis == 1 else scores.T
    if s.shape[1] < 2:
        return np.full(len(s), np.inf)
    part = -np.partition(-s, 1, axis=1)
    return part[:, 0] - part[:, 1]

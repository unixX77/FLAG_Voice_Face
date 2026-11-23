"""Unsupervised identity grouping (Stage A for voices, Stage B for faces) and metrics.

Ground-truth identities are never used to form clusters; they are only passed
to ``cluster_metrics`` for diagnostics. Label -1 means noise / unassigned.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def agglomerative(embeddings: np.ndarray, distance_threshold: float, min_size: int = 1) -> np.ndarray:
    """Average-linkage cosine clustering; clusters smaller than ``min_size`` become noise."""
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=distance_threshold).fit_predict(_unit(embeddings))
    return _drop_small(labels, min_size)


def hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int | None = None) -> np.ndarray:
    """HDBSCAN on unit vectors (Euclidean on the sphere is monotone in cosine)."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(_unit(embeddings))


def _drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    labels = labels.copy()
    if min_size > 1:
        values, counts = np.unique(labels[labels >= 0], return_counts=True)
        labels[np.isin(labels, values[counts < min_size])] = -1
    # Relabel 0..C-1 in order of first appearance for stable output.
    out = np.full_like(labels, -1)
    mapping: dict[int, int] = {}
    for i, value in enumerate(labels):
        if value >= 0:
            out[i] = mapping.setdefault(int(value), len(mapping))
    return out


def cluster_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Purity, NMI, ARI (noise counted as its own singleton points), fragmentation, coverage."""
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    assigned = predicted >= 0
    # Noise points become unique singletons so they cannot inflate agreement.
    filled = predicted.copy()
    filled[~assigned] = predicted.max(initial=-1) + 1 + np.arange((~assigned).sum())
    out = {"ari": float(adjusted_rand_score(truth, filled)),
           "nmi": float(normalized_mutual_info_score(truth, filled)),
           "coverage": float(assigned.mean()),
           "clusters": int(len(np.unique(predicted[assigned]))),
           "true_identities": int(len(np.unique(truth)))}
    if assigned.any():
        purity_hits, fragments = 0, []
        for c in np.unique(predicted[assigned]):
            _, counts = np.unique(truth[predicted == c], return_counts=True)
            purity_hits += counts.max()
        out["purity"] = float(purity_hits / assigned.sum())
        for t in np.unique(truth):
            members = predicted[(truth == t) & assigned]
            fragments.append(len(np.unique(members)) if len(members) else 0)
        fragments = np.asarray(fragments)
        out["fragmentation_mean_clusters_per_identity"] = float(fragments[fragments > 0].mean())
        out["identities_without_cluster"] = int((fragments == 0).sum())
    return out


def tune_threshold(embeddings: np.ndarray, truth: np.ndarray, thresholds, min_size: int = 1) -> tuple[float, list]:
    """Pick the agglomerative threshold with the best ARI on *labelled training-side* data."""
    table = []
    for t in thresholds:
        m = cluster_metrics(agglomerative(embeddings, t, min_size), truth)
        table.append({"threshold": float(t), **m})
    best = max(table, key=lambda r: r["ari"])
    return best["threshold"], table

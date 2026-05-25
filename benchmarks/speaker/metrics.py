"""Label-free speaker separability metrics for the benchmark.

These metrics quantify how well a backend's embedding space separates the two
speakers in the conversation *without* requiring manual speaker labels:

- ``l2_normalize`` / ``cosine_similarity``: basic embedding ops.
- ``separability``: KMeans(2) + silhouette score (cosine) — higher means the
  two speakers form tighter, better-separated clusters.
- ``target_similarity_stats``: distribution of cosine similarity to a target
  enrollment embedding, split by KMeans cluster (proxy for "how distinct the
  target speaker looks to a gate").
"""

from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array (or a single 1-D vector)."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return (arr / norm) if norm > 0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def separability(embeddings: np.ndarray, n_clusters: int = 2) -> dict[str, Any]:
    """Cluster embeddings and score separability with the silhouette coefficient.

    Args:
        embeddings: 2-D array, shape (n_segments, embedding_dim).
        n_clusters: Number of speakers to assume (2 for this benchmark).

    Returns:
        Dict with ``silhouette`` (float in [-1, 1] or None), ``labels``
        (cluster assignment per segment) and ``cluster_sizes``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    n = embeddings.shape[0]

    if n < n_clusters + 1:
        return {"silhouette": None, "labels": [], "cluster_sizes": [], "n": n}

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "separability requires scikit-learn. Install with: uv pip install scikit-learn"
        ) from e

    # Cosine geometry: L2-normalize, then KMeans (euclidean on the unit sphere
    # approximates cosine), and score the silhouette with the cosine metric.
    normed = l2_normalize(embeddings)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(normed)

    unique = np.unique(labels)
    if unique.size < 2:
        sil: float | None = None
    else:
        sil = float(silhouette_score(normed, labels, metric="cosine"))

    cluster_sizes = [int(np.sum(labels == c)) for c in range(n_clusters)]
    return {
        "silhouette": sil,
        "labels": labels.tolist(),
        "cluster_sizes": cluster_sizes,
        "n": n,
    }


def target_similarity_stats(
    embeddings: np.ndarray,
    target: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Cosine-similarity distribution of segments vs a target embedding.

    Args:
        embeddings: 2-D array (n_segments, dim).
        target: 1-D target embedding.
        labels: optional KMeans cluster labels to report per-cluster means
            (helps see whether the target separates the two clusters).

    Returns:
        Dict with overall mean/std/min/max and optional per-cluster means.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    sims = np.array([cosine_similarity(e, target) for e in embeddings])

    stats: dict[str, Any] = {
        "mean": float(sims.mean()) if sims.size else None,
        "std": float(sims.std()) if sims.size else None,
        "min": float(sims.min()) if sims.size else None,
        "max": float(sims.max()) if sims.size else None,
    }

    if labels is not None and len(labels) == len(sims):
        labels_arr = np.asarray(labels)
        per_cluster = {}
        for c in np.unique(labels_arr):
            mask = labels_arr == c
            per_cluster[int(c)] = float(sims[mask].mean())
        stats["per_cluster_mean"] = per_cluster
        # Gap between the two cluster means = how distinctly the target leans
        # toward one speaker.
        if len(per_cluster) == 2:
            vals = list(per_cluster.values())
            stats["cluster_mean_gap"] = abs(vals[0] - vals[1])

    return stats


def percentile(values: list[float], q: float) -> float | None:
    """Percentile helper that tolerates empty input."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


__all__ = [
    "l2_normalize",
    "cosine_similarity",
    "separability",
    "target_similarity_stats",
    "percentile",
]

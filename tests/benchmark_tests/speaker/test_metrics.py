"""Tests for speaker benchmark metrics (label-free separability)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.speaker.metrics import (
    cosine_similarity,
    l2_normalize,
    percentile,
    separability,
    target_similarity_stats,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


class TestL2Normalize:
    def test_unit_norm_rows(self) -> None:
        m = np.array([[3.0, 4.0], [1.0, 0.0]])
        normed = l2_normalize(m)
        assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)

    def test_handles_zero_row(self) -> None:
        m = np.array([[0.0, 0.0], [3.0, 4.0]])
        normed = l2_normalize(m)  # must not divide by zero
        assert np.isfinite(normed).all()


def _make_clusters(centers: np.ndarray, per: int, spread: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = [c + spread * rng.standard_normal((per, centers.shape[1])) for c in centers]
    return np.vstack(blocks)


class TestSeparability:
    def test_well_separated_high_silhouette(self) -> None:
        centers = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        emb = _make_clusters(centers, per=15, spread=0.1)
        result = separability(emb, n_clusters=2)
        assert result["silhouette"] is not None
        assert result["silhouette"] > 0.5
        assert sum(result["cluster_sizes"]) == 30

    def test_overlapping_low_silhouette(self) -> None:
        centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        emb = _make_clusters(centers, per=15, spread=1.0, seed=1)
        well = separability(
            _make_clusters(
                np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]), per=15, spread=0.1
            )
        )
        overlap = separability(emb, n_clusters=2)
        assert overlap["silhouette"] < well["silhouette"]

    def test_too_few_samples_returns_none(self) -> None:
        result = separability(np.array([[1.0, 2.0]]), n_clusters=2)
        assert result["silhouette"] is None


class TestTargetSimilarityStats:
    def test_basic_stats(self) -> None:
        emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        target = np.array([1.0, 0.0])
        stats = target_similarity_stats(emb, target)
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["max"] == pytest.approx(1.0)

    def test_per_cluster_means(self) -> None:
        emb = np.array([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.05, 1.0]])
        target = np.array([1.0, 0.0])
        labels = [0, 0, 1, 1]
        stats = target_similarity_stats(emb, target, labels)
        assert "per_cluster_mean" in stats
        assert stats["cluster_mean_gap"] > 0.5  # target clearly favors cluster 0


class TestPercentile:
    def test_empty(self) -> None:
        assert percentile([], 50) is None

    def test_median(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

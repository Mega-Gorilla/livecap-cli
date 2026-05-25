"""Tests for FAR/FRR/EER threshold calibration."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.speaker.calibration import (
    calibrate_from_labels,
    compute_eer,
    sweep_far_frr,
)


class TestComputeEER:
    def test_perfect_separation(self) -> None:
        target = np.array([0.9, 0.95, 0.92])
        impostor = np.array([0.1, 0.05, 0.2])
        eer, thr = compute_eer(target, impostor)
        assert eer == pytest.approx(0.0, abs=1e-9)
        assert 0.2 < thr <= 0.9

    def test_full_overlap_high_eer(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(0.5, 0.1, 200)
        impostor = rng.normal(0.5, 0.1, 200)
        eer, _ = compute_eer(target, impostor)
        assert eer > 0.3  # indistinguishable -> EER near 0.5

    def test_empty_returns_none(self) -> None:
        eer, thr = compute_eer(np.array([]), np.array([0.1]))
        assert eer is None and thr is None


class TestSweep:
    def test_monotonic_far_frr(self) -> None:
        target = np.array([0.8, 0.9, 0.85])
        impostor = np.array([0.1, 0.2, 0.15])
        rows = sweep_far_frr(target, impostor, [0.0, 0.5, 1.0])
        # At threshold 0: accept all -> FAR=1, FRR=0. At 1.0: reject all -> FAR=0, FRR=1.
        assert rows[0]["far"] == pytest.approx(1.0)
        assert rows[0]["frr"] == pytest.approx(0.0)
        assert rows[-1]["far"] == pytest.approx(0.0)
        assert rows[-1]["frr"] == pytest.approx(1.0)


def _two_speaker_embeddings(sep: float, per: int = 30, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    c0 = np.zeros(dim)
    c0[0] = sep
    c1 = np.zeros(dim)
    c1[0] = -sep
    a = c0 + 0.1 * rng.standard_normal((per, dim))
    b = c1 + 0.1 * rng.standard_normal((per, dim))
    emb = np.vstack([a, b])
    labels = [0] * per + [1] * per
    return emb, labels


class TestCalibrateFromLabels:
    def test_well_separated_low_eer(self) -> None:
        emb, labels = _two_speaker_embeddings(sep=10.0)
        cal = calibrate_from_labels(emb, labels)
        assert cal["eer"] is not None
        assert cal["eer"] < 0.05
        assert cal["n_target"] == 60  # both speakers as target, LOO
        assert cal["n_impostor"] == 60

    def test_overlapping_higher_eer(self) -> None:
        well, lw = _two_speaker_embeddings(sep=10.0, seed=1)
        over, lo = _two_speaker_embeddings(sep=0.2, seed=2)
        assert calibrate_from_labels(over, lo)["eer"] > calibrate_from_labels(well, lw)["eer"]

    def test_single_speaker_returns_none(self) -> None:
        emb = np.random.default_rng(0).standard_normal((10, 8))
        cal = calibrate_from_labels(emb, [0] * 10)
        assert cal["eer"] is None
        assert cal["n_speakers"] == 1

    def test_none_labels_counted_uncertain(self) -> None:
        emb, labels = _two_speaker_embeddings(sep=5.0)
        labels[0] = None
        labels[1] = None
        cal = calibrate_from_labels(emb, labels)
        assert cal["n_uncertain"] == 2

"""Pin AudioSet class mapping integrity + threshold policy semantics.

These tests do not require the EfficientAT clone for the core integrity
checks; the optional ``test_indices_match_efficientat_csv`` test loads the
canonical class table when the clone is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.sed.class_mapping import (
    NUM_AUDIOSET_CLASSES,
    POLICIES,
    SPEECH_LIKE_CLASSES,
    SPEECH_LIKE_INDICES,
    TARGET_CLASSES,
    TARGET_INDICES,
    load_class_names,
    max_policy,
    speech_like_indices,
    sum_policy,
    target_indices,
    target_minus_speech_policy,
    verify_indices,
)


class TestClassMappingIntegrity:
    """Constants reflect Issue #305 v3 design intent."""

    def test_target_and_speech_like_are_disjoint(self) -> None:
        assert not (TARGET_CLASSES & SPEECH_LIKE_CLASSES), (
            "TARGET and SPEECH_LIKE must not overlap; otherwise the "
            "target_minus_speech policy double-counts a single class."
        )

    def test_indices_match_class_counts(self) -> None:
        assert len(TARGET_INDICES) == len(TARGET_CLASSES)
        assert len(SPEECH_LIKE_INDICES) == len(SPEECH_LIKE_CLASSES)

    def test_indices_are_within_audioset_range(self) -> None:
        for idx in TARGET_INDICES + SPEECH_LIKE_INDICES:
            assert 0 <= idx < NUM_AUDIOSET_CLASSES, (
                f"Index {idx} is outside the AudioSet 0..{NUM_AUDIOSET_CLASSES - 1} range"
            )

    def test_indices_helpers_return_sorted_tuples(self) -> None:
        assert target_indices() == TARGET_INDICES
        assert speech_like_indices() == SPEECH_LIKE_INDICES
        assert list(TARGET_INDICES) == sorted(TARGET_INDICES)
        assert list(SPEECH_LIKE_INDICES) == sorted(SPEECH_LIKE_INDICES)

    def test_target_classes_cover_pr_b_desk_tap_family(self) -> None:
        """Pin the empirical motivation: desk_tap-style transients are present.

        PR-B calibration (#304) established that ``desk_tap`` could not be
        caught by DSP. The SED layer's whole reason for existing is to catch
        the ``Knock`` / ``Tap`` / ``Thump, thud`` family — pin that here so a
        future refactor can't silently drop them.
        """

        for required in ("Knock", "Tap", "Thump, thud", "Applause", "Clapping"):
            assert required in TARGET_CLASSES, (
                f"{required!r} must remain in TARGET_CLASSES — it is part of "
                "the PR-B desk_tap empirical motivation for Phase 2 SED."
            )


class TestThresholdPolicies:
    """Hand-derived numerical pins for the three Issue #305 v3 policies."""

    @pytest.fixture
    def probs(self) -> np.ndarray:
        """A 527-vector with specific values at TARGET / SPEECH_LIKE indices."""

        probs = np.full(NUM_AUDIOSET_CLASSES, 0.05, dtype=np.float32)
        # Plant target probabilities so we know max / sum exactly.
        for idx, value in zip(TARGET_INDICES, [0.10, 0.20, 0.30, 0.40, 0.80, 0.05, 0.15, 0.25, 0.35, 0.45]):
            probs[idx] = value
        # Plant speech-like probabilities.
        for idx, value in zip(SPEECH_LIKE_INDICES, [0.55, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]):
            probs[idx] = value
        return probs

    def test_max_policy_picks_highest_target(self, probs: np.ndarray) -> None:
        # Highest planted target value is 0.80.
        assert max_policy(probs) == pytest.approx(0.80, abs=1e-6)

    def test_sum_policy_sums_targets(self, probs: np.ndarray) -> None:
        expected = 0.10 + 0.20 + 0.30 + 0.40 + 0.80 + 0.05 + 0.15 + 0.25 + 0.35 + 0.45
        assert sum_policy(probs) == pytest.approx(expected, abs=1e-6)

    def test_target_minus_speech_subtracts_speech_max(
        self, probs: np.ndarray
    ) -> None:
        # max target = 0.80, max speech-like = 0.55 → 0.25.
        assert target_minus_speech_policy(probs) == pytest.approx(0.25, abs=1e-6)

    def test_policies_registry_matches_callables(self, probs: np.ndarray) -> None:
        # POLICIES dict is what the CLI exposes; ensure the names match
        # functions used elsewhere in tests / decision doc generators.
        assert POLICIES["max"](probs) == max_policy(probs)
        assert POLICIES["sum"](probs) == sum_policy(probs)
        assert POLICIES["target_minus_speech"](probs) == target_minus_speech_policy(probs)


class TestProbsValidation:
    """Wrong-shape inputs raise rather than silently misbehaving."""

    def test_max_policy_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="length 527"):
            max_policy(np.zeros(100, dtype=np.float32))

    def test_max_policy_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D array"):
            max_policy(np.zeros((1, 527), dtype=np.float32))


class TestVerifyAgainstEfficientATCsv:
    """Cross-check pinned indices against the canonical CSV when available.

    This guards against silent drift if EfficientAT (or a downstream rev of
    the AudioSet ontology) renames or re-orders a class we depend on.
    """

    def test_indices_match_efficientat_csv(
        self, efficientat_path: Path | None
    ) -> None:
        if efficientat_path is None:
            pytest.skip(
                "EfficientAT clone not available; see benchmarks/sed/README.md"
            )

        csv_path = efficientat_path / "metadata" / "class_labels_indices.csv"
        if not csv_path.is_file():
            pytest.skip(f"Class labels CSV not at expected location: {csv_path}")

        names = load_class_names(csv_path)
        assert len(names) == NUM_AUDIOSET_CLASSES
        verify_indices(names)

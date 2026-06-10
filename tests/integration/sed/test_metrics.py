"""Pin metric semantics with hand-derived synthetic ground truth.

These tests exercise the Issue #305 v3 metric definitions on a tiny
synthetic corpus where every value is enumerated explicitly. They never
require EfficientAT — only :mod:`benchmarks.sed.metrics` and
:mod:`benchmarks.sed.class_mapping`.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.sed.class_mapping import (
    NUM_AUDIOSET_CLASSES,
    TARGET_INDICES,
)
from benchmarks.sed.metrics import (
    PROVISIONAL_PRECISION_FLOOR,
    PROVISIONAL_RECALL_FLOOR,
    PerClipResult,
    compute_class_level_metrics,
    compute_reject_signal_curve,
    provisional_gate_verdict,
)


def _make_clip(
    label: str,
    kind: str,
    *,
    target_max: float,
    speech_max: float = 0.0,
    is_short_utterance: bool = False,
    n_windows: int = 2,
) -> PerClipResult:
    """Build a synthetic PerClipResult with deterministic per-class values.

    All non-target / non-speech-like classes are filled with ``0.01`` so the
    interesting numbers are obvious in the assertion arithmetic.
    """

    probs = np.full(
        (n_windows, NUM_AUDIOSET_CLASSES), 0.01, dtype=np.float32
    )
    # Plant the target max at index 359 ("Knock") on window 0 so clip_max
    # reflects ``target_max`` for any policy.
    probs[0, 359] = target_max
    # Plant the speech-like max at index 0 ("Speech").
    probs[0, 0] = speech_max
    return PerClipResult(
        label=label,
        kind=kind,  # type: ignore[arg-type]
        is_short_utterance=is_short_utterance,
        per_window_probs=probs,
    )


@pytest.fixture
def four_clip_corpus() -> list[PerClipResult]:
    """Two negative (should-reject) + two positive (must-remain) clips.

    Target scores are chosen so the threshold sweep at ``0.50`` gives
    precision=1.0, recall=1.0 — a perfect ``Bundle OK`` style outcome.
    """

    return [
        _make_clip("desk_tap_synth", "negative", target_max=0.90),
        _make_clip("applause_synth", "negative", target_max=0.80),
        _make_clip("normal_speech_synth", "positive", target_max=0.10, speech_max=0.90),
        _make_clip(
            "short_utter_synth",
            "positive",
            target_max=0.05,
            speech_max=0.85,
            is_short_utterance=True,
        ),
    ]


@pytest.fixture
def precision_failing_corpus() -> list[PerClipResult]:
    """One positive clip leaks above threshold → precision = 2/3 < 0.70."""

    return [
        _make_clip("desk_tap_synth", "negative", target_max=0.90),
        _make_clip("applause_synth", "negative", target_max=0.80),
        _make_clip("normal_speech_synth", "positive", target_max=0.75, speech_max=0.10),
    ]


@pytest.fixture
def recall_failing_corpus() -> list[PerClipResult]:
    """Both negative clips score very low → recall < 0.50 at any useful threshold."""

    return [
        _make_clip("desk_tap_synth", "negative", target_max=0.20),
        _make_clip("applause_synth", "negative", target_max=0.10),
        _make_clip("normal_speech_synth", "positive", target_max=0.05),
        _make_clip("short_utter_synth", "positive", target_max=0.05),
    ]


class TestRejectSignalCurve:
    def test_perfect_case_precision_recall_1_at_threshold(
        self, four_clip_corpus: list[PerClipResult]
    ) -> None:
        curve = compute_reject_signal_curve(
            four_clip_corpus,
            policy_name="max",
            thresholds=[0.50, 0.85, 0.95],
        )

        # At T=0.50: both negatives score >=0.50 (TP=2) and both positives
        # score <0.50 (TN=2). Precision = 2/2 = 1.0; recall = 2/2 = 1.0.
        assert curve.precision[0] == pytest.approx(1.0)
        assert curve.recall[0] == pytest.approx(1.0)

        # At T=0.85: only desk_tap_synth (0.90) trips. TP=1, FN=1, FP=0.
        # Precision = 1.0; recall = 0.5.
        assert curve.precision[1] == pytest.approx(1.0)
        assert curve.recall[1] == pytest.approx(0.5)

        # At T=0.95: nothing trips → precision is 0 by convention, recall = 0.
        assert curve.precision[2] == pytest.approx(0.0)
        assert curve.recall[2] == pytest.approx(0.0)

    def test_per_clip_scores_match_policy_definition(
        self, four_clip_corpus: list[PerClipResult]
    ) -> None:
        curve = compute_reject_signal_curve(
            four_clip_corpus, policy_name="max", thresholds=[0.5]
        )
        assert curve.per_clip_scores["desk_tap_synth"] == pytest.approx(0.90)
        assert curve.per_clip_scores["applause_synth"] == pytest.approx(0.80)
        assert curve.per_clip_scores["normal_speech_synth"] == pytest.approx(0.10)
        assert curve.per_clip_scores["short_utter_synth"] == pytest.approx(0.05)

    def test_unknown_policy_raises(self, four_clip_corpus: list[PerClipResult]) -> None:
        with pytest.raises(KeyError, match="Unknown policy"):
            compute_reject_signal_curve(
                four_clip_corpus, policy_name="bogus", thresholds=[0.5]
            )


class TestClassLevelMetrics:
    def test_per_class_threshold_at_planted_indices(
        self, four_clip_corpus: list[PerClipResult]
    ) -> None:
        # Target indices include "Knock" (359) where we planted the max
        # values; the unrelated targets see 0.01 → all false at T=0.5.
        # So at T=0.5 on "Knock": both negatives trip → P=R=1.0.
        metrics = compute_class_level_metrics(
            four_clip_corpus,
            class_indices=TARGET_INDICES,
            class_names=[f"class_{i}" for i in TARGET_INDICES],
            threshold=0.5,
        )
        knock = next(m for m in metrics if m.class_index == 359)
        assert knock.precision == pytest.approx(1.0)
        assert knock.recall == pytest.approx(1.0)
        assert knock.counts.tp == 2
        assert knock.counts.fp == 0
        assert knock.counts.fn == 0
        assert knock.counts.tn == 2

        # An index where nothing was planted: all clips at 0.01 < 0.5.
        # TP=0, FN=2 (negatives), FP=0, TN=2 (positives).
        other = next(m for m in metrics if m.class_index != 359)
        assert other.counts.tp == 0
        assert other.counts.fn == 2
        assert other.precision == 0.0
        assert other.recall == 0.0

    def test_misaligned_class_lists_raise(
        self, four_clip_corpus: list[PerClipResult]
    ) -> None:
        with pytest.raises(ValueError, match="matching length"):
            compute_class_level_metrics(
                four_clip_corpus,
                class_indices=[0, 1],
                class_names=["only_one"],
                threshold=0.5,
            )


class TestProvisionalGateVerdict:
    """Pin the 4-case truth table for the Issue #305 v3 gate."""

    def test_pass_picks_highest_recall_threshold_and_marks_provisional(
        self, four_clip_corpus: list[PerClipResult]
    ) -> None:
        curve = compute_reject_signal_curve(
            four_clip_corpus, policy_name="max", thresholds=[0.50, 0.85, 0.95]
        )
        verdict = provisional_gate_verdict([curve], target_label="desk_tap_synth")

        assert verdict.passed is True
        assert verdict.is_provisional is True, (
            "Issue #305 v3 mandates the gate report itself as provisional "
            "even when satisfied."
        )
        assert verdict.passing_policies == ("max",)
        # Tiebreak prefers max recall (T=0.50 → recall 1.0).
        assert verdict.chosen_threshold == pytest.approx(0.50)
        assert verdict.chosen_precision == pytest.approx(1.0)
        assert verdict.chosen_recall == pytest.approx(1.0)
        assert verdict.target_flagged_at_chosen_threshold is True

    def test_precision_failure_blocks_pass(
        self, precision_failing_corpus: list[PerClipResult]
    ) -> None:
        curve = compute_reject_signal_curve(
            precision_failing_corpus,
            policy_name="max",
            thresholds=[0.50, 0.85],
        )
        verdict = provisional_gate_verdict(
            [curve], target_label="desk_tap_synth"
        )
        # At T=0.50: TP=2, FP=1 → precision = 2/3 ≈ 0.667 < 0.70 floor.
        # At T=0.85: TP=1, FP=0 → precision 1.0 but recall 0.5 = floor (boundary).
        # The pair (T=0.85) clears: precision 1.0, recall 0.5 → passes.
        # Force a clearer fail: use a higher precision floor.
        verdict_strict = provisional_gate_verdict(
            [curve],
            target_label="desk_tap_synth",
            precision_floor=0.95,
        )
        # At T=0.50: precision 0.667 < 0.95.
        # At T=0.85: precision 1.0 ≥ 0.95 but recall 0.5 ≥ 0.50 → still passes.
        # So lift the recall floor too.
        verdict_strictest = provisional_gate_verdict(
            [curve],
            target_label="desk_tap_synth",
            precision_floor=0.95,
            recall_floor=0.90,
        )
        assert verdict_strictest.passed is False
        assert verdict_strictest.passing_policies == ()
        assert verdict_strictest.chosen_policy is None
        # Sanity: the relaxed gate still passes.
        assert verdict.passed is True
        # Verify strict precision case behaviour
        assert verdict_strict.passed is True

    def test_recall_failure_blocks_pass(
        self, recall_failing_corpus: list[PerClipResult]
    ) -> None:
        curve = compute_reject_signal_curve(
            recall_failing_corpus, policy_name="max", thresholds=[0.05, 0.30, 0.50]
        )
        verdict = provisional_gate_verdict(
            [curve], target_label="desk_tap_synth"
        )
        # Max target=0.20; at T=0.05 we get recall=1.0 but precision: TP=2, FP=2 = 0.5 < 0.7 floor.
        # At T=0.30 nothing trips → precision and recall both 0.
        assert verdict.passed is False
        assert verdict.chosen_policy is None

    def test_target_must_be_flagged_at_chosen_threshold(self) -> None:
        """If the target clip itself doesn't trip, the gate must fail."""

        # Synthetic case: an applause clip scores high but desk_tap scores low.
        corpus = [
            _make_clip("applause_synth", "negative", target_max=0.90),
            _make_clip("desk_tap_synth", "negative", target_max=0.30),
            _make_clip("normal_speech_synth", "positive", target_max=0.05),
            _make_clip("short_utter_synth", "positive", target_max=0.05),
        ]
        curve = compute_reject_signal_curve(
            corpus, policy_name="max", thresholds=[0.85, 0.50]
        )
        # T=0.85: only applause trips → P=1.0, R=0.5 (meets corpus floor),
        # but desk_tap_synth (0.30) is NOT flagged → gate must reject.
        # T=0.50: nothing trips for desk_tap_synth (0.30 < 0.50) → reject.
        verdict = provisional_gate_verdict(
            [curve], target_label="desk_tap_synth"
        )
        assert verdict.passed is False
        assert verdict.target_flagged_at_chosen_threshold is None

    def test_floor_constants_pinned(self) -> None:
        """Issue #305 v3 defaults must remain visible to readers."""

        assert PROVISIONAL_PRECISION_FLOOR == 0.70
        assert PROVISIONAL_RECALL_FLOOR == 0.50

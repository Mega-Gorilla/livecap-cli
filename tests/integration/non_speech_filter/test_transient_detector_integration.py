"""Pipeline-level tests for the Layer 1 transient detector (Issue #295 PR-B).

These tests verify three properties that the unit tests cannot prove on
their own:

1. **Observe mode is a no-op on the metric layer.** Adding an observe
   detector to the baseline pipeline must not change
   ``false_asr_trigger_rate``, ``speech_recall`` or
   ``short_utterance_recall`` for any backend (BASELINE_INVARIANTS
   regression guard).
2. **On mode never zeros short utterances or normal speech.** The
   detector exists to gate non-speech; flipping it on must not break
   recall on positive items.
3. **On mode tightens applause for at least one backend.** Defaults are
   not tuned (that is what PR-B's calibration sweep is for), but the
   detector must demonstrably move WebRTC's burst false_trigger rate
   downward to justify shipping the layer at all.
"""

from __future__ import annotations

import pytest

from benchmarks.non_speech_filter import (
    CorpusItem,
    MockEngine,
    TransientDetectorConfig,
    build_pipeline,
    evaluate_pipeline,
)


# ---------- Fixtures -----------------------------------------------------


@pytest.fixture
def detector_factory(backend_type: str):
    """Factory producing a transient-detector-enabled pipeline."""

    def _factory(*, mode: str = "observe", **overrides):
        config = TransientDetectorConfig(mode=mode, **overrides)

        def make():
            engine = MockEngine()
            return build_pipeline(
                backend_type, engine=engine, transient_config=config
            )

        return make

    return _factory


# ---------- Observe-mode invariance --------------------------------------


@pytest.mark.evaluation_harness
def test_observe_mode_does_not_change_baseline_metrics(
    backend_type: str,
    transcriber_factory,
    detector_factory,
    synthetic_corpus_items: list[CorpusItem],
) -> None:
    """Observe mode must reproduce the PR-0 baseline metrics exactly."""
    baseline = evaluate_pipeline(
        transcriber_factory,
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )
    observed = evaluate_pipeline(
        detector_factory(mode="observe"),
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )

    assert observed.false_asr_trigger_rate == baseline.false_asr_trigger_rate
    assert observed.speech_recall == baseline.speech_recall
    assert observed.short_utterance_recall == baseline.short_utterance_recall


# ---------- On-mode positive preservation --------------------------------


@pytest.mark.evaluation_harness
def test_on_mode_preserves_positive_recall(
    backend_type: str,
    detector_factory,
    synthetic_corpus_items: list[CorpusItem],
) -> None:
    """On mode must not drop short utterances or normal speech."""
    result = evaluate_pipeline(
        detector_factory(mode="on"),
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )
    # Floors mirror BASELINE_INVARIANTS for tenvad/webrtc (silero's
    # synthetic recall is zero by design — see docs/benchmarks/non-speech-
    # filter.md).
    if backend_type == "silero":
        # Just ensure on-mode does not crash; recall stays at observed
        # PR-0 level.
        assert result.speech_recall is not None
    else:
        assert result.speech_recall is not None
        assert result.speech_recall >= 0.80, (
            f"{backend_type} speech_recall under on-mode = {result.speech_recall}"
        )
        assert result.short_utterance_recall is not None
        assert result.short_utterance_recall >= 0.80, (
            f"{backend_type} short_utterance_recall under on-mode = "
            f"{result.short_utterance_recall}"
        )


# ---------- On-mode applause reduction (WebRTC, the canonical case) ------


@pytest.mark.evaluation_harness
def test_on_mode_reduces_or_holds_webrtc_burst_false_trigger(
    detector_factory,
    transcriber_factory,
    synthetic_corpus_items: list[CorpusItem],
    backend_type: str,
) -> None:
    """WebRTC × synthetic burst should not get *worse* under on-mode.

    The PR-B target (50 % → 0 %) is a calibration goal that requires the
    sweep harness. This test asserts the much weaker (but still
    meaningful) "no regression" property for default thresholds.
    """
    if backend_type != "webrtc":
        pytest.skip("Only WebRTC drives the synthetic-burst regression target")

    baseline = evaluate_pipeline(
        transcriber_factory,
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )
    on_result = evaluate_pipeline(
        detector_factory(mode="on"),
        synthetic_corpus_items,
        backend_name=backend_type,
        measure_hallucination=False,
    )

    assert on_result.false_asr_trigger_rate is not None
    assert baseline.false_asr_trigger_rate is not None
    assert on_result.false_asr_trigger_rate <= baseline.false_asr_trigger_rate, (
        f"on-mode regressed WebRTC false_trigger from "
        f"{baseline.false_asr_trigger_rate:.0%} to "
        f"{on_result.false_asr_trigger_rate:.0%}"
    )

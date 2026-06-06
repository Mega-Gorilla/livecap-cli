"""Unit tests for the engine adapters used by the non-speech filter harness.

Validates the two behaviours that codex-review flagged as broken before this
PR landed:

1. ``InstrumentedEngine`` must transparently delegate to a real engine while
   recording ``transcribe_count`` and ``last_texts`` — without this, real
   engines silently fix ``non_empty_hallucination_rate`` at 0.
2. ``evaluate_pipeline(fail_fast=False)`` must capture per-item errors in
   ``per_label[label]['error']`` instead of swallowing them silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.non_speech_filter import (
    CorpusItem,
    InstrumentedEngine,
    MockEngine,
    evaluate_pipeline,
)


class _NoCountersEngine:
    """Engine stand-in that mimics real engines (no transcribe_count / last_texts)."""

    def __init__(self, return_text: str = "ご視聴ありがとうございました") -> None:
        self._return_text = return_text

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        return self._return_text, 1.0

    def get_required_sample_rate(self) -> int:
        return 16000

    def get_engine_name(self) -> str:
        return "no-counters"

    def get_supported_languages(self) -> list[str]:
        return ["ja"]

    def cleanup(self) -> None:
        pass


class TestInstrumentedEngine:
    """``InstrumentedEngine`` adds the surface real engines lack."""

    def test_transcribe_count_increments(self) -> None:
        wrapped = InstrumentedEngine(_NoCountersEngine())
        assert wrapped.transcribe_count == 0
        wrapped.transcribe(np.zeros(1600, dtype=np.float32), 16000)
        wrapped.transcribe(np.zeros(1600, dtype=np.float32), 16000)
        assert wrapped.transcribe_count == 2

    def test_last_texts_recorded(self) -> None:
        wrapped = InstrumentedEngine(_NoCountersEngine("hello"))
        wrapped.transcribe(np.zeros(1600, dtype=np.float32), 16000)
        wrapped.transcribe(np.zeros(1600, dtype=np.float32), 16000)
        assert wrapped.last_texts == ["hello", "hello"]

    def test_delegates_non_transcribe_methods(self) -> None:
        wrapped = InstrumentedEngine(_NoCountersEngine())
        assert wrapped.get_required_sample_rate() == 16000
        assert wrapped.get_engine_name() == "no-counters"
        assert wrapped.get_supported_languages() == ["ja"]
        wrapped.cleanup()  # Must not raise.

    def test_inner_property_exposes_wrapped_engine(self) -> None:
        inner = _NoCountersEngine()
        wrapped = InstrumentedEngine(inner)
        assert wrapped.inner is inner


class TestEvaluatePipelineErrorHandling:
    """``evaluate_pipeline`` must surface or capture pipeline exceptions."""

    @pytest.fixture
    def single_corpus(self) -> list[CorpusItem]:
        return [
            CorpusItem(
                label="probe",
                kind="negative",
                is_short_utterance=False,
                audio=np.zeros(1600, dtype=np.float32),
            )
        ]

    def test_fail_fast_default_raises(self, single_corpus: list[CorpusItem]) -> None:
        """When ``fail_fast=True`` (default) a pipeline failure must escape."""

        def failing_factory() -> tuple[object, MockEngine]:
            raise RuntimeError("simulated pipeline construction failure")

        with pytest.raises(RuntimeError, match="simulated pipeline"):
            evaluate_pipeline(failing_factory, single_corpus, backend_name="probe")

    def test_fail_fast_false_captures_error(
        self, single_corpus: list[CorpusItem]
    ) -> None:
        """``fail_fast=False`` records ``error`` in per-label and continues."""

        class _FailingTranscriber:
            def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
                raise RuntimeError("simulated feed_audio failure")

            def finalize(self) -> list:
                return []

        def factory() -> tuple[object, MockEngine]:
            return _FailingTranscriber(), MockEngine()

        evaluation = evaluate_pipeline(
            factory,
            single_corpus,
            backend_name="probe",
            fail_fast=False,
        )
        assert evaluation.per_label["probe"]["error"] is not None
        assert "simulated feed_audio" in evaluation.per_label["probe"]["error"]
        # Errored item must not be counted as triggering ASR.
        assert evaluation.per_label["probe"]["triggered"] is False
        assert evaluation.false_asr_trigger_rate == 0.0

"""Tests for ``_SegmentTranscriptionOutcome`` factory invariants (Issue #332).

Internal helper (non-public), but smoke-tested here so dispatcher contracts
stay observable.
"""

from __future__ import annotations

import dataclasses

import pytest

from livecap_cli.transcription.result import TranscriptionResult
from livecap_cli.transcription.stream import _SegmentTranscriptionOutcome
from livecap_cli.transcription.utterance import REASON_FILTER_REJECT


def _make_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="ok",
        start_time=0.0,
        end_time=1.0,
        is_final=True,
        confidence=0.9,
    )


class TestSegmentTranscriptionOutcome:
    def test_success_factory_carries_result_no_reason(self) -> None:
        result = _make_result()
        outcome = _SegmentTranscriptionOutcome.success(result)
        assert outcome.result is result
        assert outcome.drop_reason is None

    def test_dropped_factory_carries_reason_no_result(self) -> None:
        outcome = _SegmentTranscriptionOutcome.dropped(REASON_FILTER_REJECT)
        assert outcome.result is None
        assert outcome.drop_reason == REASON_FILTER_REJECT

    def test_frozen_immutability(self) -> None:
        outcome = _SegmentTranscriptionOutcome.dropped(REASON_FILTER_REJECT)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.drop_reason = "tampered"  # type: ignore[misc]

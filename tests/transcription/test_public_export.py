"""Tests for public re-export surface (Issue #332).

``UtteranceSettledEvent`` / ``REASON_*`` は public re-export 対象。
``_SegmentTranscriptionOutcome`` は internal helper、``__all__`` から除外。
"""

from __future__ import annotations


class TestPublicExportSurface:
    def test_utterance_event_top_level_import(self) -> None:
        """``from livecap_cli import UtteranceSettledEvent`` 成功。"""
        from livecap_cli import UtteranceSettledEvent

        assert UtteranceSettledEvent.__name__ == "UtteranceSettledEvent"

    def test_reason_constants_top_level_import(self) -> None:
        """``REASON_*`` 4 件 + 期待値 pin。"""
        from livecap_cli import (
            REASON_EMPTY_AUDIO,
            REASON_ENERGY_GATE,
            REASON_ENGINE_EMPTY,
            REASON_FILTER_REJECT,
        )

        assert REASON_EMPTY_AUDIO == "segment:empty_audio"
        assert REASON_ENERGY_GATE == "energy_gate:low_rms"
        assert REASON_FILTER_REJECT == "confidence_filter:reject"
        assert REASON_ENGINE_EMPTY == "engine:empty_text"

    def test_segment_outcome_not_exported(self) -> None:
        """``_SegmentTranscriptionOutcome`` は private、``__all__`` に含まれない。"""
        import livecap_cli
        import livecap_cli.transcription

        assert "_SegmentTranscriptionOutcome" not in livecap_cli.__all__
        assert "_SegmentTranscriptionOutcome" not in livecap_cli.transcription.__all__
        # 名前で import attempt しても top level には存在しない
        assert not hasattr(livecap_cli, "_SegmentTranscriptionOutcome")

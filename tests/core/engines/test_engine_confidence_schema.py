"""EngineConfidence + TranscriptionResult schema の pin test (Issue #308 / PR-A.0).

実 ASR モデルを load せず、dataclass の挙動だけを verify する pure-unit test。
"""
from dataclasses import FrozenInstanceError

import pytest

from livecap_cli.engines import EngineConfidence, TranscriptionResult


class TestEngineConfidenceDefaults:
    def test_all_fields_default_to_none(self):
        ec = EngineConfidence()
        assert ec.no_speech_prob is None
        assert ec.avg_logprob is None
        assert ec.compression_ratio is None
        assert ec.token_confidence_mean is None
        assert ec.raw == {}

    def test_is_available_false_when_all_none(self):
        ec = EngineConfidence()
        assert ec.is_available is False

    def test_is_available_true_when_no_speech_prob_set(self):
        ec = EngineConfidence(no_speech_prob=0.1)
        assert ec.is_available is True

    def test_is_available_true_when_avg_logprob_set(self):
        ec = EngineConfidence(avg_logprob=-0.5)
        assert ec.is_available is True

    def test_is_available_true_when_compression_ratio_set(self):
        ec = EngineConfidence(compression_ratio=1.2)
        assert ec.is_available is True

    def test_is_available_true_when_token_confidence_mean_set(self):
        ec = EngineConfidence(token_confidence_mean=0.95)
        assert ec.is_available is True

    def test_is_available_only_checks_signal_fields_not_raw(self):
        """raw dict が non-empty でも 4 つの signal field 全 None なら False (規約)。

        PR-A.1 filter は 4 つの canonical field のみを判定対象にする。raw は
        engine 固有 overflow であり、filter には使われない。
        """
        ec = EngineConfidence(raw={"some_engine_specific_metric": 0.5})
        assert ec.is_available is False

    def test_frozen_prevents_mutation(self):
        ec = EngineConfidence()
        with pytest.raises(FrozenInstanceError):
            ec.no_speech_prob = 0.5  # type: ignore[misc]


class TestTranscriptionResultSchema:
    """``TranscriptionResult`` の attribute access semantics を pin。

    Note: PR-A.0 で導入した ``__iter__`` (Tuple[str, float] 旧契約との
    後方互換) は pre-1.0 cleanup により削除済。caller は attribute access
    に統一されている (``stream.py`` / benchmarks / test mock 等)。
    """

    def test_engine_confidence_default_is_empty(self):
        result = TranscriptionResult(text="hi", confidence=0.9)
        assert isinstance(result.engine_confidence, EngineConfidence)
        assert result.engine_confidence.is_available is False

    def test_engine_confidence_can_be_provided(self):
        ec = EngineConfidence(no_speech_prob=0.2, avg_logprob=-0.3)
        result = TranscriptionResult(text="hi", confidence=0.9, engine_confidence=ec)
        assert result.engine_confidence.is_available is True
        assert result.engine_confidence.no_speech_prob == 0.2

    def test_frozen_prevents_mutation(self):
        result = TranscriptionResult(text="hi", confidence=0.5)
        with pytest.raises(FrozenInstanceError):
            result.text = "changed"  # type: ignore[misc]

    def test_tuple_unpacking_no_longer_supported(self):
        """pre-1.0 cleanup で ``__iter__`` を削除済。tuple 旧契約は使えない。"""
        result = TranscriptionResult(text="hi", confidence=0.5)
        with pytest.raises(TypeError):
            text, confidence = result  # noqa: F841 — must raise TypeError


class TestPublicReexport:
    """`from livecap_cli.engines import ...` が __all__ で公開されているか pin。"""

    def test_engine_confidence_reexported(self):
        from livecap_cli.engines import EngineConfidence as Reexported  # noqa: F401

    def test_transcription_result_reexported(self):
        from livecap_cli.engines import TranscriptionResult as Reexported  # noqa: F401

    def test_listed_in_all(self):
        import livecap_cli.engines as engines_pkg
        assert "EngineConfidence" in engines_pkg.__all__
        assert "TranscriptionResult" in engines_pkg.__all__

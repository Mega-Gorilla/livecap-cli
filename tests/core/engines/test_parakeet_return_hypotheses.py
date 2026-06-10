"""Parakeet adapter integration test: `return_hypotheses=True` を NeMo に渡し、
Hypothesis 戻り値から engine_confidence を抽出することを pin する。

codex-review on #309 で「`return_hypotheses=True` が抜けていると Hypothesis
が adapter に届かないため engine_confidence は常に全 None になる」と
指摘された問題への regression test。

実 NeMo を起動せず、`ParakeetEngine.__new__` で初期化バイパス + FakeModel /
FakeHypothesis で transcribe path 全体を検証する。
"""
import importlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

parakeet_engine = importlib.import_module("livecap_cli.engines.parakeet_engine")
ParakeetEngine = parakeet_engine.ParakeetEngine


@dataclass
class FakeHypothesis:
    """NeMo `Hypothesis` の最小 mock。"""
    text: str = "hello"
    token_confidence: Optional[List[float]] = None
    score: Optional[float] = None
    y_sequence: Optional[List[int]] = None


@pytest.fixture
def fake_engine():
    """ParakeetEngine を __new__ で生成し、必要最小限の attribute だけ set する。

    実 NeMo / GPU / model_manager を回避するため、`load_model()` 系の path は
    使わない。test では `engine.model = MagicMock()` を直接差し込む。
    """
    engine = ParakeetEngine.__new__(ParakeetEngine)
    engine.engine_name = "parakeet_ja"
    engine.model_name = "nvidia/parakeet-tdt_ctc-0.6b-ja"
    engine.decoding_strategy = "greedy"
    engine.device = "cpu"
    engine.torch_device = "cpu"
    engine._initialized = True
    engine.model = None  # 各テストで差し替え
    engine.progress_callback = None
    engine.model_metadata = {}
    return engine


def _capture_transcribe_kwargs(fake_model: MagicMock) -> dict:
    """fake_model.transcribe の最後の呼び出しの kwargs を返す。"""
    assert fake_model.transcribe.call_args is not None
    return fake_model.transcribe.call_args.kwargs


class TestReturnHypothesesPassedToNeMo:
    """**メインの review 対応**: `return_hypotheses=True` が NeMo に渡ることを pin。"""

    def test_return_hypotheses_true_passed_when_supported(self, fake_engine):
        h = FakeHypothesis(
            text="こんにちは",
            token_confidence=[0.9, 0.8, 0.7],
            score=-1.0,
            y_sequence=[1, 2, 3],
        )
        fake_model = MagicMock()
        fake_model.transcribe.return_value = [h]
        fake_engine.model = fake_model

        audio = np.zeros(16000, dtype=np.float32)  # 1 秒、短すぎチェック回避
        result = fake_engine.transcribe(audio, 16000)

        kwargs = _capture_transcribe_kwargs(fake_model)
        assert kwargs.get("return_hypotheses") is True, (
            "NeMo の transcribe には return_hypotheses=True を渡す必要がある "
            "(codex-review on #309)"
        )

        assert result.text == "こんにちは"
        assert result.engine_confidence.token_confidence_mean == pytest.approx(0.8)
        assert result.engine_confidence.is_available is True


class TestHypothesisResultPopulatesEngineConfidence:
    """Hypothesis を返した時に engine_confidence が正しく populate される end-to-end pin。"""

    def test_token_confidence_extracted_from_hypothesis_list(self, fake_engine):
        h = FakeHypothesis(text="abc", token_confidence=[1.0, 0.6])
        fake_model = MagicMock()
        fake_model.transcribe.return_value = [h]
        fake_engine.model = fake_model

        result = fake_engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        assert result.engine_confidence.token_confidence_mean == pytest.approx(0.8)

    def test_no_score_fallback_when_token_confidence_missing(self, fake_engine):
        """PR #309 smoke verify で削除した score fallback の新挙動を pin。

        旧版: score / len(y_sequence) を avg_logprob に詰めていた。
        新版: token_confidence が取れない場合は全 None で honest に返す
              (score 逆転が filter に有害なため)。
        """
        h = FakeHypothesis(text="x", token_confidence=None, score=-6.0, y_sequence=[1, 2, 3])
        fake_model = MagicMock()
        fake_model.transcribe.return_value = [h]
        fake_engine.model = fake_model

        result = fake_engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        assert result.engine_confidence.is_available is False
        assert result.engine_confidence.token_confidence_mean is None
        assert result.engine_confidence.avg_logprob is None


class TestFallbackWhenReturnHypothesesNotSupported:
    """旧 NeMo API で return_hypotheses kwarg が拒否されたケースの degrade を pin。

    livecap-cli は対応バージョン範囲が広いため、未対応 API では adapter が
    silent crash せず engine_confidence 全 None で生存し続ける必要がある。
    """

    def test_typeerror_triggers_fallback_call_without_kwarg(self, fake_engine, caplog):
        call_count = {"n": 0}

        def transcribe_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                assert kwargs.get("return_hypotheses") is True
                raise TypeError("unexpected keyword argument 'return_hypotheses'")
            # Fallback 呼び出しは return_hypotheses なし、文字列リストを返す
            assert "return_hypotheses" not in kwargs
            return ["plain text fallback"]

        fake_model = MagicMock()
        fake_model.transcribe.side_effect = transcribe_side_effect
        fake_engine.model = fake_model

        with caplog.at_level(logging.INFO, logger="livecap_cli.engines.parakeet_engine"):
            result = fake_engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)

        assert call_count["n"] == 2, "TypeError 時は引数なしで再試行されること"
        assert result.text == "plain text fallback"
        assert result.engine_confidence.is_available is False, (
            "Hypothesis 不在なら engine_confidence は全 None で fail-open"
        )
        assert any(
            "return_hypotheses=True" in rec.message and "rejected" in rec.message
            for rec in caplog.records
        ), "fallback log が出力されること (運用時に気付ける)"


class TestStringResultLeavesEngineConfidenceUnpopulated:
    """NeMo が文字列のみ返すケース (= return_hypotheses=False 同等)。"""

    def test_string_result_does_not_populate_engine_confidence(self, fake_engine):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ["text only"]
        fake_engine.model = fake_model

        result = fake_engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)

        # return_hypotheses=True は渡している (primary path で reject されていないので)
        kwargs = _capture_transcribe_kwargs(fake_model)
        assert kwargs.get("return_hypotheses") is True

        assert result.text == "text only"
        # 文字列だけでは engine_confidence 抽出元がないため全 None
        assert result.engine_confidence.is_available is False

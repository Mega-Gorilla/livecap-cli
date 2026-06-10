"""Parakeet _extract_engine_confidence の pure-function unit test (Issue #308 / PR-A.0).

NeMo Hypothesis を mock し、token_confidence 抽出のみが有効 signal であることを pin。

PR #309 smoke verify で **score-based fallback は speech と non-speech で逆転**
することが判明したため、score fallback は意図的に削除済。本 test は新挙動を pin する。
詳細: docs/research/parakeet-ja-confidence-spec-2026-06-10.md
"""
import importlib
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

parakeet_engine = importlib.import_module(
    "livecap_cli.engines.parakeet_engine"
)
_extract = parakeet_engine._extract_engine_confidence


@dataclass
class FakeHypothesis:
    """NeMo `Hypothesis` の最小 mock (rnnt_utils.py 36-110 行の最小サブセット)。"""
    token_confidence: Optional[List[float]] = None
    score: Optional[float] = None
    y_sequence: Optional[List[int]] = None
    frame_confidence: Optional[Any] = None
    word_confidence: Optional[Any] = None


class TestExtractEngineConfidenceFromHypothesis:
    def test_none_input_returns_all_none(self):
        ec = _extract(None)
        assert ec.is_available is False

    def test_token_confidence_mean_populated(self):
        h = FakeHypothesis(token_confidence=[0.9, 0.8, 0.7])
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.8)
        assert ec.avg_logprob is None  # score fallback は削除済

    def test_token_confidence_single_value(self):
        h = FakeHypothesis(token_confidence=[0.5])
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.5)

    def test_token_confidence_skips_none_entries(self):
        h = FakeHypothesis(token_confidence=[0.8, None, 0.6])
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.7)

    def test_token_confidence_with_tuple_type(self):
        """upstream が tuple で返す可能性に備えた pin。"""
        h = FakeHypothesis(token_confidence=(0.6, 0.4))  # type: ignore[arg-type]
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.5)


class TestScoreFallbackDeprecated:
    """PR #309 smoke verify で削除した score-based fallback の挙動を pin。

    Parakeet `hypothesis.score` は log-prob 累積和 (length-normalization なし)
    で、speech (-71.5/token) と applause (-47.3/token) で逆転する。filter には
    有害な誤情報を返すため、token_confidence が取れない場合は honest に
    `is_available is False` を返す方針に変更。
    """

    def test_score_without_token_confidence_returns_all_none(self):
        """score だけある場合: 旧版は avg_logprob を populate していたが、新版は None。"""
        h = FakeHypothesis(score=-4.0, y_sequence=[1, 2])
        ec = _extract(h)
        assert ec.is_available is False
        assert ec.avg_logprob is None
        assert ec.token_confidence_mean is None
        assert ec.raw == {}  # raw も populate しない (誤情報の温床になるため)

    def test_score_int_type_does_not_populate_anything(self):
        h = FakeHypothesis(score=-6, y_sequence=[1, 2, 3])
        ec = _extract(h)
        assert ec.is_available is False

    def test_empty_token_confidence_falls_through_to_all_none(self):
        """token_confidence=[] でも score fallback は無く、全 None になる。"""
        h = FakeHypothesis(
            token_confidence=[],
            score=-2.0,
            y_sequence=[1, 2, 3, 4],
        )
        ec = _extract(h)
        assert ec.is_available is False

    def test_all_non_numeric_token_confidence_falls_through_to_all_none(self):
        h = FakeHypothesis(
            token_confidence=["bad", None, "values"],
            score=-3.0,
            y_sequence=[1, 2, 3],
        )
        ec = _extract(h)
        assert ec.is_available is False


class TestEdgeCases:
    def test_completely_empty_hypothesis_returns_all_none(self):
        h = FakeHypothesis()
        ec = _extract(h)
        assert ec.is_available is False

    def test_string_input_returns_all_none(self):
        """transcribe path で result が str の場合は engine_confidence なしの想定。"""
        ec = _extract("just a transcript string")
        assert ec.is_available is False

    def test_zero_length_sequence_with_score_returns_all_none(self):
        """seq_len=0 は (旧) raw も populate しないため全 None。"""
        h = FakeHypothesis(score=-1.0, y_sequence=[])
        ec = _extract(h)
        assert ec.is_available is False

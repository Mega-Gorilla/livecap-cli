"""Parakeet _extract_engine_confidence の pure-function unit test (Issue #308 / PR-A.0).

NeMo Hypothesis を mock し、token_confidence 抽出 + score-based fallback の
両方を pin する。実 NeMo / GPU を必要としない。
"""
import importlib
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

parakeet_engine = importlib.import_module(
    "livecap_cli.engines.parakeet_engine"
)
_extract = parakeet_engine._extract_engine_confidence


@dataclass
class FakeHypothesis:
    """NeMo `Hypothesis` の最小 mock。

    rnnt_utils.py 36-110 行の実体に合わせて関連 field のみ持つ。
    """
    token_confidence: Optional[List[float]] = None
    score: Optional[float] = None
    y_sequence: Optional[List[int]] = None
    frame_confidence: Optional[Any] = None
    word_confidence: Optional[Any] = None


class TestExtractEngineConfidenceFromHypothesis:
    def test_none_input_returns_all_none(self):
        ec = _extract(None)
        assert ec.is_available is False

    def test_token_confidence_mean_takes_precedence(self):
        h = FakeHypothesis(
            token_confidence=[0.9, 0.8, 0.7],
            score=-1.0,
            y_sequence=[1, 2, 3],
        )
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.8)
        # 優先順位 1 が hit したら fallback は触らない
        assert ec.avg_logprob is None

    def test_token_confidence_single_value(self):
        h = FakeHypothesis(token_confidence=[0.5])
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.5)

    def test_token_confidence_skips_none_entries(self):
        h = FakeHypothesis(token_confidence=[0.8, None, 0.6])
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.7)

    def test_token_confidence_empty_list_falls_through_to_score(self):
        h = FakeHypothesis(
            token_confidence=[],
            score=-2.0,
            y_sequence=[1, 2, 3, 4],
        )
        ec = _extract(h)
        assert ec.token_confidence_mean is None
        assert ec.avg_logprob == pytest.approx(-0.5)

    def test_token_confidence_all_non_numeric_falls_through_to_score(self):
        h = FakeHypothesis(
            token_confidence=["bad", None, "values"],
            score=-3.0,
            y_sequence=[1, 2, 3],
        )
        ec = _extract(h)
        assert ec.token_confidence_mean is None
        assert ec.avg_logprob == pytest.approx(-1.0)

    def test_score_fallback_when_token_confidence_missing(self):
        h = FakeHypothesis(score=-4.0, y_sequence=[1, 2])
        ec = _extract(h)
        assert ec.token_confidence_mean is None
        assert ec.avg_logprob == pytest.approx(-2.0)
        # raw に score と seq_len が記録される (calibration 用)
        assert ec.raw["parakeet_score"] == pytest.approx(-4.0)
        assert ec.raw["parakeet_seq_len"] == pytest.approx(2.0)

    def test_score_fallback_zero_length_sequence_returns_all_none(self):
        h = FakeHypothesis(score=-1.0, y_sequence=[])
        ec = _extract(h)
        assert ec.is_available is False

    def test_score_only_without_y_sequence_returns_all_none(self):
        h = FakeHypothesis(score=-1.0, y_sequence=None)
        ec = _extract(h)
        assert ec.is_available is False

    def test_y_sequence_only_without_score_returns_all_none(self):
        h = FakeHypothesis(score=None, y_sequence=[1, 2, 3])
        ec = _extract(h)
        assert ec.is_available is False

    def test_completely_empty_hypothesis_returns_all_none(self):
        h = FakeHypothesis()
        ec = _extract(h)
        assert ec.is_available is False

    def test_string_input_returns_all_none(self):
        """transcribe path で result が str の場合は engine_confidence なしの想定。"""
        ec = _extract("just a transcript string")
        assert ec.is_available is False

    def test_token_confidence_with_tuple_type(self):
        """upstream が tuple で返す可能性に備えた pin。"""
        h = FakeHypothesis(token_confidence=(0.6, 0.4))  # type: ignore[arg-type]
        ec = _extract(h)
        assert ec.token_confidence_mean == pytest.approx(0.5)

    def test_score_int_type_accepted(self):
        h = FakeHypothesis(score=-6, y_sequence=[1, 2, 3])
        ec = _extract(h)
        assert ec.avg_logprob == pytest.approx(-2.0)

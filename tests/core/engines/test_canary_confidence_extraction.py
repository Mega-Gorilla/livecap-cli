"""Unit tests for ``livecap_cli.engines.canary_engine._extract_engine_confidence``.

Issue #311 PR-A.4.2 (v2.1) で導入した Canary 用 confidence helper の
pure-function 挙動を pin する。

実 Canary model は load しない。``hypothesis`` を直接 mock した
FakeHypothesis (list[float]) / FakeHypothesisTensor (numpy/Tensor-like) で
``token_confidence_mean`` の計算と Canary-specific な torch.Tensor 入力を
扱う path を test する。

PR-A.0 の Parakeet (`test_parakeet_confidence_extraction.py`) と同じ
mock-based pattern。Phase 1 probe で発覚した型差分 (Canary は
torch.Tensor、Parakeet は List[float]) を本 file で 1 つの helper で扱う
ことを pin する。
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any, List, Optional

import numpy as np
import pytest

from livecap_cli.engines.base_engine import EngineConfidence
from livecap_cli.engines.canary_engine import _extract_engine_confidence


# ---- Fake Hypothesis types ---------------------------------------------------


@dataclass
class FakeHypothesis:
    """Parakeet-style Hypothesis (token_confidence: List[float])."""
    token_confidence: Optional[List[float]] = None
    score: Optional[float] = None


class FakeTensor:
    """``torch.Tensor`` の最小 mock。``.tolist()`` を提供。

    実 torch import を避けるため。Canary の実 hypothesis.token_confidence
    は ``torch.Tensor`` だが、helper は ``hasattr(token_conf, 'tolist')`` で
    防御するため、本 mock で同 path を test できる。
    """

    def __init__(self, values: List[float]) -> None:
        self._values = list(values)

    def tolist(self) -> List[float]:
        return list(self._values)


# ---- Test 1-3: 基本ケース (Parakeet-style list[float]) ----------------------


def test_extracts_token_confidence_mean_from_list():
    """正常 case: token_confidence list populated → 平均計算。"""
    hyp = FakeHypothesis(token_confidence=[0.1, 0.2, 0.3, 0.4, 0.5])
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean == pytest.approx(0.3, abs=1e-9)
    assert ec.no_speech_prob is None
    assert ec.avg_logprob is None
    assert ec.compression_ratio is None


def test_returns_empty_when_hypothesis_is_none():
    """None hypothesis → ``EngineConfidence()`` fallback (= fail-open)。"""
    ec = _extract_engine_confidence(None)
    assert ec == EngineConfidence()
    assert ec.is_available is False


def test_returns_empty_when_token_confidence_is_none():
    """``hypothesis.token_confidence is None`` → fallback。"""
    hyp = FakeHypothesis(token_confidence=None)
    ec = _extract_engine_confidence(hyp)
    assert ec == EngineConfidence()


# ---- Test 4-5: torch.Tensor-like input (Canary 固有) ------------------------


def test_extracts_token_confidence_from_tensor_via_tolist():
    """Canary 実 case: ``token_confidence: torch.Tensor`` を ``.tolist()`` 経由で抽出。

    Phase 1 probe で発覚した Canary 特有 path: NeMo の AED multitask greedy
    は token_confidence を torch.Tensor (GPU) で返す。helper は
    ``hasattr(token_conf, 'tolist')`` で防御し tensor を list 化して処理。
    """
    hyp = FakeHypothesis()
    hyp.token_confidence = FakeTensor([0.05, 0.10, 0.15])  # type: ignore[assignment]
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean == pytest.approx(0.10, abs=1e-9)


def test_handles_numpy_array_via_tolist():
    """numpy.ndarray も ``.tolist()`` 経由で扱える。"""
    hyp = FakeHypothesis()
    hyp.token_confidence = np.array([0.2, 0.4, 0.6], dtype=np.float32)  # type: ignore[assignment]
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean == pytest.approx(0.4, abs=1e-6)


# ---- Test 6-8: 空 / 異常 input の fallback -----------------------------------


def test_returns_empty_when_token_confidence_is_empty_list():
    """空 list → fallback。"""
    hyp = FakeHypothesis(token_confidence=[])
    ec = _extract_engine_confidence(hyp)
    assert ec == EngineConfidence()


def test_returns_empty_when_all_values_are_none():
    """全 None values → fallback (numeric が 0 件)。"""
    hyp = FakeHypothesis(token_confidence=[None, None, None])  # type: ignore[list-item]
    ec = _extract_engine_confidence(hyp)
    assert ec == EngineConfidence()


def test_skips_non_numeric_values_and_averages_remaining():
    """非数値混入 → skip + 残り平均。"""
    hyp = FakeHypothesis()
    hyp.token_confidence = [0.1, "not-a-number", 0.3, None, 0.5]  # type: ignore[list-item]
    ec = _extract_engine_confidence(hyp)
    # 0.1 + 0.3 + 0.5 = 0.9 / 3 = 0.3
    assert ec.token_confidence_mean == pytest.approx(0.3, abs=1e-9)


# ---- Test 9-10: 数値範囲 / 1 token --------------------------------------


def test_handles_single_token():
    """1 token のみ → そのまま。"""
    hyp = FakeHypothesis(token_confidence=[0.42])
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean == pytest.approx(0.42, abs=1e-9)


def test_handles_extreme_values():
    """極端値も正しく平均 (0-1 range の外側も含めて)。"""
    hyp = FakeHypothesis(token_confidence=[0.001, 0.999])
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean == pytest.approx(0.5, abs=1e-9)


# ---- Test 11: frozen dataclass mutation 不能 -------------------------------


def test_result_is_immutable_frozen_dataclass():
    """返り値は frozen dataclass で外部からの mutation 不能。"""
    hyp = FakeHypothesis(token_confidence=[0.1, 0.2])
    ec = _extract_engine_confidence(hyp)
    with pytest.raises(FrozenInstanceError):
        ec.token_confidence_mean = 0.99  # type: ignore[misc]


# ---- Test 12: 他 field は populate されない -----------------------------------


def test_only_token_confidence_mean_is_populated():
    """``no_speech_prob`` / ``avg_logprob`` / ``compression_ratio`` は populate しない。"""
    hyp = FakeHypothesis(token_confidence=[0.5])
    ec = _extract_engine_confidence(hyp)
    assert ec.token_confidence_mean is not None
    assert ec.no_speech_prob is None
    assert ec.avg_logprob is None
    assert ec.compression_ratio is None


# ---- Bonus: tolist が exception を投げる Tensor mock --------------------------


class FakeBrokenTensor:
    """``.tolist()`` が exception を投げる broken tensor mock。"""

    def tolist(self) -> List[float]:
        raise RuntimeError("simulated tolist() failure")


def test_returns_empty_when_tolist_raises():
    """``token_conf.tolist()`` が exception → fallback (model 自体は壊さない)。"""
    hyp = FakeHypothesis()
    hyp.token_confidence = FakeBrokenTensor()  # type: ignore[assignment]
    ec = _extract_engine_confidence(hyp)
    assert ec == EngineConfidence()

"""Unit tests for ``livecap_cli.engines.qwen3asr_engine._extract_engine_confidence``.

Issue #318 PR-A.5.2 で導入した qwen3asr 用 confidence helper の
pure-function 挙動を pin する。Voxtral PR-A.4.1 と同 schema (transition_scores
+ gen_tokens + special_ids → masked mean を avg_logprob field に)。

実 Qwen3-ASR model は load しない。``transition_scores`` / ``gen_tokens`` を
list / numpy / FakeTensor で mock し、masking ロジックを test する。
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import List

import numpy as np
import pytest

from livecap_cli.engines.base_engine import EngineConfidence
from livecap_cli.engines.qwen3asr_engine import _extract_engine_confidence


class FakeTensor:
    """``torch.Tensor`` の最小 mock。``.tolist()`` を提供。

    実 torch import を避けるため (Voxtral test pattern 流用)。
    """

    def __init__(self, values: List) -> None:
        self._values = list(values)

    def tolist(self) -> List:
        return list(self._values)


# ---- Test 1-3: 基本ケース --------------------------------------------------


def test_extracts_avg_logprob_from_masked_scores():
    """正常 case: transition_scores 全 token が non-special → mean が avg_logprob field に。"""
    scores = [-0.1, -0.2, -0.3, -0.4, -0.5]
    tokens = [100, 200, 300, 400, 500]  # special_ids に含まれない
    special_ids = {1, 2, 3}  # EOS/PAD/BOS とは別の ids

    ec = _extract_engine_confidence(scores, tokens, special_ids)
    assert ec.avg_logprob == pytest.approx(-0.3, abs=1e-9)
    assert ec.token_confidence_mean is None
    assert ec.no_speech_prob is None
    assert ec.compression_ratio is None


def test_returns_empty_when_transition_scores_is_none():
    """transition_scores=None → fail-open。"""
    ec = _extract_engine_confidence(None, [100, 200], set())
    assert ec == EngineConfidence()
    assert ec.is_available is False


def test_returns_empty_when_gen_tokens_is_none():
    """gen_tokens=None → fail-open。"""
    ec = _extract_engine_confidence([-0.1, -0.2], None, set())
    assert ec == EngineConfidence()


# ---- Test 4-6: 空 / 全 special / 部分 special -----------------------------


def test_returns_empty_when_scores_list_is_empty():
    """空 list → fallback。"""
    ec = _extract_engine_confidence([], [], set())
    assert ec == EngineConfidence()


def test_returns_empty_when_all_tokens_are_special():
    """全 token が special_ids → masked が空 → fallback。"""
    scores = [-0.1, -0.2, -0.3]
    tokens = [1, 2, 3]
    special_ids = {1, 2, 3}
    ec = _extract_engine_confidence(scores, tokens, special_ids)
    assert ec == EngineConfidence()


def test_partial_special_mask_correctly():
    """一部 special、残りで mean。"""
    scores = [-0.1, -0.2, -0.3, -0.4, -0.5]
    tokens = [1, 200, 300, 400, 2]  # 0番目 (1) と 4 番目 (2) は special
    special_ids = {1, 2}

    ec = _extract_engine_confidence(scores, tokens, special_ids)
    # masked = [-0.2, -0.3, -0.4] / 3 = -0.3
    assert ec.avg_logprob == pytest.approx(-0.3, abs=1e-9)


# ---- Test 7-9: tensor 互換 + numpy + extreme values ------------------------


def test_handles_tensor_like_inputs():
    """FakeTensor (`.tolist()`) を tensor-compat path で扱える。"""
    scores = FakeTensor([-0.05, -0.10, -0.15])
    tokens = FakeTensor([100, 200, 300])
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.10, abs=1e-9)


def test_handles_numpy_arrays():
    """numpy.ndarray (tolist あり) で扱える。"""
    scores = np.array([-0.2, -0.4, -0.6], dtype=np.float32)
    tokens = np.array([100, 200, 300], dtype=np.int64)
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.4, abs=1e-6)


def test_handles_single_token():
    """1 token のみ → そのまま。"""
    ec = _extract_engine_confidence([-0.42], [100], set())
    assert ec.avg_logprob == pytest.approx(-0.42, abs=1e-9)


# ---- Test 10: frozen / 他 field populate されない ------------------------


def test_result_is_immutable_frozen_dataclass():
    """返り値は frozen dataclass、外部からの mutation 不能。"""
    ec = _extract_engine_confidence([-0.1, -0.2], [100, 200], set())
    with pytest.raises(FrozenInstanceError):
        ec.avg_logprob = -0.99  # type: ignore[misc]


def test_does_not_populate_other_fields():
    """``no_speech_prob`` / ``token_confidence_mean`` / ``compression_ratio`` には詰めない。"""
    ec = _extract_engine_confidence([-0.1, -0.2], [100, 200], set())
    assert ec.avg_logprob is not None
    assert ec.no_speech_prob is None
    assert ec.token_confidence_mean is None
    assert ec.compression_ratio is None


# ---- Test 11-12: Phase 1 probe 値の再現 ---------------------------------


def test_probe_japanese_speech_realistic_value():
    """Phase 1 probe で確認した JP speech avg_logprob (約 -0.20) を helper で再現できる。"""
    # neko 60 token の場合 (probe 結果)
    scores = [-0.15] * 60
    tokens = list(range(100, 160))
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.15, abs=1e-9)


def test_probe_english_applause_realistic_value():
    """Phase 1 probe で confirmed EN applause avg_logprob (~-1.08 with rep_penalty+ngram) を helper で扱える。"""
    # applause "You are an AI." 6 tokens
    scores = [-0.9, -1.2, -1.0, -1.1, -1.2, -1.0]
    tokens = [100, 200, 300, 400, 500, 600]
    ec = _extract_engine_confidence(scores, tokens, set())
    # mean ~= -1.07 (close to probe -1.0795)
    assert ec.avg_logprob == pytest.approx(-1.0666666, abs=1e-5)


# ---- Test 13: 非数値混入の堅牢性 -----------------------------------------


def test_handles_int_score_values():
    """score が int でも float に変換される。"""
    scores = [-1, -2, 0]  # 全部 int
    tokens = [100, 200, 300]
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-1.0, abs=1e-9)

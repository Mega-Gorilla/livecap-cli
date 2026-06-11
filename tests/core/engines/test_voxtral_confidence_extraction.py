"""Unit tests for ``livecap_cli.engines.voxtral_engine._extract_engine_confidence``.

Issue #311 PR-A.4.1 (v2.1) で導入した Voxtral 用 confidence helper の
pure-function 挙動を pin する。

実 Voxtral model は load しない (engine_smoke は別 file)。
``transition_scores`` / ``gen_tokens`` を直接 mock した list / np.ndarray /
torch.Tensor で渡し、``avg_logprob`` の計算と special token 除外を test する。

PR-A.0 の WhisperS2T (``test_whispers2t_confidence_extraction.py``) /
Parakeet (``test_parakeet_confidence_extraction.py``) と同じ pattern。
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from livecap_cli.engines.base_engine import EngineConfidence
from livecap_cli.engines.voxtral_engine import _extract_engine_confidence


# ---- Test 1-3: 基本ケース ---------------------------------------------------


def test_extracts_avg_logprob_when_all_tokens_non_special():
    """全 token speech (special なし) → 全 token の平均を avg_logprob に。"""
    scores = [-0.5, -0.3, -0.7]
    tokens = [10, 20, 30]
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.5, abs=1e-9)
    # 他 field は None
    assert ec.no_speech_prob is None
    assert ec.token_confidence_mean is None
    assert ec.compression_ratio is None


def test_masks_special_tokens_and_averages_remaining():
    """一部 special 混入 → masked 平均 (special token は除外)。"""
    scores = [-0.5, -10.0, -0.7]  # 真ん中は EOS
    tokens = [10, 2, 30]          # 2 が EOS
    ec = _extract_engine_confidence(scores, tokens, {2})
    # -0.5 と -0.7 のみ平均 = -0.6
    assert ec.avg_logprob == pytest.approx(-0.6, abs=1e-9)


def test_returns_empty_when_all_tokens_are_special():
    """全 token が special → ``EngineConfidence()`` fallback (= fail-open)。"""
    scores = [-0.1, -0.2]
    tokens = [0, 1]  # 0=PAD, 1=EOS と仮定
    ec = _extract_engine_confidence(scores, tokens, {0, 1, 2})
    assert ec == EngineConfidence()
    assert ec.is_available is False


# ---- Test 4-6: None / 空 input の fallback -----------------------------------


def test_returns_empty_when_transition_scores_is_none():
    ec = _extract_engine_confidence(None, [10, 20], {2})
    assert ec == EngineConfidence()


def test_returns_empty_when_gen_tokens_is_none():
    ec = _extract_engine_confidence([-0.5, -0.3], None, {2})
    assert ec == EngineConfidence()


def test_returns_empty_when_empty_tensors():
    ec = _extract_engine_confidence([], [], {2})
    assert ec == EngineConfidence()


# ---- Test 7-8: tensor 互換性 (numpy / list / torch.Tensor-like) -------------


def test_handles_numpy_arrays():
    """numpy.ndarray が渡されても tolist() で list 化される。"""
    scores = np.array([-0.5, -0.3, -0.7], dtype=np.float32)
    tokens = np.array([10, 20, 30], dtype=np.int64)
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.5, abs=1e-6)


def test_handles_plain_python_lists():
    """list でも tolist() なしで動く。"""
    ec = _extract_engine_confidence([-0.4], [42], set())
    assert ec.avg_logprob == pytest.approx(-0.4, abs=1e-9)


# ---- Test 9-10: 数値範囲 / dtype ---------------------------------------------


def test_handles_extreme_logprob_values():
    """大きい負値 (低信頼度 token) も正しく平均。"""
    scores = [-15.0, -20.0, -25.0]
    tokens = [10, 20, 30]
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-20.0, abs=1e-9)


def test_handles_positive_logprob_edge_case():
    """log-prob が ~0 に近い (確実性極高) → 正しく平均。"""
    scores = [-0.001, -0.002, -0.003]
    tokens = [10, 20, 30]
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.002, abs=1e-9)


# ---- Test 11: special_ids 空集合 --------------------------------------------


def test_empty_special_ids_means_all_tokens_used():
    """special_ids が空集合なら全 token を平均 (mask が常に True)。"""
    scores = [-0.5, -0.3]
    tokens = [10, 20]
    ec = _extract_engine_confidence(scores, tokens, set())
    assert ec.avg_logprob == pytest.approx(-0.4, abs=1e-9)


# ---- Test 12: frozen dataclass mutation 不能 -------------------------------


def test_result_is_immutable_frozen_dataclass():
    """返り値は frozen dataclass で外部からの mutation 不能 (defense-in-depth)。"""
    ec = _extract_engine_confidence([-0.5], [10], set())
    with pytest.raises(FrozenInstanceError):
        ec.avg_logprob = -99.0  # type: ignore[misc]


# ---- Bonus: 1 token のみ ----------------------------------------------------


def test_single_token_returns_that_value_as_average():
    """1 token のみのケース (短い utterance) でも壊れず単一値を返す。"""
    ec = _extract_engine_confidence([-1.234], [42], set())
    assert ec.avg_logprob == pytest.approx(-1.234, abs=1e-9)


# ---- Bonus: special が 1 個だけ含まれる -------------------------------------


def test_with_only_eos_at_end_works_normally():
    """生成最終 token が EOS だけのケース (典型的な ASR 生成)。"""
    scores = [-0.5, -0.3, -0.7, -0.01]  # 最終 EOS は超高信頼
    tokens = [10, 20, 30, 99]  # 99=EOS
    ec = _extract_engine_confidence(scores, tokens, {99})
    # -0.5 + -0.3 + -0.7 = -1.5 / 3 = -0.5
    assert ec.avg_logprob == pytest.approx(-0.5, abs=1e-9)

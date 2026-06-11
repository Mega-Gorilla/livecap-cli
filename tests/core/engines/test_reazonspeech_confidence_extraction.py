"""Unit tests for ``livecap_cli.engines.reazonspeech_engine._extract_engine_confidence``.

Issue #317 PR-A.5.1 で導入した ReazonSpeech 用 confidence helper の
pure-function 挙動を pin する。

実 sherpa-onnx model は load しない。``OfflineRecognitionResult`` を直接
mock した FakeResult で ``ys_log_probs`` mean 計算 path を test する。

Canary PR-A.4.2 / Voxtral PR-A.4.1 と同 pattern。

reviewer feedback (Issue #317) Point 1 で確定した設計:
- ``ys_log_probs`` mean は ``EngineConfidence.avg_logprob`` field に
  populate (Voxtral と同 semantics、負の log probability、低いほど悪い)
- ``EngineConfidence.token_confidence_mean`` (probability 0-1 range) には
  詰めない (詰めると ``token_conf_threshold = 0.005`` 比較で speech が
  全 reject される critical bug)
- ``raw["ys_log_probs_mean"]`` + ``raw["ys_log_probs_n"]`` に metadata 保存
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, List, Optional

import numpy as np
import pytest

from livecap_cli.engines.base_engine import EngineConfidence
from livecap_cli.engines.reazonspeech_engine import _extract_engine_confidence


# ---- Fake sherpa-onnx result types ------------------------------------------


@dataclass
class FakeResult:
    """sherpa-onnx ``OfflineRecognitionResult`` の最小 mock。

    sherpa-onnx 1.12.39 で expose される ``ys_log_probs`` (per-token log prob)
    field を mock する。``text`` は本 helper では参照しないが完全性のため保持。
    """
    ys_log_probs: Any = None
    text: str = ""


# ---- Test 1-3: 基本ケース (正常 list, None, empty) --------------------------


def test_extracts_avg_logprob_from_ys_log_probs_list():
    """正常 case: ys_log_probs list populated → mean 計算 → avg_logprob field 詰める。"""
    fake = FakeResult(ys_log_probs=[-0.1, -0.2, -0.3, -0.4, -0.5])
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob == pytest.approx(-0.3, abs=1e-9)
    assert ec.token_confidence_mean is None
    assert ec.no_speech_prob is None
    assert ec.compression_ratio is None
    # raw[] にも保存 (Issue #317 Point 2)
    assert ec.raw == {
        "ys_log_probs_mean": pytest.approx(-0.3, abs=1e-9),
        "ys_log_probs_n": 5,
    }


def test_returns_empty_when_result_is_none():
    """None result → ``EngineConfidence()`` fallback (= fail-open)。"""
    ec = _extract_engine_confidence(None)
    assert ec == EngineConfidence()
    assert ec.is_available is False


def test_returns_empty_when_ys_log_probs_attribute_missing():
    """``ys_log_probs`` attribute なし (旧 sherpa-onnx 互換性) → fallback。"""
    class BareResult:
        text = "foo"

    ec = _extract_engine_confidence(BareResult())
    assert ec == EngineConfidence()


# ---- Test 4-6: 異常 input の fallback -----------------------------------------


def test_returns_empty_when_ys_log_probs_is_none():
    """``result.ys_log_probs is None`` → fallback。"""
    fake = FakeResult(ys_log_probs=None)
    ec = _extract_engine_confidence(fake)
    assert ec == EngineConfidence()


def test_returns_empty_when_ys_log_probs_is_empty_list():
    """空 list → fallback (mean 計算不可)。"""
    fake = FakeResult(ys_log_probs=[])
    ec = _extract_engine_confidence(fake)
    assert ec == EngineConfidence()


def test_returns_empty_when_all_values_are_none():
    """全 None values → fallback (numeric が 0 件)。"""
    fake = FakeResult(ys_log_probs=[None, None, None])
    ec = _extract_engine_confidence(fake)
    assert ec == EngineConfidence()


def test_skips_non_numeric_values_and_averages_remaining():
    """非数値混入 → skip + 残り平均。"""
    fake = FakeResult(ys_log_probs=[-0.1, "not-a-number", -0.3, None, -0.5])
    ec = _extract_engine_confidence(fake)
    # -0.1 + -0.3 + -0.5 = -0.9 / 3 = -0.3
    assert ec.avg_logprob == pytest.approx(-0.3, abs=1e-9)
    assert ec.raw["ys_log_probs_n"] == 3


# ---- Test 7-8: 数値 / 1 token / 極端値 ------------------------------------


def test_handles_single_token():
    """1 token のみ → そのまま (mean は単一値)。"""
    fake = FakeResult(ys_log_probs=[-0.42])
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob == pytest.approx(-0.42, abs=1e-9)
    assert ec.raw["ys_log_probs_n"] == 1


def test_handles_speech_probe_realistic_values():
    """probe で確認した実値 (jsut sample: 22 tokens, mean ~-0.075) を mean できる。

    PR-A.5.1 plan 段階の実機 probe 結果を再現:
    - jsut_basic5000_0001 → 22 tokens、mean=-0.0753
    """
    # 22 tokens、mean が -0.0753 になるよう調整した sample
    fake = FakeResult(ys_log_probs=[-0.025, -0.001, -0.000, -0.021, -0.022, -0.215, -0.023, -0.026, -0.135, -0.000, -0.080, -0.030, -0.040, -0.150, -0.020, -0.060, -0.100, -0.060, -0.090, -0.140, -0.225, -0.193])
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob is not None
    # 期待 ~-0.075 (probe 値、絶対値は近似)
    assert -0.10 < ec.avg_logprob < -0.05
    assert ec.raw["ys_log_probs_n"] == 22


# ---- Test 9: numpy ndarray support (sherpa-onnx 内部実装変化 防御) ---------


def test_handles_numpy_array_via_list_conversion():
    """``ys_log_probs`` が numpy.ndarray でも ``list(ys)`` で扱える。"""
    fake = FakeResult(ys_log_probs=np.array([-0.1, -0.2, -0.3], dtype=np.float32))
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob == pytest.approx(-0.2, abs=1e-6)
    assert ec.raw["ys_log_probs_n"] == 3


# ---- Test 10: frozen dataclass mutation 不能 -------------------------------


def test_result_is_immutable_frozen_dataclass():
    """返り値は frozen dataclass、外部からの mutation 不能。"""
    fake = FakeResult(ys_log_probs=[-0.1, -0.2])
    ec = _extract_engine_confidence(fake)
    with pytest.raises(FrozenInstanceError):
        ec.avg_logprob = -0.99  # type: ignore[misc]


# ---- Test 11: 他 field は populate されない (semantics 分離) -----------------


def test_does_not_populate_token_confidence_mean_field():
    """``token_confidence_mean`` (probability 0-1 range) には詰めない。

    reviewer Point 1 (CRITICAL): ys_log_probs (負の log prob) を
    token_confidence_mean (probability 0-1) field に詰めると
    ``token_conf_threshold = 0.005`` 比較で speech も全 reject される
    critical bug が発生する。本 test で field 分離を pin。
    """
    fake = FakeResult(ys_log_probs=[-0.1, -0.2])
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob is not None  # 詰める
    assert ec.token_confidence_mean is None  # 詰めない (Point 1)
    assert ec.no_speech_prob is None
    assert ec.compression_ratio is None


# ---- Test 12: int 型 value 混入も扱える (型 robust) -------------------------


def test_handles_mixed_int_and_float_values():
    """list に int が混じっても float() 変換で扱える。"""
    fake = FakeResult(ys_log_probs=[-1, -0.5, 0])  # int + float の mix
    ec = _extract_engine_confidence(fake)
    assert ec.avg_logprob == pytest.approx(-0.5, abs=1e-9)
    assert ec.raw["ys_log_probs_n"] == 3

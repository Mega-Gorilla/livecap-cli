"""WhisperS2T _extract_engine_confidence の pure-function unit test (Issue #308 / PR-A.0).

実 model を load せず、CTranslate2 backend 戻り値の dict 構造を mock して
schema 抽出ロジックを pin する。
"""
import importlib
import math
import sys

import pytest

# whispers2t_engine モジュールは whisper_s2t 依存を持つため、import 段階で
# library が不足していると ImportError になる可能性がある。
# `_extract_engine_confidence` 自体は依存を持たない pure-function なので、
# load 時に whisper_s2t がなくても module 全体は import 可能 (lazy import)。
whispers2t_engine = importlib.import_module(
    "livecap_cli.engines.whispers2t_engine"
)
_extract = whispers2t_engine._extract_engine_confidence


class TestExtractEngineConfidence:
    def test_non_dict_returns_all_none(self):
        ec = _extract("just a string")
        assert ec.is_available is False
        assert ec.no_speech_prob is None
        assert ec.avg_logprob is None
        assert ec.compression_ratio is None

    def test_dict_without_segments_returns_all_none(self):
        ec = _extract({"text": "hello"})
        assert ec.is_available is False

    def test_dict_with_empty_segments_returns_all_none(self):
        ec = _extract({"text": "hi", "segments": []})
        assert ec.is_available is False

    def test_dict_with_segments_missing_metric_fields_returns_all_none(self):
        result = {
            "text": "hi",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi"},
            ],
        }
        ec = _extract(result)
        assert ec.is_available is False

    def test_single_segment_with_avg_logprob(self):
        result = {
            "text": "hi",
            "segments": [
                {"avg_logprob": -0.5},
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.5)
        assert ec.no_speech_prob is None
        assert ec.compression_ratio is None

    def test_multi_segment_avg_logprob_mean(self):
        result = {
            "segments": [
                {"avg_logprob": -0.2},
                {"avg_logprob": -0.4},
                {"avg_logprob": -0.6},
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.4)

    def test_multi_segment_no_speech_prob_mean(self):
        result = {
            "segments": [
                {"no_speech_prob": 0.1},
                {"no_speech_prob": 0.5},
            ],
        }
        ec = _extract(result)
        assert ec.no_speech_prob == pytest.approx(0.3)
        assert ec.avg_logprob is None

    def test_compression_ratio_mean(self):
        result = {
            "segments": [
                {"compression_ratio": 1.2},
                {"compression_ratio": 1.8},
            ],
        }
        ec = _extract(result)
        assert ec.compression_ratio == pytest.approx(1.5)

    def test_all_three_signals_present(self):
        result = {
            "segments": [
                {
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.1,
                    "compression_ratio": 1.4,
                },
                {
                    "avg_logprob": -0.7,
                    "no_speech_prob": 0.3,
                    "compression_ratio": 1.6,
                },
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.5)
        assert ec.no_speech_prob == pytest.approx(0.2)
        assert ec.compression_ratio == pytest.approx(1.5)
        assert ec.is_available is True

    def test_partial_segment_fields_skipped_safely(self):
        """segments の一部だけが metric を持つ場合、その mean だけが取られる。"""
        result = {
            "segments": [
                {"avg_logprob": -0.4, "no_speech_prob": 0.1},
                {"avg_logprob": -0.6},  # no_speech_prob 欠落
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.5)
        assert ec.no_speech_prob == pytest.approx(0.1)
        assert ec.compression_ratio is None

    def test_non_numeric_value_is_skipped(self):
        result = {
            "segments": [
                {"avg_logprob": "not a number"},
                {"avg_logprob": -0.5},
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.5)

    def test_none_value_is_skipped(self):
        result = {
            "segments": [
                {"avg_logprob": None},
                {"avg_logprob": -0.4},
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.4)

    def test_non_dict_segment_is_skipped(self):
        result = {
            "segments": [
                "garbage",
                {"avg_logprob": -0.3},
            ],
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.3)


class TestTopLevelSignals:
    """実機 smoke verify (#309) で発覚した「CTranslate2 backend は signal を
    top-level に置く」構造を pin する。

    旧 test では segments 内のみを想定していたため、実機で全 None になっていた。
    """

    def test_top_level_no_speech_prob_alone(self):
        result = {
            "text": "hello",
            "no_speech_prob": 0.25,
            "start_time": 0.0,
            "end_time": 1.5,
        }
        ec = _extract(result)
        assert ec.no_speech_prob == pytest.approx(0.25)
        assert ec.avg_logprob is None
        assert ec.compression_ratio is None

    def test_top_level_all_three_signals(self):
        """smoke verify で実観測した top-level dict 構造をそのまま pin。"""
        result = {
            "text": "わが輩はねこである",
            "avg_logprob": -0.18,
            "no_speech_prob": 0.04,
            "compression_ratio": 1.6,
            "start_time": 0.0,
            "end_time": 15.6,
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.18)
        assert ec.no_speech_prob == pytest.approx(0.04)
        assert ec.compression_ratio == pytest.approx(1.6)
        assert ec.is_available is True

    def test_top_level_segments_none_does_not_break(self):
        """smoke verify で観測した ``segments: None`` の戻り値で全 None に
        retreat しないこと (= 旧 bug の regression test)。"""
        result = {
            "text": "x",
            "avg_logprob": -0.5,
            "no_speech_prob": 0.1,
            "segments": None,
        }
        ec = _extract(result)
        assert ec.avg_logprob == pytest.approx(-0.5)
        assert ec.no_speech_prob == pytest.approx(0.1)

    def test_top_level_plus_segments_mean_together(self):
        """両方の structure を持つ仮想ケース: top-level 値 + segment 値を mean。"""
        result = {
            "avg_logprob": -0.6,
            "segments": [
                {"avg_logprob": -0.4},
                {"avg_logprob": -0.2},
            ],
        }
        ec = _extract(result)
        # 3 値 (-0.6, -0.4, -0.2) の mean = -0.4
        assert ec.avg_logprob == pytest.approx(-0.4)

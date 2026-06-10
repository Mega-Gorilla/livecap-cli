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

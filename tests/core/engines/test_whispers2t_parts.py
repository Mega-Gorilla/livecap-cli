"""WhisperS2T engine: WhisperS2T の VAD を使わず、返った部分を全部使う (#518)。

WhisperS2T は音声を最長 29 秒ずつに分けて書き起こし、部分を全部リストで返す。以前の engine は
``transcribe_with_vad`` (WhisperS2T の VAD を LiveCap の VAD に重ねる) を呼び、最初の部分だけを使って
2 つ目以降の文を捨てていた。モデルは読まず、WhisperS2T の戻り値の形だけを偽物で与える。
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from livecap_cli.engines import whispers2t_engine
from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.engines.metadata import EngineMetadata
from livecap_cli.engines.whispers2t_engine import (
    WhisperS2TEngine,
    _combine_parts,
    _extract_engine_confidence,
    _file_parts,
)


def _part(text, start, end, *, no_speech_prob=None, avg_logprob=None):
    """CTranslate2 backend の 1 つの部分 (top-level に信号が載る形、#309)。"""
    part = {"text": text, "start_time": start, "end_time": end, "segments": None}
    if no_speech_prob is not None:
        part["no_speech_prob"] = no_speech_prob
    if avg_logprob is not None:
        part["avg_logprob"] = avg_logprob
    return part


class TestFileParts:
    def test_takes_every_part_of_the_file(self):
        parts = [_part("a", 0, 29.0), _part("b", 29, 57.5)]
        assert _file_parts([parts]) == parts

    def test_empty_or_missing_output_has_no_parts(self):
        assert _file_parts([]) == []
        assert _file_parts(None) == []
        assert _file_parts([[]]) == []

    def test_a_bare_result_is_one_part(self):
        assert _file_parts([{"text": "a"}]) == [{"text": "a"}]


class TestCombineParts:
    def test_english_parts_are_joined_with_a_space(self):
        text, _, _ = _combine_parts([_part("Thank you, Sadie.", 0, 29.0), _part("Good afternoon.", 29, 57.5)], "en")
        assert text == "Thank you, Sadie. Good afternoon."

    def test_japanese_parts_are_joined_without_a_space(self):
        text, _, _ = _combine_parts([_part("こんにちは。", 0, 29.0), _part("今日は", 29, 40.0)], "ja")
        assert text == "こんにちは。今日は"

    def test_confidence_is_the_duration_weighted_mean_of_the_parts(self):
        parts = [
            _part("a", 0, 29.0, no_speech_prob=0.1, avg_logprob=-0.2),
            _part("b", 29, 39.0, no_speech_prob=0.5, avg_logprob=-1.0),
        ]
        _, confidence, ec = _combine_parts(parts, "en")
        assert ec.no_speech_prob == pytest.approx((0.1 * 29 + 0.5 * 10) / 39)
        assert ec.avg_logprob == pytest.approx((-0.2 * 29 + -1.0 * 10) / 39)
        assert ec.compression_ratio is None
        assert confidence == 1.0  # segments が無い (CTranslate2) ので以前と同じ 1.0

    def test_parts_without_text_are_left_out_of_the_text_and_the_confidence(self):
        """ReazonSpeech の分割と同じく、文の無い部分は confidence を下げない。"""
        parts = [
            _part("a", 0, 29.0, no_speech_prob=0.2),
            _part("", 29, 58.0, no_speech_prob=0.95),
            _part("c", 58, 60.0, no_speech_prob=0.4),
        ]
        text, _, ec = _combine_parts(parts, "en")
        assert text == "a c"
        assert ec.no_speech_prob == pytest.approx((0.2 * 29 + 0.4 * 2) / 31)

    def test_without_any_text_the_confidence_of_every_part_is_kept(self):
        """文が無くても、filter が見る値は残す (全部の部分をまとめる)。"""
        parts = [_part("", 0, 29.0, no_speech_prob=0.9), _part(" ", 29, 30.0, no_speech_prob=0.7)]
        text, _, ec = _combine_parts(parts, "en")
        assert text == ""
        assert ec.no_speech_prob == pytest.approx((0.9 * 29 + 0.7 * 1) / 30)

    def test_a_single_part_keeps_its_own_values(self):
        part = _part("  hello  ", 0, 3.0, no_speech_prob=0.3, avg_logprob=-0.4)
        text, confidence, ec = _combine_parts([part], "en")
        assert text == "hello"
        assert ec == _extract_engine_confidence(part)
        assert confidence == 1.0

    def test_parts_without_times_get_equal_weights(self):
        parts = [{"text": "a", "no_speech_prob": 0.2}, _part("b", 29, 39.0, no_speech_prob=0.6)]
        _, _, ec = _combine_parts(parts, "en")
        assert ec.no_speech_prob == pytest.approx(0.4)

    def test_legacy_confidence_uses_the_segments_of_every_spoken_part(self):
        parts = [
            {"text": "a", "segments": [{"avg_logprob": -0.2}]},
            {"text": "b", "segments": [{"avg_logprob": -0.4}]},
        ]
        _, confidence, _ = _combine_parts(parts, "en")
        assert confidence == pytest.approx(math.exp(-0.3))


def _engine(tmp_path: Path, language: str = "en") -> WhisperS2TEngine:
    """モデルを読まない engine (``test_utterance_wav_lifetime`` と同じ組み立て方)。"""
    engine = WhisperS2TEngine.__new__(WhisperS2TEngine)
    engine._tmp_dir = tmp_path
    engine._enable_profiling = False
    engine._asr_language = language
    engine.batch_size = 1
    engine.device = "cuda"
    engine._initialized = True
    engine.model = MagicMock()
    return engine


class TestEngineTranscribe:
    def test_uses_every_part_and_not_the_builtin_vad(self, tmp_path):
        engine = _engine(tmp_path)
        engine.model.transcribe.return_value = [[
            _part("first half", 0, 29.0, no_speech_prob=0.1),
            _part("second half", 29, 57.5, no_speech_prob=0.3),
        ]]

        result = engine.transcribe(np.zeros(16000 * 58, dtype=np.float32), 16000)

        assert result.text == "first half second half"
        assert result.engine_confidence.no_speech_prob == pytest.approx((0.1 * 29 + 0.3 * 28.5) / 57.5)
        engine.model.transcribe.assert_called_once()
        engine.model.transcribe_with_vad.assert_not_called()

    @pytest.mark.parametrize("seconds", [0.1, 0.5, 0.99])
    def test_audio_shorter_than_a_second_is_padded_to_a_second(self, tmp_path, seconds):
        """WhisperS2T 1.3.1 の VAD を使わない分け方は、1 秒未満で IndexError を出す。"""
        engine = _engine(tmp_path)
        frames = []

        def fake_transcribe(paths, **kwargs):
            frames.append(sf.info(paths[0]).frames)
            return [[_part("yes", 0, 1.0)]]

        engine.model.transcribe.side_effect = fake_transcribe

        result = engine.transcribe(np.full(int(16000 * seconds), 0.1, dtype=np.float32), 16000)

        assert result.text == "yes"
        assert frames == [16000]

    def test_audio_of_a_second_or_more_is_passed_as_is(self, tmp_path):
        engine = _engine(tmp_path)
        frames = []

        def fake_transcribe(paths, **kwargs):
            frames.append(sf.info(paths[0]).frames)
            return [[_part("ok", 0, 1.5)]]

        engine.model.transcribe.side_effect = fake_transcribe
        engine.transcribe(np.zeros(24000, dtype=np.float32), 16000)
        assert frames == [24000]

    def test_audio_shorter_than_the_minimum_is_still_skipped(self, tmp_path):
        engine = _engine(tmp_path)
        result = engine.transcribe(np.zeros(800, dtype=np.float32), 16000)  # 0.05 秒
        assert result.text == ""
        engine.model.transcribe.assert_not_called()

    def test_no_parts_is_an_empty_result(self, tmp_path):
        engine = _engine(tmp_path)
        engine.model.transcribe.return_value = [[]]
        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        assert result.text == ""


class TestNoUseVadOption:
    def test_the_engine_rejects_use_vad(self):
        with pytest.raises(TypeError, match="use_vad"):
            WhisperS2TEngine(use_vad=False)

    def test_the_factory_rejects_use_vad(self):
        with pytest.raises(TypeError, match="use_vad"):
            EngineFactory.create_engine("whispers2t", device="cpu", use_vad=True)

    def test_metadata_has_no_use_vad_default(self):
        assert "use_vad" not in EngineMetadata.get("whispers2t").default_params

    def test_the_engine_module_never_calls_the_builtin_vad(self):
        """WhisperS2T の VAD (``transcribe_with_vad``) を呼ぶ行が、engine に戻ってこないようにする。"""
        source = Path(whispers2t_engine.__file__).read_text(encoding="utf-8")
        calls = [line for line in source.splitlines() if ".transcribe_with_vad(" in line]
        assert calls == []

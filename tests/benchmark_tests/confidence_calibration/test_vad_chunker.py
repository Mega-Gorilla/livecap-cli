"""Tests for ``benchmarks.confidence_calibration._vad_chunker`` (Issue #338 PR-β)。

``detect_speech_segments()`` を pure logic として合成 probability stream で test。
``compute_vad_probabilities()`` / ``chunk_audio_by_vad()`` は SileroVAD 実 model
が必要なため engine_smoke marker でも gated 可能 (本 PR では mock VAD で代替)。
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.confidence_calibration._vad_chunker import (
    FRAME_SEC,
    FRAME_SIZE,
    SAMPLE_RATE,
    chunk_audio_by_vad,
    compute_vad_probabilities,
    detect_speech_segments,
)


# ----------------- detect_speech_segments (pure logic) ------------------


class TestDetectSpeechSegmentsBasic:
    def test_empty_input(self):
        assert detect_speech_segments([]) == []

    def test_all_silence(self):
        probs = [0.1] * 100
        assert detect_speech_segments(probs) == []

    def test_all_speech(self):
        # 100 frames × 0.032s = 3.2 sec、max_segment_sec=3.0 で 2 chunk
        probs = [0.9] * 100
        segments = detect_speech_segments(
            probs, max_segment_sec=3.0, min_silence_sec=0.3
        )
        # 100 frames 連続 speech、3.0 sec (94 frames) を超えるので 2 chunk に split
        assert len(segments) >= 1
        # 全 frames が cover される
        total_frames = sum(end - start for start, end in segments)
        assert total_frames == 100

    def test_single_speech_segment(self):
        # 0-10 silence、10-40 speech、40-50 silence
        probs = [0.1] * 10 + [0.9] * 30 + [0.1] * 10
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,  # 16 frames 以上
            min_silence_sec=0.3,  # 10 frames 以上
        )
        assert len(segments) == 1
        start, end = segments[0]
        assert start == 10
        # end は silence が min_silence_frames (10) 累積して確定、つまり 40
        assert end == 40


class TestDetectSpeechSegmentsHysteresis:
    def test_short_silence_does_not_split(self):
        """min_silence_sec より短い silence は segment を切らない。"""
        # 10 silence + 30 speech + 5 silence (< 10 frames) + 30 speech + 10 silence
        probs = [0.1] * 10 + [0.9] * 30 + [0.1] * 5 + [0.9] * 30 + [0.1] * 10
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,
            max_segment_sec=10.0,  # 大きく取って split を防ぐ
            min_silence_sec=0.3,  # 10 frames 必要、5 frames では切れない
        )
        # 短 silence は無視され 1 segment に統合される
        assert len(segments) == 1
        start, end = segments[0]
        assert start == 10
        # 末尾 silence が min_silence 確定し end ≈ 75 (10 silence までで segment 終了)
        assert end == 75

    def test_long_silence_splits(self):
        """min_silence_sec 以上の silence は segment を切る。"""
        # 10 silence + 30 speech + 15 silence (> 10) + 30 speech + 10 silence
        probs = [0.1] * 10 + [0.9] * 30 + [0.1] * 15 + [0.9] * 30 + [0.1] * 10
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,
            max_segment_sec=10.0,
            min_silence_sec=0.3,
        )
        assert len(segments) == 2

    def test_too_short_speech_dropped(self):
        """min_speech_sec 未満の speech は drop。"""
        # 3 frames だけ speech (< min_speech_frames=16)
        probs = [0.1] * 10 + [0.9] * 3 + [0.1] * 20
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,  # 16 frames
            min_silence_sec=0.3,
        )
        assert segments == []


class TestDetectSpeechSegmentsMaxSegmentSplit:
    def test_long_speech_split_into_chunks(self):
        """max_segment_sec を超える speech は均等 split。"""
        # 200 frames 連続 speech = 6.4 sec、max_segment_sec=3.0 → 2 chunk (各 100)
        probs = [0.9] * 200
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.1,
            max_segment_sec=3.0,
            min_silence_sec=0.3,
        )
        assert len(segments) >= 2
        # 各 chunk の frames は max_segment_frames (94) 以下
        max_frames = int(round(3.0 / FRAME_SEC))
        for start, end in segments:
            assert end - start <= max_frames + 1  # 端数許容

    def test_short_speech_not_split(self):
        """max_segment_sec 以下なら split しない。"""
        # 30 frames = 0.96 sec、max_segment_sec=3.0
        probs = [0.1] * 10 + [0.9] * 30 + [0.1] * 10
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,
            max_segment_sec=3.0,
            min_silence_sec=0.3,
        )
        assert len(segments) == 1


class TestDetectSpeechSegmentsEdgeCases:
    def test_tail_speech_without_trailing_silence(self):
        """audio 末尾が speech のまま終わる case (trailing silence 無し)。"""
        # 10 silence + 30 speech (末尾、silence 無し)
        probs = [0.1] * 10 + [0.9] * 30
        segments = detect_speech_segments(
            probs,
            min_speech_sec=0.5,
            min_silence_sec=0.3,
        )
        # in_speech のまま終了 → tail 処理で segment 確定
        assert len(segments) == 1
        start, end = segments[0]
        assert start == 10
        assert end == 40

    def test_boundary_at_threshold(self):
        """probability == threshold は speech 扱い (>= threshold)。"""
        probs = [0.1] * 5 + [0.5] * 30 + [0.1] * 15
        segments = detect_speech_segments(
            probs,
            threshold=0.5,
            min_speech_sec=0.5,
            min_silence_sec=0.3,
        )
        assert len(segments) == 1

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError, match="durations must be positive"):
            detect_speech_segments([0.1, 0.9], min_speech_sec=0.0)
        with pytest.raises(ValueError, match="durations must be positive"):
            detect_speech_segments([0.1, 0.9], max_segment_sec=-1.0)


# ----------------- compute_vad_probabilities + chunk_audio_by_vad -------


class FakeVAD:
    """SileroVAD-compatible mock、process() で固定 probability sequence を返す。"""

    def __init__(self, probs: list[float]):
        self._probs = list(probs)
        self._idx = 0
        self.reset_called = False

    def reset(self):
        self.reset_called = True
        self._idx = 0

    def process(self, audio: np.ndarray) -> float:
        if self._idx >= len(self._probs):
            return 0.0
        p = self._probs[self._idx]
        self._idx += 1
        return p


class TestComputeAndChunk:
    def test_compute_probabilities_with_mock_vad(self):
        # 5 frames worth of audio (= 5 × 512 = 2560 samples)
        audio = np.zeros(5 * FRAME_SIZE, dtype=np.float32)
        vad = FakeVAD([0.1, 0.9, 0.9, 0.9, 0.1])
        probs = compute_vad_probabilities(audio, vad=vad)
        assert vad.reset_called
        assert probs == [0.1, 0.9, 0.9, 0.9, 0.1]

    def test_compute_drops_trailing_partial_frame(self):
        """512 samples 未満の末尾は drop。"""
        # 5.5 frames worth = 5 × 512 + 256 samples
        audio = np.zeros(5 * FRAME_SIZE + 256, dtype=np.float32)
        vad = FakeVAD([0.5] * 10)  # 余裕
        probs = compute_vad_probabilities(audio, vad=vad)
        assert len(probs) == 5  # 末尾 256 samples は drop

    def test_chunk_audio_by_vad_returns_seconds(self):
        """chunk_audio_by_vad は (start_sec, end_sec) を返す。"""
        # 50 frames、middle 30 frames が speech
        audio = np.zeros(50 * FRAME_SIZE, dtype=np.float32)
        probs_seq = [0.1] * 10 + [0.9] * 30 + [0.1] * 10
        vad = FakeVAD(probs_seq)
        segments = chunk_audio_by_vad(
            audio,
            vad=vad,
            min_speech_sec=0.5,
            min_silence_sec=0.3,
        )
        assert len(segments) == 1
        start_sec, end_sec = segments[0]
        # frame 10 ~ 40 → 0.32 ~ 1.28 sec
        assert start_sec == pytest.approx(10 * FRAME_SEC, abs=0.001)
        assert end_sec == pytest.approx(40 * FRAME_SEC, abs=0.001)


# ----------------- Constants ----------------------------------------------


class TestConstants:
    def test_frame_size_matches_silero_default(self):
        assert FRAME_SIZE == 512
        assert SAMPLE_RATE == 16000
        assert FRAME_SEC == pytest.approx(0.032, abs=1e-6)

"""VADFileSegmenter の実 backend 統合テスト (Issue #366 Phase 1)。

実 Silero VAD + 実音声で adapter を検証する (core テストは MockVADBackend で
torch-free に契約を固定済み — こちらは実環境の裏取り)。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from livecap_cli.vad import VADFileSegmenter, VADProcessor

TEST_AUDIO_JA = (
    Path(__file__).parent.parent.parent / "assets/audio/ja/jsut_basic5000_0001.wav"
)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "r") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr


class TestRealBackend:
    def test_real_speech_detected(self):
        """実音声 (JSUT) → 1 件以上の発話区間"""
        audio, sr = _load_wav(TEST_AUDIO_JA)
        segmenter = VADFileSegmenter(VADProcessor())

        segments = segmenter(audio, sr)

        assert len(segments) >= 1
        duration = len(audio) / sr
        for start, end in segments:
            assert 0.0 <= start < end <= duration + 0.5

    def test_silence_returns_empty(self, tmp_path):
        """無音 → [] (file mode で ASR 呼び出しゼロになる契約の実環境確認)"""
        audio = np.zeros(16000 * 2, dtype=np.float32)
        segmenter = VADFileSegmenter(VADProcessor())

        assert segmenter(audio, 16000) == []

    def test_repeat_call_consistent(self):
        """同一 adapter の連続呼び出しで結果が安定 (reset lifecycle)"""
        audio, sr = _load_wav(TEST_AUDIO_JA)
        segmenter = VADFileSegmenter(VADProcessor())

        first = segmenter(audio, sr)
        second = segmenter(audio, sr)

        assert first == second

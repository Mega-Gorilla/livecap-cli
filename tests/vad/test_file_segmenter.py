"""`VADFileSegmenter` の単体テスト (Issue #366 Phase 1、torch-free)。

`MockVADBackend` (scripted probabilities) で実 VADProcessor + 実 state machine
を駆動し、adapter の契約 (final のみ / finalize 回収 / reset lifecycle /
例外後の回復 / 複数ファイル非持越) を固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from livecap_cli.vad import VADConfig, VADFileSegmenter, VADProcessor

from tests.vad.test_processor import MockVADBackend

# frame 512 samples @16kHz = 32ms。境界が予測しやすい小さめ設定
_CONFIG = VADConfig(
    threshold=0.5,
    min_speech_ms=64,    # 2 frames
    min_silence_ms=64,   # 2 frames
    speech_pad_ms=32,    # 1 frame
)

_FRAME = 512
_SR = 16000


def _audio(frames: int) -> np.ndarray:
    return np.zeros(_FRAME * frames, dtype=np.float32)


def _make(probabilities: list[float]) -> VADFileSegmenter:
    return VADFileSegmenter(
        VADProcessor(config=_CONFIG, backend=MockVADBackend(probabilities=probabilities))
    )


class TestSegmentation:
    def test_single_utterance_returns_final_segment_only(self):
        """発話 1 回 → final segment 1 件 (interim は除外)"""
        probs = [0.3] * 5 + [0.9] * 10 + [0.3] * 10
        segmenter = _make(probs)

        segments = segmenter(_audio(25), _SR)

        assert len(segments) == 1
        start, end = segments[0]
        assert 0.0 <= start < end
        # 発話開始 frame 5 (0.16s) - pad 1 frame ≈ 0.128s、終了 ≈ frame 15 (0.48s) 近傍
        assert start == pytest.approx(0.128, abs=0.1)
        assert end == pytest.approx(0.48, abs=0.15)

    def test_all_silence_returns_empty(self):
        """全無音 → [] (pipeline 側で ASR 呼び出しゼロになる契約)"""
        segmenter = _make([0.2] * 30)

        assert segmenter(_audio(30), _SR) == []

    def test_trailing_speech_collected_by_finalize(self):
        """末尾発話中に EOF → finalize() の segment が含まれる"""
        probs = [0.3] * 5 + [0.9] * 10  # 発話のまま音声終了
        segmenter = _make(probs)

        segments = segmenter(_audio(15), _SR)

        assert len(segments) == 1
        assert segments[0][1] > segments[0][0]

    def test_multiple_utterances(self):
        probs = [0.9] * 10 + [0.2] * 10 + [0.9] * 10 + [0.2] * 10
        segmenter = _make(probs)

        segments = segmenter(_audio(40), _SR)

        assert len(segments) == 2
        assert segments[0][1] <= segments[1][0] + 0.033  # pad 重なり許容 (1 frame)


class TestLifecycle:
    """reset lifecycle: 状態をファイル間・例外後に持ち越さない (#366 条件 4)"""

    def test_second_call_returns_identical_result(self):
        """同一 adapter の 2 回目呼び出しも同一結果 (reset 起点の絶対時刻)"""
        probs = [0.3] * 5 + [0.9] * 10 + [0.3] * 10
        segmenter = _make(probs)  # MockVADBackend.reset() は script を先頭に戻す

        first = segmenter(_audio(25), _SR)
        second = segmenter(_audio(25), _SR)

        assert first == second
        assert len(first) == 1

    def test_recovers_after_exception(self):
        """1 回目に backend が例外 → 2 回目は正常結果 (先頭 reset で回復)"""

        class FlakyBackend(MockVADBackend):
            def __init__(self, probabilities):
                super().__init__(probabilities=probabilities)
                self._raise_once = True

            def process(self, audio):
                if self._raise_once and self._index == 3:
                    self._raise_once = False
                    raise RuntimeError("backend broke mid-file")
                return super().process(audio)

        probs = [0.3] * 5 + [0.9] * 10 + [0.3] * 10
        segmenter = VADFileSegmenter(
            VADProcessor(config=_CONFIG, backend=FlakyBackend(probs))
        )

        with pytest.raises(RuntimeError):
            segmenter(_audio(25), _SR)

        segments = segmenter(_audio(25), _SR)
        assert len(segments) == 1

    def test_process_files_multiple_files_no_state_carryover(self, tmp_path):
        """pipeline の複数ファイル処理でも各ファイルが独立に segment される"""
        import wave

        from livecap_cli.transcription.file_pipeline import FileTranscriptionPipeline

        def _write_wav(path, seconds=0.8):
            with wave.open(str(path), "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_SR)
                w.writeframes(
                    np.zeros(int(_SR * seconds), dtype=np.int16).tobytes()
                )

        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_wav(f1)
        _write_wav(f2)

        # 0.8s = 25 frames。script は reset で先頭へ戻る → 両ファイルとも同じ判定
        probs = [0.3] * 5 + [0.9] * 10 + [0.3] * 10
        segmenter = _make(probs)
        pipeline = FileTranscriptionPipeline(segmenter=segmenter)
        try:
            results = pipeline.process_files(
                [f1, f2],
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert [r.success for r in results] == [True, True]
        counts = [r.metadata["detected_segment_count"] for r in results]
        assert counts[0] == counts[1] == 1  # 状態持越があると 2 file 目がずれる

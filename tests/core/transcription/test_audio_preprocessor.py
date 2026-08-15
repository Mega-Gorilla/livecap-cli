"""`AudioPreprocessor` 注入点の契約テスト (Issue #366 Phase 3)。

- ロード直後・segmenter 前に**厳密 1 回**適用
- segmenter と ASR slice が**同一の処理済み配列**を見る
- 戻り値契約 (ndarray / 1-D / shape・dtype 一致 = float32) の fail-fast
- preprocessor 例外は file-level failure (#362 経路)
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from livecap_cli.transcription.file_pipeline import (
    FileProcessingResult,
    FileTranscriptionCancelled,
    FileTranscriptionPipeline,
)

_SR = 16000
_SENTINEL = np.float32(0.123)


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(np.zeros(int(_SR * seconds), dtype=np.int16).tobytes())


def _write_tone_wav(path: Path, seconds: float = 1.0) -> None:
    """NoiseGate の envelope 状態が出力に現れる非自明な信号 (440Hz sine)。"""
    t = np.arange(int(_SR * seconds), dtype=np.float64)
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t / _SR)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes((tone * 32767).astype(np.int16).tobytes())


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.wav"
    _write_wav(path)
    return path


def _sentinel_preprocessor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.full_like(audio, _SENTINEL)


class TestApplication:
    def test_applied_exactly_once_per_file(self, tmp_path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_wav(f1)
        _write_wav(f2)
        calls = [0]

        def counting(audio, sample_rate):
            calls[0] += 1
            return audio

        pipeline = FileTranscriptionPipeline(audio_preprocessor=counting)
        try:
            results = pipeline.process_files(
                [f1, f2],
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert [r.success for r in results] == [True, True]
        assert calls[0] == 2  # 1 ファイルに厳密 1 回

    def test_segmenter_and_asr_see_same_processed_array(self, audio_path):
        """VAD 判定 (segmenter) と ASR slice の両方が処理済み配列を見る"""
        seen: dict = {}

        def recording_segmenter(
            audio: np.ndarray, sample_rate: int
        ) -> List[Tuple[float, float]]:
            seen["segmenter"] = audio.copy()
            return [(0.0, 1.0)]

        def recording_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            seen["transcriber"] = audio.copy()
            return "text"

        pipeline = FileTranscriptionPipeline(
            segmenter=recording_segmenter,
            audio_preprocessor=_sentinel_preprocessor,
        )
        try:
            result = pipeline.process_file(
                audio_path,
                segment_transcriber=recording_transcriber,
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert result.success is True
        assert np.all(seen["segmenter"] == _SENTINEL)
        assert np.all(seen["transcriber"] == _SENTINEL)

    def test_asr_sees_processed_even_without_segmenter(self, audio_path):
        """segmenter=None (全音声 1 segment) でも ASR は処理済み音声を見る"""
        seen: dict = {}

        def recording_transcriber(audio, sample_rate):
            seen["audio"] = audio.copy()
            return "text"

        pipeline = FileTranscriptionPipeline(
            audio_preprocessor=_sentinel_preprocessor
        )
        try:
            pipeline.process_file(
                audio_path,
                segment_transcriber=recording_transcriber,
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert np.all(seen["audio"] == _SENTINEL)

    def test_identity_preprocessor_equals_no_preprocessor(self, audio_path):
        """identity preprocessor は preprocessor なしと bit-identical"""
        received: dict = {}

        def make_transcriber(key):
            def transcriber(audio, sample_rate):
                received[key] = audio.copy()
                return "text"

            return transcriber

        for key, preprocessor in (
            ("none", None),
            ("identity", lambda a, s: a),
        ):
            pipeline = FileTranscriptionPipeline(audio_preprocessor=preprocessor)
            try:
                result = pipeline.process_file(
                    audio_path,
                    segment_transcriber=make_transcriber(key),
                    write_subtitles=False,
                )
            finally:
                pipeline.close()
            assert result.success is True

        assert np.array_equal(received["none"], received["identity"])


class TestReturnContract:
    """戻り値契約の fail-fast (Phase 2 の型厳格化と同方針)"""

    @pytest.mark.parametrize(
        ("bad_preprocessor", "exc", "match"),
        [
            (lambda a, s: list(a), TypeError, "must return np.ndarray"),
            (lambda a, s: a.reshape(1, -1), ValueError, "1-D shape"),
            (lambda a, s: a[:-100], ValueError, "1-D shape"),
            (lambda a, s: a.astype(np.int16), ValueError, "dtype"),
            (lambda a, s: a.astype(np.complex64), ValueError, "dtype"),
        ],
        ids=["list", "2d", "shorter", "int16", "complex64"],
    )
    def test_contract_violation_fails_fast(
        self, audio_path, bad_preprocessor, exc, match
    ):
        pipeline = FileTranscriptionPipeline(audio_preprocessor=bad_preprocessor)
        try:
            with pytest.raises(exc, match=match):
                pipeline.process_file(
                    audio_path,
                    segment_transcriber=lambda a, s: "text",
                    write_subtitles=False,
                )
        finally:
            pipeline.close()

    def test_preprocessor_exception_propagates_from_process_file(self, audio_path):
        def broken(audio, sample_rate):
            raise RuntimeError("gate exploded")

        pipeline = FileTranscriptionPipeline(audio_preprocessor=broken)
        try:
            with pytest.raises(RuntimeError, match="gate exploded"):
                pipeline.process_file(
                    audio_path,
                    segment_transcriber=lambda a, s: "text",
                    write_subtitles=False,
                )
        finally:
            pipeline.close()

    def test_process_files_converts_to_failed_result(self, audio_path):
        """file-level failure は process_files が failed result へ変換 (#362)"""
        errors: list[tuple[str, Exception | None]] = []

        pipeline = FileTranscriptionPipeline(
            audio_preprocessor=lambda a, s: list(a)  # TypeError 契約違反
        )
        try:
            results = pipeline.process_files(
                [audio_path],
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
                error_callback=lambda msg, exc: errors.append((msg, exc)),
            )
        finally:
            pipeline.close()

        assert len(results) == 1
        assert results[0].success is False
        assert "must return np.ndarray" in (results[0].error or "")
        assert len(errors) == 1
        assert isinstance(errors[0][1], TypeError)


class TestCancellationOrdering:
    """PR #372 レビュー: ロード後の cancel 確認は preprocessor より前"""

    def test_cancel_after_load_skips_preprocessor(self, audio_path):
        """ロード中に cancel された場合、任意 callable を実行しない

        (副作用のある公開 preprocessor が cancel 後に 1 回走るのを防ぐ)
        """
        calls = [0]

        def counting(audio, sample_rate):
            calls[0] += 1
            return audio

        cancel_calls = [0]

        def should_cancel() -> bool:
            # 1 回目 (抽出前) は False、2 回目 (ロード直後) で cancel
            cancel_calls[0] += 1
            return cancel_calls[0] >= 2

        pipeline = FileTranscriptionPipeline(audio_preprocessor=counting)
        try:
            with pytest.raises(FileTranscriptionCancelled):
                pipeline.process_file(
                    audio_path,
                    segment_transcriber=lambda a, s: "text",
                    write_subtitles=False,
                    should_cancel=should_cancel,
                )
        finally:
            pipeline.close()

        assert calls[0] == 0  # preprocessor は実行されない


class TestStateIsolationAcrossFiles:
    """PR #372 レビュー: 1 file 目の例外後、2 file 目が fresh state で成功"""

    def test_fresh_gate_after_previous_file_failed(self, tmp_path):
        from livecap_cli.audio.noise_gate import NoiseGate

        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_tone_wav(f1)
        _write_tone_wav(f2)

        calls = [0]
        captured: dict = {}

        def per_file_gate(audio, sample_rate):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("first file exploded")
            captured["input"] = audio.copy()
            out = NoiseGate(threshold_db=-35, sample_rate=sample_rate).process(audio)
            captured["output"] = out.copy()
            return out

        pipeline = FileTranscriptionPipeline(audio_preprocessor=per_file_gate)
        try:
            results = pipeline.process_files(
                [f1, f2],
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert results[0].success is False       # 1 file 目は file-level failure
        assert "first file exploded" in (results[0].error or "")
        assert results[1].success is True        # 2 file 目は成功

        # 2 file 目の出力が fresh gate の出力と bit-identical
        # (envelope / gate 状態が前 file から漏れていれば冒頭 ramp がずれる)
        reference = NoiseGate(threshold_db=-35, sample_rate=_SR).process(
            captured["input"]
        )
        assert np.array_equal(captured["output"], reference)

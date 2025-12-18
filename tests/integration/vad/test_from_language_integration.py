"""Integration tests for VADProcessor.from_language().

Tests the integration of language-optimized VAD with StreamTranscriber
using real VAD backends (TenVAD for Japanese, WebRTC for English) and
real audio files.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from livecap_cli import FileSource, StreamTranscriber, TranscriptionResult, VADProcessor


# Test audio file paths
TEST_AUDIO_JA = Path(__file__).parent.parent.parent / "assets/audio/ja/jsut_basic5000_0001.wav"
TEST_AUDIO_EN = Path(__file__).parent.parent.parent / "assets/audio/en/librispeech_1089-134686-0001.wav"


class MockEngine:
    """Mock transcription engine for testing.

    Returns fixed text without requiring actual ASR model.
    """

    def __init__(
        self,
        return_text: str = "test transcription",
        return_confidence: float = 0.95,
        sample_rate: int = 16000,
    ):
        self._return_text = return_text
        self._return_confidence = return_confidence
        self._sample_rate = sample_rate
        self.transcribe_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        self.transcribe_count += 1
        return (self._return_text, self._return_confidence)

    def get_required_sample_rate(self) -> int:
        return self._sample_rate


class TestFromLanguageWithStreamTranscriber:
    """Integration tests for from_language() with StreamTranscriber."""

    @pytest.fixture
    def ja_audio_file(self) -> Path:
        if not TEST_AUDIO_JA.exists():
            pytest.skip(f"Japanese test audio not found: {TEST_AUDIO_JA}")
        return TEST_AUDIO_JA

    @pytest.fixture
    def en_audio_file(self) -> Path:
        if not TEST_AUDIO_EN.exists():
            pytest.skip(f"English test audio not found: {TEST_AUDIO_EN}")
        return TEST_AUDIO_EN

    @pytest.fixture
    def mock_engine(self) -> MockEngine:
        return MockEngine(return_text="transcribed text")

    def test_ja_vad_with_stream_transcriber(
        self, ja_audio_file: Path, mock_engine: MockEngine
    ):
        """Japanese VAD (TenVAD) integrates correctly with StreamTranscriber."""
        # Suppress TenVAD license warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = VADProcessor.from_language("ja")

        # Verify TenVAD is selected
        assert "tenvad" in vad.backend_name

        # Create transcriber with language-optimized VAD
        with StreamTranscriber(
            engine=mock_engine,
            vad_processor=vad,
            source_id="test-ja",
        ) as transcriber:
            # Verify VAD is properly injected
            assert "tenvad" in transcriber._vad.backend_name

            # Process audio file
            with FileSource(str(ja_audio_file)) as source:
                results = list(transcriber.transcribe_sync(source))

        # Should produce transcription results
        assert len(results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in results)
        assert all(r.source_id == "test-ja" for r in results)
        assert mock_engine.transcribe_count > 0

    def test_en_vad_with_stream_transcriber(
        self, en_audio_file: Path, mock_engine: MockEngine
    ):
        """English VAD (WebRTC) integrates correctly with StreamTranscriber."""
        vad = VADProcessor.from_language("en")

        # Verify WebRTC is selected
        assert "webrtc" in vad.backend_name

        # Create transcriber with language-optimized VAD
        with StreamTranscriber(
            engine=mock_engine,
            vad_processor=vad,
            source_id="test-en",
        ) as transcriber:
            # Verify VAD is properly injected
            assert "webrtc" in transcriber._vad.backend_name

            # Process audio file
            with FileSource(str(en_audio_file)) as source:
                results = list(transcriber.transcribe_sync(source))

        # Should produce transcription results
        assert len(results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in results)
        assert all(r.source_id == "test-en" for r in results)
        assert mock_engine.transcribe_count > 0


class TestFromLanguageAudioProcessing:
    """Integration tests for from_language() audio processing."""

    @pytest.fixture
    def ja_audio_file(self) -> Path:
        if not TEST_AUDIO_JA.exists():
            pytest.skip(f"Japanese test audio not found: {TEST_AUDIO_JA}")
        return TEST_AUDIO_JA

    @pytest.fixture
    def en_audio_file(self) -> Path:
        if not TEST_AUDIO_EN.exists():
            pytest.skip(f"English test audio not found: {TEST_AUDIO_EN}")
        return TEST_AUDIO_EN

    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio file as numpy array."""
        import soundfile as sf

        audio, sr = sf.read(str(file_path))
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio

    def test_ja_vad_detects_speech(self, ja_audio_file: Path):
        """Japanese VAD (TenVAD) detects speech in Japanese audio."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = VADProcessor.from_language("ja")

        audio = self._load_audio(ja_audio_file)

        # Process audio
        segments = vad.process_chunk(audio, sample_rate=16000)

        # Finalize to get any remaining speech
        final_segment = vad.finalize()

        # Should detect speech (either during processing or finalize)
        total_segments = len(segments) + (1 if final_segment else 0)
        assert total_segments > 0, "TenVAD should detect speech in Japanese audio"

    def test_en_vad_detects_speech(self, en_audio_file: Path):
        """English VAD (WebRTC) detects speech in English audio."""
        vad = VADProcessor.from_language("en")

        audio = self._load_audio(en_audio_file)

        # Process audio
        segments = vad.process_chunk(audio, sample_rate=16000)

        # Finalize to get any remaining speech
        final_segment = vad.finalize()

        # Should detect speech (either during processing or finalize)
        total_segments = len(segments) + (1 if final_segment else 0)
        assert total_segments > 0, "WebRTC should detect speech in English audio"

    def test_ja_vad_produces_valid_segments(self, ja_audio_file: Path):
        """Japanese VAD produces valid VADSegments with audio data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = VADProcessor.from_language("ja")

        audio = self._load_audio(ja_audio_file)
        segments = vad.process_chunk(audio, sample_rate=16000)
        final_segment = vad.finalize()

        all_segments = segments + ([final_segment] if final_segment else [])

        for segment in all_segments:
            # Segment should have audio data
            assert segment.audio is not None
            assert len(segment.audio) > 0
            # Timing should be valid
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time

    def test_en_vad_produces_valid_segments(self, en_audio_file: Path):
        """English VAD produces valid VADSegments with audio data."""
        vad = VADProcessor.from_language("en")

        audio = self._load_audio(en_audio_file)
        segments = vad.process_chunk(audio, sample_rate=16000)
        final_segment = vad.finalize()

        all_segments = segments + ([final_segment] if final_segment else [])

        for segment in all_segments:
            # Segment should have audio data
            assert segment.audio is not None
            assert len(segment.audio) > 0
            # Timing should be valid
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time


class TestFromLanguageCallbackFlow:
    """Integration tests for from_language() with callback-based API."""

    @pytest.fixture
    def ja_audio_file(self) -> Path:
        if not TEST_AUDIO_JA.exists():
            pytest.skip(f"Japanese test audio not found: {TEST_AUDIO_JA}")
        return TEST_AUDIO_JA

    @pytest.fixture
    def mock_engine(self) -> MockEngine:
        return MockEngine(return_text="callback test")

    def test_callback_flow_with_language_vad(
        self, ja_audio_file: Path, mock_engine: MockEngine
    ):
        """Language-optimized VAD works with callback-based transcription."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            vad = VADProcessor.from_language("ja")

        transcriber = StreamTranscriber(
            engine=mock_engine,
            vad_processor=vad,
        )

        callback_results: list[TranscriptionResult] = []
        transcriber.set_callbacks(
            on_result=lambda r: callback_results.append(r),
        )

        with FileSource(str(ja_audio_file)) as source:
            for chunk in source:
                transcriber.feed_audio(chunk, source.sample_rate)

        # Finalize to get remaining results
        final = transcriber.finalize()
        if final:
            callback_results.append(final)

        transcriber.close()

        # Should receive results via callback
        assert len(callback_results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in callback_results)

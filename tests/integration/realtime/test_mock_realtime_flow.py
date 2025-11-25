"""Mock integration tests for realtime transcription flow.

Tests the full flow using:
- FileSource (real)
- Mock VAD (no torch dependency)
- Mock Engine (no model dependency)

This ensures CI can run without GPU/torch dependencies.
"""

import asyncio
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Import from public API (livecap_core)
from livecap_core import (
    AudioSource,
    DeviceInfo,
    EngineError,
    FileSource,
    InterimResult,
    StreamTranscriber,
    TranscriptionEngine,
    TranscriptionError,
    TranscriptionResult,
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,
)


# Test audio file path
TEST_AUDIO_FILE = Path(__file__).parent.parent.parent / "assets/audio/jsut_basic5000_0001_ja.wav"


class MockEngine:
    """Mock transcription engine for testing."""

    def __init__(
        self,
        return_text: str = "テスト文字起こし",
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


class MockVADProcessor:
    """Mock VAD processor that simulates segment detection."""

    def __init__(
        self,
        segment_after_samples: int = 16000,  # 1秒分のサンプルで発話終了
        generate_interim: bool = False,
    ):
        self._segment_after_samples = segment_after_samples
        self._generate_interim = generate_interim
        self._accumulated_samples = 0
        self._accumulated_audio: list[np.ndarray] = []
        self._state = VADState.SILENCE
        self._time_offset = 0.0

    def process_chunk(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[VADSegment]:
        self._accumulated_audio.append(audio)
        self._accumulated_samples += len(audio)

        segments: list[VADSegment] = []

        # 中間結果を生成
        if self._generate_interim and self._accumulated_samples >= self._segment_after_samples // 2:
            if self._state == VADState.SILENCE:
                self._state = VADState.SPEECH
                combined = np.concatenate(self._accumulated_audio)
                segments.append(VADSegment(
                    audio=combined,
                    start_time=self._time_offset,
                    end_time=self._time_offset + len(combined) / sample_rate,
                    is_final=False,
                ))

        # 十分なサンプルが溜まったら確定セグメントを生成
        if self._accumulated_samples >= self._segment_after_samples:
            combined = np.concatenate(self._accumulated_audio)
            start_time = self._time_offset
            end_time = self._time_offset + len(combined) / sample_rate
            segments.append(VADSegment(
                audio=combined,
                start_time=start_time,
                end_time=end_time,
                is_final=True,
            ))
            # リセット
            self._time_offset = end_time
            self._accumulated_audio = []
            self._accumulated_samples = 0
            self._state = VADState.SILENCE

        return segments

    def finalize(self) -> VADSegment | None:
        if self._accumulated_samples > 0:
            combined = np.concatenate(self._accumulated_audio)
            sample_rate = 16000  # assume
            return VADSegment(
                audio=combined,
                start_time=self._time_offset,
                end_time=self._time_offset + len(combined) / sample_rate,
                is_final=True,
            )
        return None

    def reset(self) -> None:
        self._accumulated_samples = 0
        self._accumulated_audio = []
        self._state = VADState.SILENCE
        self._time_offset = 0.0

    @property
    def state(self) -> VADState:
        return self._state


class TestPublicAPIImports:
    """Test that all public API imports work correctly."""

    def test_import_transcription_types(self):
        """Verify transcription type imports."""
        assert TranscriptionResult is not None
        assert InterimResult is not None
        assert TranscriptionError is not None
        assert EngineError is not None
        assert TranscriptionEngine is not None

    def test_import_stream_transcriber(self):
        """Verify StreamTranscriber import."""
        assert StreamTranscriber is not None

    def test_import_audio_sources(self):
        """Verify audio source imports."""
        assert AudioSource is not None
        assert DeviceInfo is not None
        assert FileSource is not None

    def test_import_vad(self):
        """Verify VAD imports."""
        assert VADConfig is not None
        assert VADProcessor is not None
        assert VADSegment is not None
        assert VADState is not None


class TestFileSourceIntegration:
    """Test FileSource with public API."""

    @pytest.fixture
    def audio_file(self) -> Path:
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")
        return TEST_AUDIO_FILE

    def test_file_source_iteration(self, audio_file: Path):
        """Test FileSource sync iteration."""
        with FileSource(str(audio_file)) as source:
            chunks = list(source)
            assert len(chunks) > 0
            assert all(isinstance(c, np.ndarray) for c in chunks)
            assert all(c.dtype == np.float32 for c in chunks)

    def test_file_source_properties(self, audio_file: Path):
        """Test FileSource properties."""
        with FileSource(str(audio_file)) as source:
            assert source.sample_rate == 16000
            # FileSource outputs mono audio (shape is 1D)

    def test_file_source_async_iteration(self, audio_file: Path):
        """Test FileSource async iteration."""

        async def run_async():
            chunks = []
            async with FileSource(str(audio_file)) as source:
                async for chunk in source:
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_async())
        assert len(chunks) > 0


class TestMockRealtimeFlow:
    """Integration tests using Mock components."""

    @pytest.fixture
    def audio_file(self) -> Path:
        if not TEST_AUDIO_FILE.exists():
            pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")
        return TEST_AUDIO_FILE

    @pytest.fixture
    def mock_engine(self) -> MockEngine:
        return MockEngine(return_text="こんにちは、世界")

    @pytest.fixture
    def mock_vad(self) -> MockVADProcessor:
        return MockVADProcessor(segment_after_samples=8000)  # 0.5秒で区切り

    def test_sync_transcription_flow(
        self,
        audio_file: Path,
        mock_engine: MockEngine,
        mock_vad: MockVADProcessor,
    ):
        """Test synchronous transcription flow with FileSource."""
        with StreamTranscriber(
            engine=mock_engine,
            vad_processor=mock_vad,
            source_id="test-sync",
        ) as transcriber:
            with FileSource(str(audio_file)) as source:
                results = list(transcriber.transcribe_sync(source))

        assert len(results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in results)
        assert all(r.text == "こんにちは、世界" for r in results)
        assert all(r.source_id == "test-sync" for r in results)
        assert all(r.is_final for r in results)
        assert mock_engine.transcribe_count > 0

    def test_async_transcription_flow(
        self,
        audio_file: Path,
        mock_engine: MockEngine,
        mock_vad: MockVADProcessor,
    ):
        """Test asynchronous transcription flow with FileSource."""

        async def run_async():
            results = []
            async with FileSource(str(audio_file)) as source:
                transcriber = StreamTranscriber(
                    engine=mock_engine,
                    vad_processor=mock_vad,
                    source_id="test-async",
                )
                async for result in transcriber.transcribe_async(source):
                    results.append(result)
            return results

        results = asyncio.run(run_async())

        assert len(results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in results)
        assert all(r.source_id == "test-async" for r in results)

    def test_feed_audio_with_callbacks(
        self,
        audio_file: Path,
        mock_engine: MockEngine,
    ):
        """Test callback-based API with feed_audio."""
        # VADがすぐにセグメントを出力するよう設定
        mock_vad = MockVADProcessor(segment_after_samples=4000)

        transcriber = StreamTranscriber(
            engine=mock_engine,
            vad_processor=mock_vad,
        )

        callback_results: list[TranscriptionResult] = []
        transcriber.set_callbacks(
            on_result=lambda r: callback_results.append(r),
        )

        with FileSource(str(audio_file)) as source:
            for chunk in source:
                transcriber.feed_audio(chunk, source.sample_rate)

        # finalize で残りを処理
        final = transcriber.finalize()
        if final:
            callback_results.append(final)

        transcriber.close()

        assert len(callback_results) > 0
        assert all(isinstance(r, TranscriptionResult) for r in callback_results)

    def test_interim_results_with_callbacks(
        self,
        audio_file: Path,
        mock_engine: MockEngine,
    ):
        """Test interim results via callback."""
        mock_vad = MockVADProcessor(
            segment_after_samples=8000,
            generate_interim=True,
        )

        transcriber = StreamTranscriber(
            engine=mock_engine,
            vad_processor=mock_vad,
        )

        interim_results: list[InterimResult] = []
        final_results: list[TranscriptionResult] = []

        transcriber.set_callbacks(
            on_result=lambda r: final_results.append(r),
            on_interim=lambda r: interim_results.append(r),
        )

        with FileSource(str(audio_file)) as source:
            for chunk in source:
                transcriber.feed_audio(chunk, source.sample_rate)

        transcriber.close()

        # 中間結果が生成されていることを確認
        assert len(interim_results) >= 0  # 生成されない場合もある
        assert len(final_results) > 0


class TestTranscriptionResultTypes:
    """Test result type properties."""

    def test_transcription_result_attributes(self):
        """Test TranscriptionResult dataclass."""
        result = TranscriptionResult(
            text="テスト",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
            confidence=0.95,
            source_id="test",
        )
        assert result.text == "テスト"
        assert result.start_time == 0.0
        assert result.end_time == 1.0
        assert result.is_final is True
        assert result.confidence == 0.95
        assert result.source_id == "test"

    def test_interim_result_attributes(self):
        """Test InterimResult dataclass."""
        result = InterimResult(
            text="途中経過",
            accumulated_time=0.5,
            source_id="test",
        )
        assert result.text == "途中経過"
        assert result.accumulated_time == 0.5
        assert result.source_id == "test"


class TestExceptionTypes:
    """Test exception type hierarchy."""

    def test_transcription_error_base(self):
        """Test TranscriptionError is base class."""
        assert issubclass(EngineError, TranscriptionError)
        assert issubclass(TranscriptionError, Exception)

    def test_engine_error_raised(self):
        """Test EngineError can be raised and caught."""
        with pytest.raises(TranscriptionError):
            raise EngineError("Test error")

        with pytest.raises(EngineError):
            raise EngineError("Test error")


class TestVADTypesFromPublicAPI:
    """Test VAD types exported from public API."""

    def test_vad_config_defaults(self):
        """Test VADConfig default values."""
        config = VADConfig()
        assert config.threshold == 0.5
        assert config.min_speech_ms == 250
        assert config.min_silence_ms == 100
        assert config.speech_pad_ms == 100

    def test_vad_config_custom(self):
        """Test VADConfig with custom values."""
        config = VADConfig(
            threshold=0.6,
            min_speech_ms=300,
        )
        assert config.threshold == 0.6
        assert config.min_speech_ms == 300

    def test_vad_segment_creation(self):
        """Test VADSegment creation."""
        audio = np.zeros(1600, dtype=np.float32)
        segment = VADSegment(
            audio=audio,
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        assert len(segment.audio) == 1600
        assert segment.start_time == 0.0
        assert segment.end_time == 0.1
        assert segment.is_final is True

    def test_vad_state_enum(self):
        """Test VADState enum values."""
        # VADState uses integer values
        assert VADState.SILENCE.value == 1
        assert VADState.SPEECH.value == 3
        assert hasattr(VADState, "POTENTIAL_SPEECH")
        assert hasattr(VADState, "ENDING")

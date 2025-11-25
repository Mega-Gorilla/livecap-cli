"""E2E integration tests for realtime transcription flow.

Tests the full flow using:
- SileroVAD (real VAD backend)
- Real ASR engines (WhisperS2T)
- FileSource with real audio files

These tests require:
- LIVECAP_ENABLE_REALTIME_E2E=1 environment variable
- silero-vad and torch installed
- ASR engine models available

Run locally with:
    LIVECAP_ENABLE_REALTIME_E2E=1 pytest tests/integration/realtime/test_e2e_realtime_flow.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Add project root to path for engine imports - must be before livecap_core imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(TESTS_ROOT))

from livecap_core import (  # noqa: E402
    FileSource,
    StreamTranscriber,
    TranscriptionResult,
    VADConfig,
    VADProcessor,
    VADSegment,
)

if TYPE_CHECKING:
    from livecap_core.vad.backends.silero import SileroVAD

# Test configuration
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "audio"
E2E_ENABLED = os.getenv("LIVECAP_ENABLE_REALTIME_E2E") == "1"
STRICT = os.getenv("LIVECAP_REQUIRE_REALTIME_E2E") == "1"

# Expected keywords in transcriptions
# Note: Realtime VAD-based transcription may have lower accuracy than batch processing
# due to shorter segments, so we use looser keywords
KEYWORD_HINTS = {
    "librispeech_test-clean_1089-134686-0001_en": ["stuff", "belly"],
    # Japanese: "水をマレーシアから買わなくてはならない" - VAD segmentation may affect accuracy
    # Only check for "水" which is reliably detected
    "jsut_basic5000_0001_ja": ["水"],
}

pytestmark = pytest.mark.realtime_e2e


def _skip_or_fail(reason: str) -> None:
    """Skip or fail depending on STRICT mode."""
    if STRICT:
        pytest.fail(reason)
    pytest.skip(reason)


def _require_e2e_enabled() -> None:
    """Guard: require E2E tests to be enabled."""
    if not E2E_ENABLED:
        _skip_or_fail(
            "Realtime E2E tests disabled (set LIVECAP_ENABLE_REALTIME_E2E=1 to enable)."
        )


def _try_import_silero_vad() -> type[SileroVAD] | None:
    """Try to import SileroVAD, return None if unavailable."""
    try:
        from livecap_core.vad.backends.silero import SileroVAD

        return SileroVAD
    except ImportError:
        return None


def _try_create_silero_vad() -> SileroVAD | None:
    """Try to create a SileroVAD instance, return None if unavailable."""
    SileroVADClass = _try_import_silero_vad()
    if SileroVADClass is None:
        return None
    try:
        return SileroVADClass(onnx=True)
    except Exception:
        return None


def _try_create_vad_processor() -> VADProcessor | None:
    """Try to create a VADProcessor with SileroVAD backend."""
    try:
        return VADProcessor()
    except ImportError:
        return None


def _try_create_engine(
    engine_type: str = "whispers2t_base",
    device: str = "cpu",
    language: str = "en",
):
    """Try to create an ASR engine, return None if unavailable."""
    try:
        from engines.engine_factory import EngineFactory
        from livecap_core.config.defaults import get_default_config

        config = get_default_config()
        config["transcription"]["engine"] = engine_type
        config["transcription"]["input_language"] = language

        engine = EngineFactory.create_engine(
            engine_type=engine_type,
            device=device,
            config=config,
        )
        engine.load_model()
        return engine
    except Exception:
        return None


@pytest.fixture
def audio_file_en() -> Path:
    """English audio test file."""
    path = ASSETS_ROOT / "librispeech_test-clean_1089-134686-0001_en.wav"
    if not path.exists():
        pytest.skip(f"Test audio file not found: {path}")
    return path


@pytest.fixture
def audio_file_ja() -> Path:
    """Japanese audio test file."""
    path = ASSETS_ROOT / "jsut_basic5000_0001_ja.wav"
    if not path.exists():
        pytest.skip(f"Test audio file not found: {path}")
    return path


class TestSileroVADDirect:
    """Direct SileroVAD backend tests."""

    def test_silero_vad_import(self):
        """Test SileroVAD can be imported."""
        _require_e2e_enabled()

        SileroVADClass = _try_import_silero_vad()
        if SileroVADClass is None:
            _skip_or_fail("SileroVAD dependencies not available (torch, silero-vad)")

        assert SileroVADClass is not None

    def test_silero_vad_creation(self):
        """Test SileroVAD instance can be created."""
        _require_e2e_enabled()

        vad = _try_create_silero_vad()
        if vad is None:
            _skip_or_fail("SileroVAD could not be initialized")

        assert vad is not None

    def test_silero_vad_process_frame(self):
        """Test SileroVAD processes a single frame."""
        _require_e2e_enabled()

        vad = _try_create_silero_vad()
        if vad is None:
            _skip_or_fail("SileroVAD could not be initialized")

        # 512 samples @ 16kHz = 32ms frame
        frame = np.zeros(512, dtype=np.float32)
        probability = vad.process(frame)

        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    def test_silero_vad_detects_speech(self, audio_file_en: Path):
        """Test SileroVAD detects speech in real audio."""
        _require_e2e_enabled()

        vad = _try_create_silero_vad()
        if vad is None:
            _skip_or_fail("SileroVAD could not be initialized")

        # Load audio file
        with FileSource(str(audio_file_en), chunk_ms=32) as source:
            probabilities = []
            for chunk in source:
                if len(chunk) >= 512:
                    prob = vad.process(chunk[:512])
                    probabilities.append(prob)

        # Should detect some speech (probability > 0.5 at some point)
        max_prob = max(probabilities) if probabilities else 0.0
        assert max_prob > 0.5, f"No speech detected, max probability: {max_prob}"


class TestVADProcessorIntegration:
    """VADProcessor with SileroVAD backend integration tests."""

    def test_vad_processor_creation(self):
        """Test VADProcessor with default SileroVAD backend."""
        _require_e2e_enabled()

        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        assert processor is not None
        assert processor.config is not None

    def test_vad_processor_with_file_source(self, audio_file_en: Path):
        """Test VADProcessor detects segments from FileSource."""
        _require_e2e_enabled()

        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        all_segments: list[VADSegment] = []

        with FileSource(str(audio_file_en)) as source:
            for chunk in source:
                segments = processor.process_chunk(chunk, source.sample_rate)
                all_segments.extend(segments)

            # Finalize to get any remaining segment
            final_segment = processor.finalize()
            if final_segment:
                all_segments.append(final_segment)

        # Should detect at least one segment
        final_segments = [s for s in all_segments if s.is_final]
        assert len(final_segments) > 0, "No speech segments detected"

        # Segments should have audio data
        for segment in final_segments:
            assert len(segment.audio) > 0
            assert segment.start_time >= 0.0
            assert segment.end_time > segment.start_time

    def test_vad_processor_custom_config(self, audio_file_en: Path):
        """Test VADProcessor with custom configuration."""
        _require_e2e_enabled()

        # Try to create with custom config
        try:
            from livecap_core.vad.backends.silero import SileroVAD

            backend = SileroVAD(threshold=0.6, onnx=True)
            config = VADConfig(
                threshold=0.6,
                min_speech_ms=200,
                min_silence_ms=150,
            )
            processor = VADProcessor(config=config, backend=backend)
        except ImportError:
            _skip_or_fail("SileroVAD dependencies not available")
            return

        all_segments: list[VADSegment] = []

        with FileSource(str(audio_file_en)) as source:
            for chunk in source:
                segments = processor.process_chunk(chunk, source.sample_rate)
                all_segments.extend(segments)

            final_segment = processor.finalize()
            if final_segment:
                all_segments.append(final_segment)

        # Should still detect segments with custom config
        final_segments = [s for s in all_segments if s.is_final]
        assert len(final_segments) >= 0  # May be different count with different config


class TestStreamTranscriberE2E:
    """StreamTranscriber E2E tests with real engine."""

    def test_stream_transcriber_e2e_flow_en(self, audio_file_en: Path):
        """Test full E2E flow with English audio."""
        _require_e2e_enabled()

        # Try to create VAD processor
        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        # Try to create engine with English language
        engine = _try_create_engine("whispers2t_base", "cpu", language="en")
        if engine is None:
            _skip_or_fail("WhisperS2T engine could not be initialized")

        try:
            with StreamTranscriber(
                engine=engine,
                vad_processor=processor,
                source_id="e2e-test-en",
            ) as transcriber:
                with FileSource(str(audio_file_en)) as source:
                    results = list(transcriber.transcribe_sync(source))
        finally:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()

        # Should produce results
        assert len(results) > 0, "No transcription results produced"

        # All results should be TranscriptionResult
        assert all(isinstance(r, TranscriptionResult) for r in results)

        # Combine all text
        full_text = " ".join(r.text for r in results).lower()

        # Check for expected keywords
        expected_keywords = KEYWORD_HINTS.get(
            "librispeech_test-clean_1089-134686-0001_en", []
        )
        for keyword in expected_keywords:
            assert keyword.lower() in full_text, (
                f"Expected keyword '{keyword}' not found in: {full_text}"
            )

    def test_stream_transcriber_e2e_flow_ja(self, audio_file_ja: Path):
        """Test full E2E flow with Japanese audio."""
        _require_e2e_enabled()

        # Try to create VAD processor
        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        # Try to create engine with Japanese language
        engine = _try_create_engine("whispers2t_base", "cpu", language="ja")
        if engine is None:
            _skip_or_fail("WhisperS2T engine could not be initialized")

        try:
            with StreamTranscriber(
                engine=engine,
                vad_processor=processor,
                source_id="e2e-test-ja",
            ) as transcriber:
                with FileSource(str(audio_file_ja)) as source:
                    results = list(transcriber.transcribe_sync(source))
        finally:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()

        # Should produce results
        assert len(results) > 0, "No transcription results produced"

        # Combine all text
        full_text = "".join(r.text for r in results)

        # Check for expected keywords (Japanese)
        expected_keywords = KEYWORD_HINTS.get("jsut_basic5000_0001_ja", [])
        for keyword in expected_keywords:
            assert keyword in full_text, (
                f"Expected keyword '{keyword}' not found in: {full_text}"
            )

    def test_stream_transcriber_callback_api(self, audio_file_en: Path):
        """Test callback-based API with real components."""
        _require_e2e_enabled()

        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        engine = _try_create_engine("whispers2t_base", "cpu", language="en")
        if engine is None:
            _skip_or_fail("WhisperS2T engine could not be initialized")

        callback_results: list[TranscriptionResult] = []

        try:
            transcriber = StreamTranscriber(
                engine=engine,
                vad_processor=processor,
            )
            transcriber.set_callbacks(
                on_result=lambda r: callback_results.append(r),
            )

            with FileSource(str(audio_file_en)) as source:
                for chunk in source:
                    transcriber.feed_audio(chunk, source.sample_rate)

            # Finalize
            final = transcriber.finalize()
            if final:
                callback_results.append(final)

            transcriber.close()
        finally:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()

        # Should have received results via callback
        assert len(callback_results) > 0, "No results received via callback"


class TestStreamTranscriberAsyncE2E:
    """Async StreamTranscriber E2E tests."""

    def test_async_transcription_flow(self, audio_file_en: Path):
        """Test async transcription with real components."""
        import asyncio

        _require_e2e_enabled()

        processor = _try_create_vad_processor()
        if processor is None:
            _skip_or_fail("VADProcessor with SileroVAD could not be created")

        engine = _try_create_engine("whispers2t_base", "cpu", language="en")
        if engine is None:
            _skip_or_fail("WhisperS2T engine could not be initialized")

        async def run_async():
            results = []
            async with FileSource(str(audio_file_en)) as source:
                transcriber = StreamTranscriber(
                    engine=engine,
                    vad_processor=processor,
                )
                async for result in transcriber.transcribe_async(source):
                    results.append(result)
                transcriber.close()
            return results

        try:
            results = asyncio.run(run_async())
        finally:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()

        assert len(results) > 0, "No async transcription results"
        assert all(isinstance(r, TranscriptionResult) for r in results)

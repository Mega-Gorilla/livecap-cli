"""Unit tests for StreamTranscriber."""

import asyncio
from typing import Tuple

import numpy as np

from livecap_core.transcription import (
    EngineError,
    StreamTranscriber,
    TranscriptionError,
)
from livecap_core.vad import VADSegment, VADState


class MockEngine:
    """テスト用モックエンジン"""

    def __init__(
        self,
        return_text: str = "テスト",
        return_confidence: float = 0.9,
        sample_rate: int = 16000,
        should_fail: bool = False,
    ):
        self._return_text = return_text
        self._return_confidence = return_confidence
        self._sample_rate = sample_rate
        self._should_fail = should_fail
        self.call_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        self.call_count += 1
        if self._should_fail:
            raise RuntimeError("Mock engine failure")
        return (self._return_text, self._return_confidence)

    def get_required_sample_rate(self) -> int:
        return self._sample_rate


class MockVADProcessor:
    """テスト用モックVADプロセッサ"""

    def __init__(self, segments: list[VADSegment] | None = None):
        self._segments = segments or []
        self._segment_index = 0
        self._state = VADState.SILENCE
        self._finalize_segment: VADSegment | None = None

    def process_chunk(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[VADSegment]:
        if self._segment_index < len(self._segments):
            segment = self._segments[self._segment_index]
            self._segment_index += 1
            return [segment]
        return []

    def finalize(self) -> VADSegment | None:
        return self._finalize_segment

    def reset(self) -> None:
        self._segment_index = 0
        self._state = VADState.SILENCE

    @property
    def state(self) -> VADState:
        return self._state


class MockAudioSource:
    """テスト用モック音声ソース"""

    def __init__(self, chunks: list[np.ndarray] | None = None, sample_rate: int = 16000):
        self._chunks = chunks or []
        self.sample_rate = sample_rate

    def __iter__(self):
        for chunk in self._chunks:
            yield chunk

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class TestStreamTranscriberBasics:
    """StreamTranscriber 基本機能テスト"""

    def test_create_with_engine(self):
        """エンジンで作成"""
        engine = MockEngine()
        vad = MockVADProcessor()  # モック VAD を注入（silero-vad 不要）
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)
        assert transcriber.sample_rate == 16000
        assert transcriber.source_id == "default"

    def test_create_with_custom_source_id(self):
        """カスタムソースIDで作成"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, source_id="mic1", vad_processor=vad)
        assert transcriber.source_id == "mic1"

    def test_create_with_vad_processor(self):
        """VADプロセッサ注入で作成"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)
        assert transcriber._vad is vad


class TestStreamTranscriberExceptions:
    """例外型テスト"""

    def test_transcription_error_hierarchy(self):
        """例外の継承関係"""
        assert issubclass(EngineError, TranscriptionError)
        assert issubclass(TranscriptionError, Exception)

    def test_engine_error_raised_on_failure(self):
        """エンジン失敗時のEngineError"""
        engine = MockEngine(should_fail=True)
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        # feed_audioではエラーがキャッチされる
        transcriber.feed_audio(np.zeros(512, dtype=np.float32))
        # 結果キューは空
        assert transcriber.get_result(timeout=0) is None


class TestStreamTranscriberFeedAudio:
    """feed_audio テスト"""

    def test_feed_audio_with_final_segment(self):
        """確定セグメントの処理"""
        engine = MockEngine(return_text="こんにちは")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        result = transcriber.get_result(timeout=0.1)
        assert result is not None
        assert result.text == "こんにちは"
        assert result.is_final is True
        assert result.start_time == 0.0
        assert result.end_time == 0.1

    def test_feed_audio_with_interim_segment(self):
        """中間セグメントの処理"""
        engine = MockEngine(return_text="途中")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.5,
            is_final=False,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        interim = transcriber.get_interim()
        assert interim is not None
        assert interim.text == "途中"
        assert interim.accumulated_time == 0.5

    def test_feed_audio_empty_segment(self):
        """空セグメントの処理"""
        engine = MockEngine()
        segment = VADSegment(
            audio=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.0,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        assert transcriber.get_result(timeout=0) is None


class TestStreamTranscriberCallbacks:
    """コールバックテスト"""

    def test_on_result_callback(self):
        """確定結果コールバック"""
        engine = MockEngine(return_text="結果")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        callback_results = []
        transcriber.set_callbacks(
            on_result=lambda r: callback_results.append(r)
        )

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        assert len(callback_results) == 1
        assert callback_results[0].text == "結果"

    def test_on_interim_callback(self):
        """中間結果コールバック"""
        engine = MockEngine(return_text="途中経過")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.5,
            is_final=False,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        callback_results = []
        transcriber.set_callbacks(
            on_interim=lambda r: callback_results.append(r)
        )

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        assert len(callback_results) == 1
        assert callback_results[0].text == "途中経過"


class TestStreamTranscriberFinalize:
    """finalize テスト"""

    def test_finalize_with_remaining_segment(self):
        """残りセグメントのfinalize"""
        engine = MockEngine(return_text="最終")
        vad = MockVADProcessor()
        vad._finalize_segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            is_final=True,
        )
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        result = transcriber.finalize()

        assert result is not None
        assert result.text == "最終"
        assert result.is_final is True

    def test_finalize_without_segment(self):
        """セグメントなしでfinalize"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        result = transcriber.finalize()

        assert result is None


class TestStreamTranscriberReset:
    """reset テスト"""

    def test_reset_clears_queue(self):
        """resetでキューがクリアされる"""
        engine = MockEngine(return_text="テスト")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        transcriber.feed_audio(np.zeros(512, dtype=np.float32))
        assert transcriber.get_result(timeout=0) is not None  # 結果がある

        # 再度feed_audioする前にreset
        vad._segment_index = 0  # リセット
        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        transcriber.reset()

        assert transcriber.get_result(timeout=0) is None  # キューがクリアされた


class TestStreamTranscriberSyncAPI:
    """同期API テスト"""

    def test_transcribe_sync(self):
        """transcribe_sync基本動作"""
        engine = MockEngine(return_text="同期テスト")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        audio_source = MockAudioSource(
            chunks=[np.zeros(512, dtype=np.float32)]
        )

        results = list(transcriber.transcribe_sync(audio_source))

        assert len(results) >= 1
        assert results[0].text == "同期テスト"


class TestStreamTranscriberAsyncAPI:
    """非同期API テスト"""

    def test_transcribe_async(self):
        """transcribe_async基本動作"""

        async def run_test():
            engine = MockEngine(return_text="非同期テスト")
            segment = VADSegment(
                audio=np.zeros(1600, dtype=np.float32),
                start_time=0.0,
                end_time=0.1,
                is_final=True,
            )
            vad = MockVADProcessor(segments=[segment])
            transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

            audio_source = MockAudioSource(
                chunks=[np.zeros(512, dtype=np.float32)]
            )

            results = []
            async for result in transcriber.transcribe_async(audio_source):
                results.append(result)

            return results

        results = asyncio.run(run_test())
        assert len(results) >= 1
        assert results[0].text == "非同期テスト"


class TestStreamTranscriberContextManager:
    """コンテキストマネージャテスト"""

    def test_context_manager(self):
        """with文での使用"""
        engine = MockEngine()
        vad = MockVADProcessor()
        with StreamTranscriber(engine=engine, vad_processor=vad) as transcriber:
            assert transcriber is not None

    def test_close(self):
        """close呼び出し"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)
        transcriber.close()  # エラーなく実行できる


class TestStreamTranscriberProperties:
    """プロパティテスト"""

    def test_vad_state(self):
        """vad_stateプロパティ"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        assert transcriber.vad_state == VADState.SILENCE

    def test_sample_rate(self):
        """sample_rateプロパティ"""
        engine = MockEngine(sample_rate=48000)
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        assert transcriber.sample_rate == 48000

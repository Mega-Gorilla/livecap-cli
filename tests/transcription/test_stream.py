"""Unit tests for StreamTranscriber."""

import asyncio
from typing import Tuple

import numpy as np

from livecap_cli.transcription import (
    EngineError,
    StreamTranscriber,
    TranscriptionError,
)
from livecap_cli.vad import VADSegment, VADState


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
        self._current_time: float = 0.0

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

    @property
    def current_time(self) -> float:
        return self._current_time


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
            audio=np.full(1600, 0.1, dtype=np.float32),
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
            audio=np.full(1600, 0.1, dtype=np.float32),
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
            audio=np.full(1600, 0.1, dtype=np.float32),
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
        engine = MockEngine(return_text="確定結果テスト")
        segment = VADSegment(
            audio=np.full(1600, 0.1, dtype=np.float32),
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
        assert callback_results[0].text == "確定結果テスト"

    def test_on_interim_callback(self):
        """中間結果コールバック"""
        engine = MockEngine(return_text="途中経過")
        segment = VADSegment(
            audio=np.full(1600, 0.1, dtype=np.float32),
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
            audio=np.full(1600, 0.1, dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            is_final=True,
        )
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        results = transcriber.finalize()

        assert len(results) == 1
        assert results[0].text == "最終"
        assert results[0].is_final is True

    def test_finalize_without_segment(self):
        """セグメントなしでfinalize"""
        engine = MockEngine()
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(engine=engine, vad_processor=vad)

        results = transcriber.finalize()

        assert results == []


class TestStreamTranscriberReset:
    """reset テスト"""

    def test_reset_clears_queue(self):
        """resetでキューがクリアされる"""
        engine = MockEngine(return_text="テスト結果確認")
        segment = VADSegment(
            audio=np.full(1600, 0.1, dtype=np.float32),
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
            audio=np.full(1600, 0.1, dtype=np.float32),
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
                audio=np.full(1600, 0.1, dtype=np.float32),
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


class TestEnergyGate:
    """#292 EnergyGate: per-segment energy ガードのテスト。

    3 callsites (final_sync / final_async / interim) で engine.transcribe()
    が low-energy segment で **呼ばれない** ことを mock の call_count で検証する。
    """

    @staticmethod
    def _quiet_audio(n: int = 1600, amp: float = 0.0001) -> np.ndarray:
        """very-quiet audio: amp=0.0001 → -80 dBFS。"""
        return np.full(n, amp, dtype=np.float32)

    @staticmethod
    def _loud_audio(n: int = 1600, amp: float = 0.1) -> np.ndarray:
        """typical-speech-level audio: amp=0.1 → -20 dBFS。"""
        return np.full(n, amp, dtype=np.float32)

    # === sync path ===

    def test_low_energy_skips_engine_sync(self):
        """低 RMS segment では engine.transcribe() が呼ばれない (sync)。"""
        engine = MockEngine()
        segment = VADSegment(
            audio=self._quiet_audio(),
            start_time=0.0, end_time=0.1, is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=-45.0,
        )
        t.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine.call_count == 0
        assert t._dropped_low_energy_final_sync == 1
        assert t._dropped_low_energy_final_async == 0
        assert t._dropped_low_energy_interim == 0

    def test_high_energy_passes_through_sync(self):
        """十分なエネルギーの segment は engine.transcribe() に渡る (sync)。"""
        engine = MockEngine()
        segment = VADSegment(
            audio=self._loud_audio(),
            start_time=0.0, end_time=0.1, is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=-45.0,
        )
        t.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine.call_count == 1
        assert t._dropped_low_energy_final_sync == 0

    def test_opt_out_with_neg_inf(self):
        """engine_min_rms_dbfs=-inf で完全 opt-out: どんな低 RMS でも engine 呼出。"""
        engine = MockEngine()
        segment = VADSegment(
            audio=self._quiet_audio(),  # -80 dBFS
            start_time=0.0, end_time=0.1, is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=float("-inf"),
        )
        t.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine.call_count == 1
        assert t._dropped_low_energy_final_sync == 0

    # === interim path ===

    def test_interim_path_skips_engine(self):
        """非確定 segment (interim) でも low-energy なら engine 不呼び。"""
        engine = MockEngine()
        segment = VADSegment(
            audio=self._quiet_audio(),
            start_time=0.0, end_time=0.5, is_final=False,  # interim
        )
        vad = MockVADProcessor(segments=[segment])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=-45.0,
        )
        t.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine.call_count == 0
        assert t._dropped_low_energy_interim == 1
        assert t._dropped_low_energy_final_sync == 0

    # === async path ===

    def test_async_path_skips_engine(self):
        """非同期パスでも low-energy で engine 不呼び。"""
        engine = MockEngine()
        segment = VADSegment(
            audio=self._quiet_audio(),
            start_time=0.0, end_time=0.1, is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=-45.0,
        )

        async def run():
            result = await t._transcribe_segment_async(segment)
            assert result is None

        asyncio.run(run())
        assert engine.call_count == 0
        assert t._dropped_low_energy_final_async == 1

    # === metric choice ===

    def test_metric_max_frame_resists_padding_dilution(self):
        """max_frame_rms (default): 短文 + padding が pass する境界を確認。

        50ms @ amp=0.1 (-20 dBFS) + 950ms silence の segment。
        whole_rms だと希釈で ~-33 dBFS → -25 dB threshold で drop。
        max_frame_rms だと speech 部分が -20 dBFS → -25 dB で pass。
        """
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        audio[: int(0.05 * sr)] = 0.1
        seg = VADSegment(audio=audio, start_time=0.0, end_time=1.0, is_final=True)
        vad = MockVADProcessor(segments=[seg])

        # max_frame_rms: pass
        engine_max = MockEngine()
        t_max = StreamTranscriber(
            engine=engine_max, vad_processor=vad,
            engine_min_rms_dbfs=-25.0,
            engine_energy_metric="max_frame_rms",
        )
        t_max.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine_max.call_count == 1, "max_frame_rms should pass padded short speech"

        # whole_rms: drop
        engine_whole = MockEngine()
        vad2 = MockVADProcessor(segments=[seg])
        t_whole = StreamTranscriber(
            engine=engine_whole, vad_processor=vad2,
            engine_min_rms_dbfs=-25.0,
            engine_energy_metric="whole_rms",
        )
        t_whole.feed_audio(np.zeros(512, dtype=np.float32))
        assert engine_whole.call_count == 0, "whole_rms should drop padded short speech (dilution)"

    # === validation ===

    def test_invalid_metric_raises(self):
        engine = MockEngine()
        vad = MockVADProcessor()
        import pytest
        with pytest.raises(ValueError, match="engine_energy_metric"):
            StreamTranscriber(
                engine=engine, vad_processor=vad,
                engine_energy_metric="bogus_metric",
            )

    def test_invalid_frame_ms_raises(self):
        engine = MockEngine()
        vad = MockVADProcessor()
        import pytest
        with pytest.raises(ValueError, match="engine_energy_frame_ms"):
            StreamTranscriber(
                engine=engine, vad_processor=vad,
                engine_energy_frame_ms=0.0,
            )
        with pytest.raises(ValueError, match="engine_energy_frame_ms"):
            StreamTranscriber(
                engine=engine, vad_processor=vad,
                engine_energy_frame_ms=-1.0,
            )

    # === close() telemetry log ===

    def test_close_logs_dropped_counts(self, caplog):
        """close() 時に drop counter の内訳が logger.info で出力される。"""
        import logging
        caplog.set_level(logging.INFO, logger="livecap_cli.transcription.stream")

        engine = MockEngine()
        # 1 final_sync drop + 1 interim drop を発生させる
        final_seg = VADSegment(
            audio=self._quiet_audio(),
            start_time=0.0, end_time=0.1, is_final=True,
        )
        interim_seg = VADSegment(
            audio=self._quiet_audio(),
            start_time=0.1, end_time=0.6, is_final=False,
        )
        vad = MockVADProcessor(segments=[final_seg, interim_seg])
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=-45.0,
        )
        t.feed_audio(np.zeros(512, dtype=np.float32))
        t.feed_audio(np.zeros(512, dtype=np.float32))
        # counter pre-close
        assert t._dropped_low_energy_final_sync == 1
        assert t._dropped_low_energy_interim == 1

        t.close()

        # close() の log
        records = [r for r in caplog.records if "EnergyGate dropped" in r.message]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "1 final-sync" in msg
        assert "1 interim" in msg
        assert "max_frame_rms" in msg

    def test_close_no_log_when_opted_out(self, caplog):
        """opt-out 時は close() で log を出さない (drop=0 のため自明)。"""
        import logging
        caplog.set_level(logging.INFO, logger="livecap_cli.transcription.stream")

        engine = MockEngine()
        vad = MockVADProcessor()
        t = StreamTranscriber(
            engine=engine, vad_processor=vad,
            engine_min_rms_dbfs=float("-inf"),
        )
        t.close()
        assert not any("EnergyGate dropped" in r.message for r in caplog.records)

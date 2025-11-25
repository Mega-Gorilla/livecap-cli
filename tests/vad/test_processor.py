"""Unit tests for VADProcessor."""

import numpy as np

from livecap_core.vad import VADConfig, VADProcessor, VADState


class MockVADBackend:
    """テスト用モックバックエンド"""

    def __init__(self, probabilities: list[float] | None = None):
        self._probabilities = probabilities or []
        self._index = 0
        self._reset_called = False

    def process(self, audio: np.ndarray) -> float:
        """固定の確率を返す"""
        if self._index < len(self._probabilities):
            prob = self._probabilities[self._index]
            self._index += 1
            return prob
        return 0.0

    def reset(self) -> None:
        self._reset_called = True
        self._index = 0


class TestVADProcessorBasics:
    """VADProcessor 基本機能テスト"""

    def test_create_with_mock_backend(self):
        """モックバックエンドで作成"""
        backend = MockVADBackend()
        processor = VADProcessor(backend=backend)
        assert processor.state == VADState.SILENCE

    def test_constants(self):
        """定数"""
        assert VADProcessor.SAMPLE_RATE == 16000
        assert VADProcessor.FRAME_SAMPLES == 512


class TestVADProcessorProcessChunk:
    """process_chunk テスト"""

    def test_process_empty_audio(self):
        """空の音声"""
        backend = MockVADBackend()
        processor = VADProcessor(backend=backend)
        segments = processor.process_chunk(np.array([], dtype=np.float32))
        assert segments == []

    def test_process_short_audio(self):
        """短い音声（1フレーム未満）"""
        backend = MockVADBackend()
        processor = VADProcessor(backend=backend)
        audio = np.zeros(256, dtype=np.float32)  # 半フレーム
        segments = processor.process_chunk(audio)
        assert segments == []

    def test_process_single_frame(self):
        """1フレームの音声"""
        backend = MockVADBackend(probabilities=[0.3])
        processor = VADProcessor(backend=backend)
        audio = np.zeros(512, dtype=np.float32)
        segments = processor.process_chunk(audio)
        assert segments == []
        assert processor.state == VADState.SILENCE

    def test_process_multiple_frames(self):
        """複数フレームの音声"""
        # 10フレーム、全て低確率
        backend = MockVADBackend(probabilities=[0.3] * 10)
        processor = VADProcessor(backend=backend)
        audio = np.zeros(5120, dtype=np.float32)  # 10 frames
        segments = processor.process_chunk(audio)
        assert segments == []
        assert processor.state == VADState.SILENCE

    def test_detects_speech_segment(self):
        """音声セグメント検出"""
        # 高確率フレーム → 低確率フレーム でセグメント検出
        # min_speech_ms=64 (2 frames), min_silence_ms=64 (2 frames), speech_pad_ms=32 (1 frame)
        config = VADConfig(
            threshold=0.5,
            min_speech_ms=64,
            min_silence_ms=64,
            speech_pad_ms=32,
        )
        # 10 high prob → 10 low prob
        backend = MockVADBackend(probabilities=[0.7] * 10 + [0.3] * 10)
        processor = VADProcessor(config=config, backend=backend)

        audio = np.zeros(10240, dtype=np.float32)  # 20 frames
        segments = processor.process_chunk(audio)

        # セグメントが検出されるはず
        assert len(segments) >= 1
        assert segments[-1].is_final is True


class TestVADProcessorCurrentTime:
    """current_time プロパティテスト"""

    def test_current_time_increases(self):
        """処理時間が増加"""
        backend = MockVADBackend(probabilities=[0.3] * 10)
        processor = VADProcessor(backend=backend)

        assert processor.current_time == 0.0

        audio = np.zeros(512, dtype=np.float32)
        processor.process_chunk(audio)

        # 1フレーム = 32ms = 0.032s
        assert abs(processor.current_time - 0.032) < 0.001


class TestVADProcessorResidual:
    """残余バッファテスト"""

    def test_residual_carried_over(self):
        """残余が次のチャンクに引き継がれる"""
        backend = MockVADBackend(probabilities=[0.3] * 10)
        processor = VADProcessor(backend=backend)

        # 1.5 フレーム分の音声
        audio = np.zeros(768, dtype=np.float32)  # 512 + 256
        processor.process_chunk(audio)

        # 内部の残余バッファを確認（プライベートだが動作確認のため）
        assert processor._residual is not None
        assert len(processor._residual) == 256

        # 追加の音声で残余が処理される
        audio2 = np.zeros(256, dtype=np.float32)
        processor.process_chunk(audio2)

        # 残余 + 新音声 = 512 = 1 フレーム
        assert processor._residual is None


class TestVADProcessorFinalize:
    """finalize テスト"""

    def test_finalize_returns_remaining_segment(self):
        """finalize で残りセグメントを返す"""
        config = VADConfig(threshold=0.5, min_speech_ms=64)
        backend = MockVADBackend(probabilities=[0.7] * 10)
        processor = VADProcessor(config=config, backend=backend)

        audio = np.zeros(5120, dtype=np.float32)  # 10 frames
        processor.process_chunk(audio)

        assert processor.state == VADState.SPEECH

        segment = processor.finalize()
        assert segment is not None
        assert segment.is_final is True


class TestVADProcessorReset:
    """reset テスト"""

    def test_reset_clears_state(self):
        """reset で状態クリア"""
        config = VADConfig(threshold=0.5, min_speech_ms=64)
        backend = MockVADBackend(probabilities=[0.7] * 10)
        processor = VADProcessor(config=config, backend=backend)

        audio = np.zeros(5120, dtype=np.float32)
        processor.process_chunk(audio)

        assert processor.state == VADState.SPEECH
        assert processor.current_time > 0

        processor.reset()

        assert processor.state == VADState.SILENCE
        assert processor.current_time == 0.0
        assert backend._reset_called is True


class TestVADProcessorResampling:
    """リサンプリングテスト"""

    def test_resample_48khz(self):
        """48kHz からのリサンプリング"""
        backend = MockVADBackend(probabilities=[0.3] * 100)
        processor = VADProcessor(backend=backend)

        # 48kHz で 512 * 3 = 1536 samples
        audio_48k = np.zeros(1536, dtype=np.float32)
        processor.process_chunk(audio_48k, sample_rate=48000)

        # リサンプリング後は 512 samples = 1 frame
        assert abs(processor.current_time - 0.032) < 0.001

    def test_resample_44100hz(self):
        """44.1kHz からのリサンプリング"""
        backend = MockVADBackend(probabilities=[0.3] * 100)
        processor = VADProcessor(backend=backend)

        # 44.1kHz で ~1411 samples ≈ 512 @ 16kHz
        audio_44k = np.zeros(1411, dtype=np.float32)
        processor.process_chunk(audio_44k, sample_rate=44100)

        # 約1フレーム処理される
        assert processor.current_time >= 0.03

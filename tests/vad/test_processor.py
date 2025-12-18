"""Unit tests for VADProcessor."""

import numpy as np

from livecap_cli.vad import VADConfig, VADProcessor, VADState


class MockVADBackend:
    """テスト用モックバックエンド"""

    def __init__(
        self,
        probabilities: list[float] | None = None,
        frame_size: int = 512,
        name: str = "mock",
    ):
        self._probabilities = probabilities or []
        self._index = 0
        self._reset_called = False
        self._frame_size = frame_size
        self._name = name

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

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def name(self) -> str:
        return self._name


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

    def test_frame_size_from_backend(self):
        """フレームサイズをバックエンドから取得"""
        backend = MockVADBackend(frame_size=320)
        processor = VADProcessor(backend=backend)
        assert processor.frame_size == 320

    def test_backend_name(self):
        """バックエンド名を取得"""
        backend = MockVADBackend(name="test_backend")
        processor = VADProcessor(backend=backend)
        assert processor.backend_name == "test_backend"


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


class TestVADProcessorFromLanguage:
    """from_language ファクトリメソッドのテスト"""

    def test_from_language_ja_uses_tenvad(self):
        """日本語はTenVADを使用"""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # TenVAD license warning
            processor = VADProcessor.from_language("ja")
        assert "tenvad" in processor.backend_name

    def test_from_language_en_uses_webrtc(self):
        """英語はWebRTCを使用"""
        processor = VADProcessor.from_language("en")
        assert "webrtc" in processor.backend_name

    def test_from_language_unsupported_raises_valueerror(self):
        """未サポート言語はValueError"""
        import pytest

        with pytest.raises(ValueError, match="No optimized preset"):
            VADProcessor.from_language("zh")

    def test_from_language_en_applies_all_vad_config_params(self):
        """英語: vad_configの全パラメータがVADConfigに適用される"""
        from livecap_cli.vad.presets import get_best_vad_for_language

        processor = VADProcessor.from_language("en")

        # プリセットから期待値を取得
        _, preset = get_best_vad_for_language("en")
        expected = preset["vad_config"]

        # 全パラメータを検証
        assert processor.config.min_speech_ms == expected["min_speech_ms"]
        assert processor.config.min_silence_ms == expected["min_silence_ms"]
        assert processor.config.speech_pad_ms == expected["speech_pad_ms"]

    def test_from_language_en_applies_backend_params(self):
        """英語: backendパラメータがWebRTCに適用される"""
        from livecap_cli.vad.presets import get_best_vad_for_language

        processor = VADProcessor.from_language("en")

        # プリセットから期待値を取得
        _, preset = get_best_vad_for_language("en")
        expected_backend = preset["backend"]

        # バックエンドのconfigプロパティで検証
        backend_config = processor._backend.config
        assert backend_config["mode"] == expected_backend["mode"]
        assert backend_config["frame_duration_ms"] == expected_backend["frame_duration_ms"]

    def test_from_language_ja_applies_all_vad_config_params(self):
        """日本語: vad_configの全パラメータがVADConfigに適用される"""
        import warnings

        from livecap_cli.vad.presets import get_best_vad_for_language

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            processor = VADProcessor.from_language("ja")

        # プリセットから期待値を取得
        _, preset = get_best_vad_for_language("ja")
        expected = preset["vad_config"]

        # 全パラメータを検証
        assert processor.config.threshold == expected["threshold"]
        assert processor.config.neg_threshold == expected["neg_threshold"]
        assert processor.config.min_speech_ms == expected["min_speech_ms"]
        assert processor.config.min_silence_ms == expected["min_silence_ms"]
        assert processor.config.speech_pad_ms == expected["speech_pad_ms"]

    def test_from_language_ja_applies_backend_params(self):
        """日本語: backendパラメータがTenVADに適用される"""
        import warnings

        from livecap_cli.vad.presets import get_best_vad_for_language

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            processor = VADProcessor.from_language("ja")

        # プリセットから期待値を取得
        _, preset = get_best_vad_for_language("ja")
        expected_backend = preset["backend"]

        # バックエンドのconfigプロパティで検証
        backend_config = processor._backend.config
        assert backend_config["hop_size"] == expected_backend["hop_size"]

        # frame_sizeもhop_sizeと一致
        assert processor.frame_size == expected_backend["hop_size"]

    def test_from_language_error_message_includes_supported_languages(self):
        """エラーメッセージにサポート言語が含まれる"""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            VADProcessor.from_language("fr")

        error_message = str(exc_info.value)
        assert "en" in error_message
        assert "ja" in error_message
        assert "VADProcessor()" in error_message


class TestVADProcessorCreateBackend:
    """_create_backend ヘルパーメソッドのテスト"""

    def test_create_backend_unknown_type_raises_valueerror(self):
        """未知のVADタイプはValueError"""
        import pytest

        with pytest.raises(ValueError, match="Unknown VAD type"):
            VADProcessor._create_backend("unknown", {})

    def test_create_backend_silero(self):
        """Sileroバックエンドの作成"""
        import pytest

        try:
            backend = VADProcessor._create_backend("silero", {})
            assert backend.name == "silero"
            assert backend.frame_size == 512
        except (ImportError, RuntimeError, AttributeError) as e:
            # Silero requires torch/torchaudio - skip if not available or version mismatch
            # AttributeError can occur during circular import with CUDA version conflicts
            pytest.skip(f"Silero VAD not available: {e}")

    def test_create_backend_webrtc(self):
        """WebRTCバックエンドの作成"""
        backend = VADProcessor._create_backend(
            "webrtc", {"mode": 1, "frame_duration_ms": 30}
        )
        assert "webrtc" in backend.name
        # 30ms @ 16kHz = 480 samples
        assert backend.frame_size == 480

    def test_create_backend_tenvad(self):
        """TenVADバックエンドの作成"""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            backend = VADProcessor._create_backend("tenvad", {"hop_size": 256})
        assert backend.name == "tenvad"
        assert backend.frame_size == 256

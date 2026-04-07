"""NoiseGate のユニットテスト + ベンチマーク。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from livecap_cli.audio.noise_gate import NoiseGate, _process_loop


# === 初期化テスト ===


class TestNoiseGateInit:
    """パラメータ検証テスト。"""

    def test_default_parameters(self):
        ng = NoiseGate()
        assert ng._threshold == pytest.approx(10 ** (-35 / 20))
        assert ng._envelope == 0.0
        assert ng._gate_open == 0
        assert ng._release_counter == 0

    def test_custom_parameters(self):
        ng = NoiseGate(threshold_db=-50, attack_ms=1.0, release_ms=50, sample_rate=16000)
        assert ng._threshold == pytest.approx(10 ** (-50 / 20))

    def test_invalid_threshold_clamped(self):
        """範囲外の threshold はデフォルト値にフォールバック。"""
        ng = NoiseGate(threshold_db=-100)  # -100 < -80: invalid
        assert ng._threshold == pytest.approx(10 ** (-35 / 20))

    def test_invalid_attack_clamped(self):
        ng = NoiseGate(attack_ms=0.01)  # 0.01 < 0.1: invalid
        # Should fall back to 0.5ms
        expected_samples = max(1, int(0.5 * 16000 / 1000))
        assert ng._attack_samples == expected_samples

    def test_invalid_release_clamped(self):
        ng = NoiseGate(release_ms=2000)  # 2000 > 1000: invalid
        expected_samples = max(1, int(30 * 16000 / 1000))
        assert ng._release_samples == expected_samples


# === process() テスト ===


class TestNoiseGateProcess:
    """process() の基本動作テスト。"""

    def test_empty_chunk(self):
        ng = NoiseGate()
        result = ng.process(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_silence_is_attenuated(self):
        """無音入力はノイズフロアレベルに減衰される。"""
        ng = NoiseGate(threshold_db=-35)
        # 非常に小さい音（-60dB 以下）
        silence = np.full(1600, 0.0001, dtype=np.float32)
        output = ng.process(silence)

        # 出力は入力よりかなり小さい（noise_floor で減衰）
        assert np.max(np.abs(output)) < np.max(np.abs(silence))

    def test_loud_signal_passes_through(self):
        """閾値以上の大きな音はそのまま通過する。"""
        ng = NoiseGate(threshold_db=-35)
        # 大きな音（-10dB ≈ 0.316）
        loud = np.full(1600, 0.3, dtype=np.float32)
        output = ng.process(loud)

        # ゲートが開いた後の出力は入力と同じ
        # アタック時間が短い（0.5ms = 8 samples）ので、ほぼ全サンプルが通過
        assert np.allclose(output[10:], loud[10:], atol=1e-6)

    def test_gate_open_close_transition(self):
        """音 → 無音の遷移でゲートが閉じる。"""
        ng = NoiseGate(threshold_db=-35, release_ms=10, sample_rate=16000)

        # 大きな音 → 無音
        loud = np.full(800, 0.3, dtype=np.float32)
        silent = np.full(800, 0.0001, dtype=np.float32)
        chunk = np.concatenate([loud, silent])

        output = ng.process(chunk)

        # 前半はほぼパススルー
        assert np.mean(np.abs(output[:800])) > 0.1
        # 後半は減衰（ただしリリース時間分の遅延あり）
        tail = output[1200:]  # リリース後の部分
        assert np.mean(np.abs(tail)) < 0.01

    def test_preserves_waveform_shape(self):
        """ゲートが開いている間は波形が保持される。"""
        ng = NoiseGate(threshold_db=-35)
        # サイン波（十分な音量）
        t = np.arange(1600) / 16000
        sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        output = ng.process(sine)

        # ゲートが開いた後の波形は入力と一致
        assert np.allclose(output[20:], sine[20:], atol=1e-5)

    def test_output_dtype_matches_input(self):
        ng = NoiseGate()
        audio = np.zeros(100, dtype=np.float32)
        output = ng.process(audio)
        assert output.dtype == audio.dtype


# === reset() テスト ===


class TestNoiseGateReset:
    def test_reset_clears_state(self):
        ng = NoiseGate()
        # 音を流してゲートを開く
        loud = np.full(1600, 0.3, dtype=np.float32)
        ng.process(loud)
        assert ng._gate_open == 1

        ng.reset()
        assert ng._envelope == 0.0
        assert ng._gate_open == 0
        assert ng._release_counter == 0


# === StreamTranscriber 統合テスト ===


class TestStreamTranscriberIntegration:
    """StreamTranscriber の noise_gate パラメータ統合テスト。"""

    def test_none_noise_gate_is_default(self):
        """noise_gate=None で従来動作。"""
        from livecap_cli.transcription.stream import StreamTranscriber
        from tests.transcription.test_stream import MockEngine, MockVADProcessor

        engine = MockEngine(return_text="テスト結果テキスト")
        vad = MockVADProcessor()
        transcriber = StreamTranscriber(
            engine=engine, vad_processor=vad, noise_gate=None
        )
        assert transcriber._noise_gate is None

    def test_noise_gate_applied_in_feed_audio(self):
        """noise_gate が feed_audio で適用される。"""
        from livecap_cli.transcription.stream import StreamTranscriber
        from livecap_cli.vad import VADSegment
        from tests.transcription.test_stream import MockEngine, MockVADProcessor

        engine = MockEngine(return_text="テスト結果テキスト")
        segment = VADSegment(
            audio=np.zeros(1600, dtype=np.float32),
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        vad = MockVADProcessor(segments=[segment])

        # NoiseGate を process 呼び出し回数でトラッキング
        ng = NoiseGate()
        process_calls = []
        original_process = ng.process

        def tracking_process(chunk):
            process_calls.append(len(chunk))
            return original_process(chunk)

        ng.process = tracking_process

        transcriber = StreamTranscriber(
            engine=engine, vad_processor=vad, noise_gate=ng
        )
        transcriber.feed_audio(np.zeros(512, dtype=np.float32))

        assert len(process_calls) == 1


# === ベンチマーク ===


class TestNoiseGateBenchmark:
    """パフォーマンスベンチマーク。"""

    def test_numba_performance(self):
        """numba JIT 版が 1ms/chunk 以下であることを確認。"""
        ng = NoiseGate()
        chunk = np.random.randn(1600).astype(np.float32) * 0.1

        # warmup (JIT コンパイル)
        for _ in range(3):
            ng.process(chunk)

        # ベンチマーク
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            ng.process(chunk)
        elapsed = time.perf_counter() - start

        ms_per_chunk = (elapsed / iterations) * 1000
        print(f"\nNoiseGate numba: {ms_per_chunk:.4f} ms/chunk ({iterations} iterations)")

        # 1ms/chunk 以下（保守的な閾値）
        assert ms_per_chunk < 1.0, f"Too slow: {ms_per_chunk:.4f} ms/chunk"

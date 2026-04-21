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
        # PR B: close_threshold_db=None は threshold - 6 dB に解決
        assert ng._close_threshold == pytest.approx(10 ** (-41 / 20))
        # PR B: noise_floor_db=-inf が既定 → hard-mute (noise_floor = 0.0)
        assert ng._noise_floor == 0.0
        assert ng._envelope == 0.0
        assert ng._gate_open == 0
        assert ng._release_counter == 0

    def test_custom_parameters(self):
        ng = NoiseGate(threshold_db=-50, attack_ms=1.0, release_ms=50, sample_rate=16000)
        assert ng._threshold == pytest.approx(10 ** (-50 / 20))
        # auto hysteresis: threshold - 6 = -56 dB
        assert ng._close_threshold == pytest.approx(10 ** (-56 / 20))

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
        # PR C (Issue #283): fallback は 100 ms (既定と一致)
        expected_samples = max(1, int(100 * 16000 / 1000))
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


# === ヒステリシス / hard-mute テスト (PR B / Issue #280 C-1, C-2) ===


class TestHysteresis:
    """close_threshold_db によるヒステリシス挙動のテスト。"""

    def test_auto_close_threshold_default(self):
        """close_threshold_db=None → threshold_db - 6 dB に自動解決。"""
        ng = NoiseGate(threshold_db=-35)
        assert ng._close_threshold == pytest.approx(10 ** (-41 / 20))

    def test_explicit_close_threshold(self):
        """明示的 close_threshold_db が使われる。"""
        ng = NoiseGate(threshold_db=-35, close_threshold_db=-50)
        assert ng._close_threshold == pytest.approx(10 ** (-50 / 20))

    def test_legacy_single_threshold_opt_in(self):
        """close_threshold_db=threshold_db で legacy 単一閾値挙動 (ヒステリシスなし)。"""
        ng = NoiseGate(threshold_db=-35, close_threshold_db=-35)
        assert ng._threshold == ng._close_threshold

    def test_close_threshold_above_open_clamped(self, caplog):
        """close > open は警告 + threshold_db に clamp (= 単一閾値)。"""
        import logging

        with caplog.at_level(logging.WARNING):
            ng = NoiseGate(threshold_db=-35, close_threshold_db=-20)
        assert "close_threshold" in caplog.text.lower()
        assert ng._close_threshold == ng._threshold

    def test_close_threshold_below_minus_80_clamped(self, caplog):
        """close < -80 は警告 + -80 に clamp。"""
        import logging

        with caplog.at_level(logging.WARNING):
            ng = NoiseGate(threshold_db=-35, close_threshold_db=-100)
        assert "close_threshold" in caplog.text.lower()
        assert ng._close_threshold == pytest.approx(10 ** (-80 / 20))

    def test_hysteresis_prevents_flicker(self):
        """threshold 付近を振動する envelope でも gate 状態が安定 (flicker 抑制)。

        open=-35, close=-41 (デフォルト 6 dB band) に対し、入力を
        -34 dB と -37 dB で振動させる。-37 dB は close (-41) を
        下回らないため、ゲートは一度開いたら閉じない。
        """
        ng = NoiseGate(threshold_db=-35, release_ms=5, sample_rate=16000)
        # -34 dB ≈ 0.02, -37 dB ≈ 0.0141 (両方 close=-41 より上)
        hi = float(10 ** (-34 / 20))
        lo = float(10 ** (-37 / 20))
        samples = np.array([hi, lo] * 800, dtype=np.float32)
        ng.process(samples)
        # 振動しても最後まで開いたまま
        assert ng._gate_open == 1

    def test_hysteresis_allows_clean_close(self):
        """close_threshold をしっかり下回れば gate は閉じる。"""
        ng = NoiseGate(threshold_db=-35, release_ms=5, sample_rate=16000)
        # 前半: -10 dB で gate 開く
        loud = np.full(800, 0.3, dtype=np.float32)
        # 後半: -80 dB で gate close (close=-41 より十分下)
        silent = np.full(1600, 0.0001, dtype=np.float32)
        samples = np.concatenate([loud, silent])
        ng.process(samples)
        # release + envelope decay 後は閉じる
        assert ng._gate_open == 0


class TestHardMute:
    """noise_floor_db パラメータ化と hard-mute のテスト。"""

    def test_hard_mute_default(self):
        """既定 (noise_floor_db=-inf) で hard-mute、ゲート閉鎖時の出力は完全ゼロ。"""
        ng = NoiseGate(threshold_db=-35)
        assert ng._noise_floor == 0.0
        # 閾値以下の無音入力 → gate 閉鎖 → 出力ゼロ
        silence = np.full(1600, 0.0001, dtype=np.float32)
        output = ng.process(silence)
        assert np.all(output == 0.0)

    def test_soft_mute_opt_in_negative_60(self):
        """noise_floor_db=-60 で legacy soft-mute (出力 = 入力 × 0.001)。"""
        ng = NoiseGate(threshold_db=-35, noise_floor_db=-60)
        assert ng._noise_floor == pytest.approx(10 ** (-60 / 20))
        # 閾値以下の無音入力 → gate 閉鎖 → 出力 = 入力 × 0.001
        silence = np.full(1600, 0.0001, dtype=np.float32)
        output = ng.process(silence)
        # すべて hard-mute ではない (ゼロより大きい値が残る)
        assert np.max(np.abs(output)) > 0.0
        assert np.allclose(output, silence * ng._noise_floor, atol=1e-8)

    def test_inf_explicit_hard_mute(self):
        """float('-inf') を明示指定しても hard-mute。"""
        ng = NoiseGate(threshold_db=-35, noise_floor_db=float("-inf"))
        assert ng._noise_floor == 0.0

    def test_noise_floor_out_of_range_warning(self, caplog):
        """noise_floor_db > 0 は警告 + fallback to hard-mute。"""
        import logging

        with caplog.at_level(logging.WARNING):
            ng = NoiseGate(threshold_db=-35, noise_floor_db=10)
        assert "noise_floor" in caplog.text.lower()
        assert ng._noise_floor == 0.0

    def test_noise_floor_too_low_warning(self, caplog):
        """noise_floor_db < -120 は警告 + fallback to hard-mute。"""
        import logging

        with caplog.at_level(logging.WARNING):
            ng = NoiseGate(threshold_db=-35, noise_floor_db=-150)
        assert "noise_floor" in caplog.text.lower()
        assert ng._noise_floor == 0.0

    def test_gate_open_not_affected_by_noise_floor(self):
        """ゲート開放時は noise_floor に関わらず入力をパススルー。"""
        ng = NoiseGate(threshold_db=-35, noise_floor_db=float("-inf"))
        loud = np.full(1600, 0.3, dtype=np.float32)
        output = ng.process(loud)
        # 開いてからは loud そのまま (hard-mute 設定でも影響なし)
        assert np.allclose(output[50:], loud[50:], atol=1e-6)


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

"""NoiseGate calibration の end-to-end 回帰テスト (issue #291)。

synthetic impulsive noise を ``analyze_noise_samples()`` に通し、
旧 threshold (chunk RMS p95 + 10 dB) では NoiseGate が頻繁に開く一方で
新 threshold (peak_p95 + 6 dB) では gate が閉じ続けることを assert する。

これが本 issue #291 の bug を直接 encode する最終防衛線。
"""

from __future__ import annotations

import numpy as np
import pytest

from livecap_cli.audio import NoiseGate, analyze_noise_samples

SR = 16000
CHUNK = 1600  # 100 ms


def _synthesize_burst_noise(
    duration_s: float = 5.0,
    base_rms: float = 0.0032,  # ~-50 dBFS RMS
    burst_amp: float = 0.05,  # ~-26 dBFS peak
    burst_len_samples: int = 30,
    burst_interval_s: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """白色ノイズ底 + 周期的バースト (キーボード/呼吸を模擬)。

    crest factor が大きい (約 24 dB) ため、旧 RMS-based 推奨閾値では
    envelope が容易に超え、新 peak-based 推奨閾値では超えない。
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    audio = (rng.standard_normal(n) * base_rms).astype(np.float32)
    for start in range(0, n, int(burst_interval_s * SR)):
        end = min(start + burst_len_samples, n)
        if end > start:
            sign = rng.choice([-1, 1], size=end - start).astype(np.float32)
            audio[start:end] += burst_amp * sign
    return audio


def _collect_per_chunk(audio: np.ndarray) -> tuple[list[float], list[float]]:
    """``cmd_levels`` と同じやり方で per-chunk RMS と peak を dB 列に変換。"""
    rms_db: list[float] = []
    peak_db: list[float] = []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        c = audio[i : i + CHUNK]
        rms = float(np.sqrt(np.mean(c**2)))
        peak = float(np.max(np.abs(c)))
        rms_db.append(20 * np.log10(max(rms, 1e-10)))
        peak_db.append(20 * np.log10(max(peak, 1e-10)))
    return rms_db, peak_db


def _frac_gate_open(gate: NoiseGate, audio: np.ndarray) -> float:
    """hard-mute 既定 (noise_floor_db=-inf) では output==0 ⟺ gate closed。

    入力が至るところ non-zero なら、output が non-zero な比率が
    そのまま「gate が開いていた sample 比率」になる。
    """
    out = gate.process(audio.copy())
    return float(np.count_nonzero(out)) / len(out)


@pytest.fixture(scope="module")
def burst_noise() -> np.ndarray:
    return _synthesize_burst_noise()


@pytest.fixture(scope="module")
def analysis(burst_noise: np.ndarray):
    rms_db, peak_db = _collect_per_chunk(burst_noise)
    return analyze_noise_samples(rms_db, peak_db)


class TestNoiseGateCalibrationRegression:
    """#291 root cause regression: unit mismatch causing hallucinations."""

    def test_old_threshold_opens_gate_on_impulsive_noise(
        self, burst_noise: np.ndarray, analysis
    ) -> None:
        """旧バグ算法 (RMS p95 + 10 dB) を再現し、gate が開くことを確認。

        これが「無音時 hallucination」の物理的根拠 — envelope が
        threshold を超えるため。
        """
        old_threshold = analysis.noise_rms_p95_db + 10.0
        gate = NoiseGate(threshold_db=old_threshold)
        frac_open = _frac_gate_open(gate, burst_noise)
        assert frac_open > 0.05, (
            f"旧 threshold ({old_threshold:.1f} dB) で gate が想定通り開いていない "
            f"(open frac={frac_open:.3f}). 回帰テストの前提が崩れている可能性。"
        )

    def test_new_threshold_keeps_gate_closed_on_impulsive_noise(
        self, burst_noise: np.ndarray, analysis
    ) -> None:
        """新算法 (peak_p95 + 6 dB) では gate が閉じ続けることを確認。

        本 issue の修正が機能していることの最終確認。
        """
        gate = NoiseGate(threshold_db=analysis.suggested_threshold_db)
        frac_open = _frac_gate_open(gate, burst_noise)
        assert frac_open < 0.02, (
            f"新 suggested ({analysis.suggested_threshold_db:.1f} dB) で gate が "
            f"閉じきっていない (open frac={frac_open:.3f}). margin 不足の可能性。"
        )

    def test_speech_still_opens_new_threshold(self, analysis) -> None:
        """正規の speech は新 threshold でも開く (over-margin で死んでいない)。"""
        speech = (
            np.random.default_rng(0).standard_normal(SR) * 0.1
        ).astype(np.float32)
        gate = NoiseGate(threshold_db=analysis.suggested_threshold_db)
        frac_open = _frac_gate_open(gate, speech)
        assert frac_open > 0.9, (
            f"新 suggested ({analysis.suggested_threshold_db:.1f} dB) で speech が "
            f"通過していない (open frac={frac_open:.3f}). over-margin の可能性。"
        )

    def test_new_threshold_is_higher_than_old(self, analysis) -> None:
        """新 suggested が旧 suggested より上にある (root-cause の方向性)。"""
        old_suggested = analysis.noise_rms_p95_db + 10.0
        assert analysis.suggested_threshold_db > old_suggested
